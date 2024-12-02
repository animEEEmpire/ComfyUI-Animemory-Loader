#! python3
# -*- encoding: utf-8 -*-

import os
import json
import math
import cv2
import torch
import PIL
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

from packaging import version
from PIL import Image
from safetensors.torch import load_file
from typing import Optional, Union, List
from types import SimpleNamespace

to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

def preprocess_image(image, w, h):
    image = image.convert("RGB")
    image_array = np.asarray(image, dtype=np.uint8)
    image_array = cv2.resize(image_array, (w, h), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(image_array)
    init_image = to_tensor(image) * 2 - 1
    return image, init_image.unsqueeze(0)


def postprocess_image(image):
    image = torch.clip((image + 1.) / 2., 0., 1.).cpu()
    pil_images = list()
    for images_chunk in image.chunk(1):
        pil_images += [to_pil(image) for image in images_chunk]
    return pil_images


def cal_outimage_w_h(input_image, base_size=1024):
    w, h = input_image.size
    ratio = 2 * base_size / ( w+h )
    out_w, out_h = map(lambda x: max((x - x % 8), 8), (round(w*ratio), round(h*ratio)))  # resize to integer multiple of 64
    return out_w, out_h


def nonlinearity(x):
    return x*torch.sigmoid(x)


class SpatialNorm(nn.Module):
    def __init__(
        self, f_channels, zq_channels=None, norm_layer=nn.GroupNorm, freeze_norm_layer=False, add_conv=False, **norm_layer_params
    ):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if zq_channels is not None:
            if freeze_norm_layer:
                for p in self.norm_layer.parameters:
                    p.requires_grad = False
            self.add_conv = add_conv
            if self.add_conv:
                self.conv = nn.Conv2d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
            self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
            self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, f, zq=None):
        norm_f = self.norm_layer(f)
        if zq is not None:
            f_size = f.shape[-2:]
            if version.parse(torch.__version__) < version.parse('2.1') and zq.dtype == torch.bfloat16:
                zq = zq.to(dtype=torch.float32)
                zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
                zq = zq.to(dtype=torch.bfloat16)
            else:
                zq = torch.nn.functional.interpolate(zq, size=f_size, mode="nearest")
            if self.add_conv:
                zq = self.conv(zq)
            norm_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return norm_f


def Normalize(in_channels, zq_ch=None, add_conv=None):
    return SpatialNorm(
            in_channels, zq_ch, norm_layer=nn.GroupNorm,
            freeze_norm_layer=False, add_conv=add_conv, num_groups=32, eps=1e-6, affine=True
        )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        if version.parse(torch.__version__) < version.parse('2.1') and x.dtype == torch.bfloat16:
            x = x.to(dtype=torch.float32)
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            x = x.to(dtype=torch.bfloat16)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels, zq_ch, add_conv=add_conv)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb, zq=None):
        h = x
        h = self.norm1(h, zq)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h, zq)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels, zq_ch, add_conv=add_conv)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, zq=None):
        h_ = x
        h_ = self.norm(h_, zq)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_
    
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, zq_ch=None, add_conv=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       zq_ch=zq_ch,
                                       add_conv=add_conv)
        self.mid.attn_1 = AttnBlock(block_in, zq_ch, add_conv=add_conv)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       zq_ch=zq_ch,
                                       add_conv=add_conv)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         zq_ch=zq_ch,
                                         add_conv=add_conv))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in, zq_ch, add_conv=add_conv))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, zq_ch, add_conv=add_conv)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, zq):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    
class MoVQ(nn.Module):
    
    def __init__(self, embed_dim: int, **kwargs):
        super().__init__()
        ddconfig = kwargs.pop("ddconfig")
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", ())
        z_channels = ddconfig["z_channels"]
        self.config = SimpleNamespace(**ddconfig)
        self.encoder = Encoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder(zq_ch=z_channels, **ddconfig)
        self.embed_dim = embed_dim
        self.dtype = None
        self.device = None
    

    @staticmethod
    def get_model_config(pretrained_model_name_or_path, subfolder):
        config_path = os.path.join(pretrained_model_name_or_path, subfolder, "config.json")
        assert os.path.exists(config_path), "config file not exists."
        with open(config_path, "r") as f:
            config = json.loads(f.read())
        return config


    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        subfolder="",
        torch_dtype=torch.float32,
    ):
        config = cls.get_model_config(pretrained_model_name_or_path, subfolder)
        model = cls(generator_params=config)
        ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, "model.pt")
        assert os.path.exists(ckpt_path), f"ckpt path not exists, please check {ckpt_path}"
        assert torch_dtype != torch.float16, "torch_dtype doesn't support fp16"
        ckpt_weight = torch.load(ckpt_path)
        model.load_state_dict(ckpt_weight, strict=False)
        model.to(dtype=torch_dtype)
        return model


    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        super(MoVQ, self).to(*args, **kwargs)
        self.dtype = dtype if dtype is not None else self.dtype
        self.device = device if device is not None else self.device
        return self


    @torch.no_grad()
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h


    @torch.no_grad()
    def decode(self, quant):
        decoder_input = self.post_quant_conv(quant)
        decoded = self.decoder(decoder_input, quant)
        return decoded


if __name__ == "__main__":

    device = "cuda:0"
    dtype = torch.bfloat16

    vae_pretrained_model_name_or_path = "vae/movq"
    test_img_path = "photo.jpg"
    test_img = Image.open(test_img_path).convert("RGB")
    w, h = cal_outimage_w_h(test_img)
    test_img, init_image = preprocess_image(test_img, w, h)
    init_image = init_image.to(device).to(dtype)

    movq = MoVQ.from_pretrained(vae_pretrained_model_name_or_path, torch_dtype=dtype)
    movq.to(device)
    movq.eval()

    lantent_codes = movq.encode(init_image)
    out_images = movq.decode(lantent_codes)

    generate_images = postprocess_image(out_images)
    generate_images[0].save("aftermovq.jpg")
    breakpoint()