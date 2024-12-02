import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import trange, tqdm

import comfy
from comfy import model_management
from comfy import utils
from comfy.supported_models import supported_models_base
from comfy.model_detection import count_blocks
from comfy.k_diffusion.sampling import default_noise_sampler
from . import animemory_clip,animemory_vae


bak_unet_config_from_diffusers_unet = comfy.model_detection.unet_config_from_diffusers_unet

Animemory_TYPE = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
        'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': None, 'in_channels': 4, 'model_channels': 320,
        'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
        'use_linear_in_transformer': True, 'context_dim': 5376, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
        'use_temporal_attention': False, 'use_temporal_resblock': False}

class Animemory(comfy.supported_models.SDXL):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 5376,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }
    def clip_target(self, **kwargs):
        return supported_models_base.ClipTarget(animemory_clip.AnimemoryTokenizer, animemory_clip.AnimemoryClipModel)

    def process_clip_state_dict(self, state_dict):
        state_dict = utils.state_dict_prefix_replace(state_dict,{"conditioner.embedders.1":"clip_g"})
        state_dict = utils.state_dict_prefix_replace(state_dict,{"clip_g.model":"clip_g.transformer"})
        state_dict = utils.state_dict_prefix_replace(state_dict,{"conditioner.embedders.0":"clip_l"})
        out = {}
        ks = list(state_dict.keys())
        for k in ks:
            out[k] = state_dict.pop(k)
        return out


def hook_config(func):
    @functools.wraps(func)
    def forward(*args,**kwargs):
        ret = func(*args, **kwargs)
        if ret is not None:
            return ret
        
        state_dict = args[0]
        match = {}
        transformer_depth = []
        attn_res = 1
        down_blocks = count_blocks(state_dict, "down_blocks.{}")
        for i in range(down_blocks):
            attn_blocks = count_blocks(state_dict, "down_blocks.{}.attentions.".format(i) + '{}')
            res_blocks = count_blocks(state_dict, "down_blocks.{}.resnets.".format(i) + '{}')
            for ab in range(attn_blocks):
                transformer_count = count_blocks(state_dict, "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + '{}')
                transformer_depth.append(transformer_count)
                if transformer_count > 0:
                    match["context_dim"] = state_dict["down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(i, ab)].shape[1]

            attn_res *= 2
            if attn_blocks == 0:
                for i in range(res_blocks):
                    transformer_depth.append(0)

        match["transformer_depth"] = transformer_depth

        match["model_channels"] = state_dict["conv_in.weight"].shape[0]
        match["in_channels"] = state_dict["conv_in.weight"].shape[1]
        match["adm_in_channels"] = None
        if "class_embedding.linear_1.weight" in state_dict:
            match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
        elif "add_embedding.linear_1.weight" in state_dict:
            match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

        matches = True
        Animemory_TYPE['dtype'] = kwargs.get("dtype", None)
        for k in match:
            if match[k] != Animemory_TYPE[k]:
                matches = False
                break
        if matches:
            return comfy.model_detection.convert_config(Animemory_TYPE)
        return None
    return forward

if Animemory not in comfy.supported_models.models:
    comfy.supported_models.models.append(Animemory)
comfy.model_detection.unet_config_from_diffusers_unet = hook_config(comfy.model_detection.unet_config_from_diffusers_unet)

animemory_vae_param ={"embed_dim":4,
            "ddconfig":{ 
            "double_z": False,
            "z_channels": 4,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 256,
            "ch_mult": [1,2,2,4],
            "num_res_blocks": 2,
            "attn_resolutions": [32],
            "dropout": 0.0,
            }
        } 
def hook_vae(func):
    def run(*args,**kwargs):
        if 'decoder.mid.attn_1.norm.conv_b.bias' in kwargs['sd']:
            self = args[0]
            self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype) 
            self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype)
            self.downscale_ratio = 8
            self.upscale_ratio = 8
            self.latent_channels = 4
            self.output_channels = 3
            self.process_input = lambda image: image * 2.0 - 1.0
            self.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            self.first_stage_model = animemory_vae.MoVQ(**animemory_vae_param)
            self.first_stage_model = self.first_stage_model.eval()

            m, u = self.first_stage_model.load_state_dict(kwargs['sd'], strict=False)

            device = model_management.vae_device()
            self.device = device
            offload_device = model_management.vae_offload_device()
            dtype = model_management.vae_dtype()
            self.vae_dtype = dtype
            self.first_stage_model.to(self.vae_dtype)
            self.output_device = model_management.intermediate_device()

            self.patcher = comfy.model_patcher.ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
        else:
            func(*args,**kwargs)
    return run

comfy.sd.VAE.__init__ = hook_vae(comfy.sd.VAE.__init__)

def ret_noise(func):
    def run(*args,**kwargs):
        return args[1]
    return run

euler_xpred_enable = False

def euler_a_xpred(func):
    global euler_xpred_enable
    def run(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
        global euler_xpred_enable
        if not euler_xpred_enable:
            return func(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            
            sigma_t = sigmas[i]
            sigma_s = sigmas[i + 1]
            alpha_t = (1 - sigma_t**2)**0.5
            alpha_s = (1 - sigma_s**2)**0.5

            coef_sample = (sigma_s / sigma_t)**2 * alpha_t / alpha_s
            coef_noise = (sigma_s / sigma_t) * (1 - (alpha_t / alpha_s)**2)**0.5
            coef_x = alpha_s * (1 - alpha_t**2 / alpha_s**2) / sigma_t**2
            x = coef_sample * x + coef_x * denoised + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * coef_noise
        return x
    return run

comfy.k_diffusion.sampling.sample_euler_ancestral = euler_a_xpred(comfy.k_diffusion.sampling.sample_euler_ancestral)


def hook_scale(func):
    global euler_xpred_enable
    def run(*arg,**kwargs):
        global euler_xpred_enable
        ret = func(*arg,**kwargs)
        if isinstance(ret[2].first_stage_model,animemory_vae.MoVQ):
            euler_xpred_enable = True
            ret[0].model.latent_format.scale_factor = 1
            _sgfile = Path(__file__).parent/"utils/sigmas.dat"
            _sg = torch.load(_sgfile,map_location="cpu")
            ret[0].model.model_sampling.sigmas = _sg.float()
            ret[0].model.model_sampling.log_sigmas = _sg.log().float()
            ret[0].model.model_sampling.calculate_input = ret_noise(ret[0].model.model_sampling.calculate_input)
            ret[0].model.model_sampling.noise_scaling = ret_noise(ret[0].model.model_sampling.noise_scaling)
            ret[0].model.model_sampling.calculate_denoised = ret_noise(ret[0].model.model_sampling.calculate_denoised)
        else:
            euler_xpred_enable = False
        return ret
    return run

comfy.sd.load_checkpoint_guess_config = hook_scale(comfy.sd.load_checkpoint_guess_config)

class AnimemoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
        }

    RETURN_TYPES = ()
    # RETURN_TYPES = ("TEXT",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"
    CATEGORY = "Example"

    def test(self, model, mode):
        pass

NODE_CLASS_MAPPINGS = {"AnimemoryNode": AnimemoryNode}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"AnimemoryNode": "AnimemoryNode"}