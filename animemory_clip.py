import os

from comfy import sd1_clip
import torch
from transformers import (
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    ByT5Tokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    AltCLIPTextModel,
    XLMRobertaTokenizerFast,
    T5Config,
    T5ForConditionalGeneration
)
from transformers.models.t5.modeling_t5 import T5Stack

curpath = os.path.dirname(os.path.abspath(__file__))

class T5Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=None, max_length=227, pad_with_end=True, embedding_directory=None, embedding_size=4096, embedding_key='clip_l', tokenizer_class=CLIPTokenizer, has_start_token=True, pad_to_max_length=True, min_length=None):
        super().__init__()
        # if tokenizer_path is None:
        #     tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(os.path.join(curpath,"tokenizer"))
        self.max_length = max_length
        self.min_length = min_length

        empty = self.tokenizer('')["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

        self.pad_token = 2

class ALTCLIPTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=None, max_length=227, pad_with_end=True, embedding_directory=None, embedding_size=1280, embedding_key='clip_g', tokenizer_class=CLIPTokenizer, has_start_token=True, pad_to_max_length=True, min_length=None):
        super().__init__()
        # if tokenizer_path is None:
        #     tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sd1_tokenizer")
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(os.path.join(curpath,"tokenizer"))
        self.max_length = max_length
        self.min_length = min_length

        empty = self.tokenizer('')["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length

        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

        self.pad_token = 2

class AnimemoryTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = T5Tokenizer()
        self.clip_g = ALTCLIPTokenizer()

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

class my_t5_encoder(torch.nn.Module):
    def __init__(self, config: T5Config, embed_tokens=None):
        super().__init__()
        self.encoder = T5Stack(config, embed_tokens)
        self.embed_tokens_encoder = torch.nn.Embedding(250002, 4096, padding_idx=1)


    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        subfolder="",
        embed_tokens=None,
        emb_name='embed_tokens_encoder.pt', 
        torch_dtype=torch.float16,
    ):

        config = T5Stack.config_class.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder)
        model = cls(config=config, embed_tokens=embed_tokens)
        model.encoder = T5Stack.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        embed_tokens_encoder_path = torch.load(os.path.join(pretrained_model_name_or_path, subfolder, emb_name))
        model.embed_tokens_encoder.load_state_dict(embed_tokens_encoder_path)
        model.encoder.to(torch_dtype)
        model.embed_tokens_encoder.to(torch_dtype)
        return model


    def encode_text_clip(self, input, attention_mask):
        pass


    def make_attn_mask(self, attn_mask):
        seq_len = attn_mask.shape[1]
        query = attn_mask.unsqueeze(1).float()
        # target = query.permute([0, 2, 1]).contiguous().float()
        attn_mask = query.repeat([1, seq_len, 1]).unsqueeze(1).repeat([1, self.num_head, 1, 1])
        # attn_mask = target.bmm(query).unsqueeze(1).repeat([1, self.num_head, 1, 1])
        attn_mask = attn_mask.view([-1, seq_len, seq_len])
        return attn_mask


    def forward(self, text, attention_mask):
        embeddings = self.embed_tokens_encoder(text)
        encoder_outputs = self.encoder(inputs_embeds=embeddings, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states[-2]
        hidden_states = self.encoder.final_layer_norm(hidden_states)
        return hidden_states

class myaltclip(torch.nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.model_hf = CLIPTextModelWithProjection(config)
        self.linear_proj = torch.nn.Linear(in_features=1280, out_features=1280)

    # def from_pretrained(self, pretrained_model_name_or_path, subfolder=""):
    #     self.model_hf = self.model_hf.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
    #     linear_proj_state = torch.load(os.path.join(pretrained_model_name_or_path, subfolder, 'linear_proj.pth'))
    #     print("linear_proj_state",linear_proj_state.keys())
    #     self.linear_proj.load_state_dict(linear_proj_state)
    #     return self

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        subfolder="",
        linear_proj_name="linear_proj.pth",
    ):
        config = CLIPTextModelWithProjection.config_class.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        model = cls(config=config)
        model.model_hf = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder)
        linear_proj_state = torch.load(os.path.join(pretrained_model_name_or_path, subfolder, linear_proj_name))
        model.linear_proj.load_state_dict(linear_proj_state)
        return model


    def expand_mask(self, mask=None, dtype="", tgt_len=None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(
            bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


    def make_attn_mask(self, attn_mask):
        seq_len = attn_mask.shape[1]
        query = attn_mask.unsqueeze(1).float()
        # target = query.permute([0, 2, 1]).contiguous().float()
        attn_mask = query.repeat([1, seq_len, 1]).unsqueeze(
            1).repeat([1, self.num_head, 1, 1])
        # attn_mask = target.bmm(query).unsqueeze(1).repeat([1, self.num_head, 1, 1])
        attn_mask = attn_mask.view([-1, seq_len, seq_len])
        return attn_mask

    def gradient_checkpointing_enable(self,):
        self.model_hf.gradient_checkpointing_enable()

    def forward(self, text, attention_mask):

        hidden_states = self.model_hf.text_model.embeddings(
            input_ids=text, position_ids=None)
        # if attention_mask is None:
        #     print('Warning: attention_mask is None in altclip!')
        new_attn_mask = self.expand_mask(
            attention_mask, hidden_states.dtype) if not attention_mask is None else None
        encoder_outputs = self.model_hf.text_model.encoder(
           inputs_embeds=hidden_states,
           attention_mask=new_attn_mask,
           causal_attention_mask=None,
           output_attentions=False,
           output_hidden_states=True,
           return_dict=True,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.model_hf.text_model.final_layer_norm(
            last_hidden_state)
        # x = last_hidden_state
        pooled_output = last_hidden_state[torch.arange(
            last_hidden_state.shape[0]), 0] @ self.model_hf.text_projection.weight

        pooled_output = self.linear_proj(pooled_output)
        return last_hidden_state, pooled_output


    def forward_last2(self, text, attention_mask):
        hidden_states = self.model_hf.text_model.embeddings(
             input_ids=text, position_ids=None)
        if attention_mask is None:
            print('Warning: attention_mask is None in altclip!')
        new_attn_mask = self.expand_mask(attention_mask, hidden_states.dtype) if not attention_mask is None else None
        encoder_outputs = self.model_hf.text_model.encoder(
           inputs_embeds=hidden_states,
           attention_mask=new_attn_mask,
           causal_attention_mask=None,
           output_attentions=False,
           output_hidden_states=True,
           return_dict=True,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.model_hf.text_model.final_layer_norm(last_hidden_state)
        last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0]), 0] @ self.model_hf.text_projection.weight
        pooled_output = self.linear_proj(last_hidden_state)

        extra_features = encoder_outputs.hidden_states[-2]
        extra_features = self.model_hf.text_model.final_layer_norm(extra_features)
        return extra_features, pooled_output


class AnimemoryT5Encoder(sd1_clip.SDClipModel):
    def __init__(
        self, version="configs/dual_text_encoder/text_encoder_t5", device="cuda", max_length=227, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__(special_tokens={"start": 0, "end": 2, "pad": 1})
        text_encoder_cfg_dict = {
                        "d_ff": 10240,
                        "d_kv": 64,
                        "d_model": 4096,
                        "decoder_start_token_id": 0,
                        "dense_act_fn": "gelu_new",
                        "dropout_rate": 0.1,
                        "eos_token_id": 1,
                        "feed_forward_proj": "gated-gelu",
                        "initializer_factor": 1.0,
                        "is_encoder_decoder": False,
                        "is_gated_act": True,
                        "layer_norm_epsilon": 1e-06,
                        "model_type": "t5",
                        "num_decoder_layers": 24,
                        "num_heads": 64,
                        "num_layers": 24,
                        "output_past": True,
                        "pad_token_id": 0,
                        "relative_attention_max_distance": 128,
                        "relative_attention_num_buckets": 32,
                        "tie_word_embeddings": False,
                        "torch_dtype": "bfloat16",
                        "transformers_version": "4.30.2",
                        "use_cache": False,
                        "vocab_size": 32128
                        }
        text_encoder_cfg = T5Config(**text_encoder_cfg_dict)
        self.transformer = my_t5_encoder(config=text_encoder_cfg)
        self.device = device
        self.max_length = max_length
        self.id_pad = 1
        self.id_end = 2
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, text):
        return self(text)

    def forward(self, text):
        # breakpoint()
        tokens = torch.asarray(text).to(self.device)
        for batch_pos in range(len(text)):
            index = text[batch_pos].index(self.id_end)
            tokens[batch_pos, index+1:tokens.shape[1]] = self.id_pad
        masks = (tokens != self.id_pad).to(device=tokens.device, dtype=torch.int64)

        embeddings  = self.transformer.embed_tokens_encoder(tokens)
        encoder_outputs = self.transformer.encoder(inputs_embeds=embeddings, attention_mask=masks, output_hidden_states=True)

        hidden_states = encoder_outputs.hidden_states[-2]
        hidden_states = self.transformer.encoder.final_layer_norm(hidden_states)
        return hidden_states, None

class AnimemoryALTClipEncoder2(sd1_clip.SDClipModel):

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        version="configs/dual_text_encoder/text_encoder_2",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=True,
        legacy=False,
    ):  # clip-vit-base-patch32
        super().__init__(special_tokens={"start": 0, "end": 2, "pad": 1})
        assert layer in self.LAYERS
        text_encoder_cfg_dict_2 = {"attention_dropout": 0.0,
                            "bos_token_id": 0,
                            "dropout": 0.0,
                            "eos_token_id": 2,
                            "hidden_act": "gelu",
                            "hidden_size": 1280,
                            "initializer_factor": 1.0,
                            "initializer_range": 0.02,
                            "intermediate_size": 5120,
                            "layer_norm_eps": 1e-05,
                            "max_position_embeddings": 77,
                            "model_type": "clip_text_model",
                            "num_attention_heads": 20,
                            "num_hidden_layers": 32,
                            "pad_token_id": 1,
                            "projection_dim": 1280,
                            "torch_dtype": "float32",
                            "transformers_version": "4.30.2",
                            "vocab_size": 250002}

        text_encoder_cfg_2 = CLIPTextConfig(**text_encoder_cfg_dict_2)
        self.transformer = myaltclip(text_encoder_cfg_2)

        self.device = device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy
        self.id_pad = 1
        self.id_end = 2


    def forward(self, text):
        # breakpoint()
        tokens = torch.asarray(text).to(self.device)
        for batch_pos in range(len(text)):
            index = text[batch_pos].index(self.id_end)
            tokens[batch_pos, index+1:tokens.shape[1]] = self.id_pad
        masks = (tokens != self.id_pad).to(device=tokens.device, dtype=torch.int64)
        # last_hidden_state, pooled_output = self.model(tokens, None)
        
        # 这里特殊处理一下
        max_embeddings_multiples = (tokens.shape[1] - 2) // (self.max_length - 2)
        if max_embeddings_multiples > 1:
            text_embeddings = []
            pools = []
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                tokens_chunk = tokens[:, i * (self.max_length - 2) : (i + 1) * (self.max_length - 2) + 2].clone()
                # cover the head and the tail by the starting and the ending tokens
                tokens_chunk[:, 0] = tokens[0, 0]

                if not masks is None:
                    masks_chunk = masks[:, i * (self.max_length - 2) : (i + 1) * (self.max_length - 2) + 2].clone()
                    masks_chunk[:, 0] = torch.ones_like(masks_chunk[:, 0])

                if self.id_pad == self.id_end:  # v1
                    tokens_chunk[:, -1] = tokens[0, -1]
                    masks_chunk[:, -1] = masks[:,-1]
                else:  # v2
                    for j in range(len(tokens_chunk)):
                        if tokens_chunk[j, -1] != self.id_end and tokens_chunk[j, -1] != self.id_pad:  # 最後に普通の文字がある
                            tokens_chunk[j, -1] = self.id_end
                            masks_chunk[j, -1] = 1

                        if tokens_chunk[j, 1] == self.id_pad:  # BOSだけであとはPAD
                            tokens_chunk[j, 1] = self.id_end
                            masks_chunk[j, 1] = 1

                text_embedding, pool_i = self.transformer.forward_last2(tokens_chunk, masks_chunk)
                if i == 0:
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    text_embedding = text_embedding[:, 1:]
                else:
                    text_embedding = text_embedding[:, 1:-1]

                text_embeddings.append(text_embedding)
                pools.append(pool_i)
                
            last_hidden_state = torch.concat(text_embeddings, axis=1)
            pooled_output = pools[0]
        else:
            last_hidden_state, pooled_output = self.transformer.forward_last2(tokens, masks)

        if self.return_pooled:
            return last_hidden_state, pooled_output
        return last_hidden_state

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, text):
        return self(text)

class AnimemoryClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.clip_l = AnimemoryT5Encoder()
        self.clip_g = AnimemoryALTClipEncoder2()
        self.dtypes = set([dtype])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return torch.cat([l_out, g_out], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)
