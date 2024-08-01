import os
import json
import torch
import copy
import functools
import safetensors
import torch.nn as nn

from collections import defaultdict
from datasets import load_from_disk
from typing import Optional
from tqdm import tqdm
from tensorrt_llm.models.llama.model import LLaMAForCausalLM
from tensorrt_llm.models.llama.config import LLaMAConfig
from tensorrt_llm.models.llama.convert import load_weights_from_hf_model, smooth_llama_model
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import W8A8_SQ_PLUGIN_LIST, QuantAlgo
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D


class LLaMAForCausalLMWrapper(LLaMAForCausalLM):

    def __init__(self, config: LLaMAConfig):
        super().__init__(config)

    @classmethod
    def quantize(
        cls,
        hf_model_dir: str,
        output_dir: str,
        dtype: str = 'auto',
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        *,
        device: str = 'cuda',
        calib_dataset: str = 'cnn_dailymail',
        num_samplers: int = 512,
        calib_batches: int = 512,
        calib_batch_size: int = 1,
        calib_max_seq_length: int = 512,
        random_seed: int = 1234,
        tokenizer_max_seq_length: int = 2048,
        **kwargs,
    ):
        DEFAULT_MODELOPT_FLOW = [
            QuantAlgo.W4A16_AWQ, QuantAlgo.FP8, QuantAlgo.W8A8_SQ_PER_CHANNEL,
            QuantAlgo.W4A8_AWQ
        ]
        config = LLaMAConfig.from_hugging_face(hf_model_dir,
                                               dtype=dtype,
                                               mapping=mapping,
                                               quant_config=quant_config,
                                               **kwargs)

        if quant_config.quant_algo in DEFAULT_MODELOPT_FLOW:
            super().quantize(hf_model_dir,
                             output_dir,
                             dtype=config.dtype,
                             mapping=config.mapping,
                             quant_config=config.quantization,
                             device=device,
                             calib_dataset=calib_dataset,
                             calib_batches=calib_batches,
                             calib_batch_size=calib_batch_size,
                             calib_max_seq_length=calib_max_seq_length,
                             random_seed=random_seed,
                             tokenizer_max_seq_length=tokenizer_max_seq_length)
        else:
            # non-modelopt, the legacy TRT-LLM native quantization algorithm:
            # sq, int4/int8 weights only, int8 kv cache
            NATIVE_QUANT_FLOW = [QuantAlgo.W4A16, QuantAlgo.W8A16, None
                                 ] + W8A8_SQ_PLUGIN_LIST
            is_valid_native_quant = (quant_config.quant_algo in NATIVE_QUANT_FLOW) and \
                (quant_config.kv_cache_quant_algo in [QuantAlgo.INT8, None])
            assert quant_config.quant_algo is not None or quant_config.kv_cache_quant_algo is not None, \
                "There is no point to call the quantize function if both quant_algo and kv_cache_quant_algo is None"
            assert is_valid_native_quant, f"Internal error: shall call Modelopt for this quantization {quant_config}"

            replace_quantize(hf_model_dir,
                             output_dir,
                             config=config,
                             device=device,
                             calib_dataset=calib_dataset,
                             num_samplers=num_samplers)

def replace_quantize(hf_model_dir: str,
            output_dir: str,
            config: LLaMAConfig,
            device: str = 'cuda',
            calib_dataset: str = 'cnn_dailymail',
            num_samplers: int = 512,):
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=4)

    mapping = config.mapping
    assert mapping.rank == -1, "You shall call quantize only once in one rank, assert rank==-1 for precaution"
    quant_config = config.quantization

    use_smooth_quant = quant_config.use_plugin_sq
    int8_kv_cache = quant_config.kv_cache_quant_algo == QuantAlgo.INT8

    assert use_smooth_quant or int8_kv_cache, "Call from_hugging_face when there is no quantization"
    if use_smooth_quant:
        assert quant_config.smoothquant_val is not None, "A smooth value must be specified when using smooth quant"

    assert hf_model_dir is not None
    ## only load and call smooth quant routine once for all ranks
    hf_config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    assert "llava" not in hf_config.model_type, "Smooth quant llava/vila is not supported yet"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        device_map='auto' if device != 'cpu' else 'cpu',
        torch_dtype='auto' if not use_smooth_quant else torch.float16,
        trust_remote_code=True)

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                            trust_remote_code=True,
                                            use_fast=False,
                                            padding_side='left')

    dataset = load_calib_dataset(calib_dataset)

    act_range, qkv_para, smoother = replace_smooth_quant(hf_model, tokenizer, dataset, num_samplers,
                                                quant_config.smoothquant_val)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(
            hf_model,
            config=config,
            act_range=act_range,
            qkv_para=qkv_para,
            smoother=smoother,
        )
        for k, v in weights.items():
            weights[k] = v.contiguous()
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights


def replace_smooth_quant(model, tokenizer, dataset, num_samplers, smoothquant: Optional[float] = None):
    assert model is not None
    act_range = {}
    llama_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    llama_smoother = {}

    act_range = replace_capture_activation_range(model, tokenizer, dataset, num_samples=num_samplers)
    if smoothquant is not None:
        smooth_llama_model(model, act_range, smoothquant, llama_qkv_para,
                           llama_smoother)
    return act_range, llama_qkv_para, llama_smoother


@torch.no_grad()
def replace_capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token = tokenizer.eos_token

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        datapoint = dataset[i:i + 1]
        line = copy.copy(datapoint)
        line[0] = line[0] + ' TL;DR: '
        line[0] = line[0].strip()
        line[0] = line[0].replace(" n't", "n't")
        input_ids = tokenizer(line,
                              return_tensors="pt",
                              max_length=seq_len,
                              padding=True,
                              truncation=True).input_ids.to(device)
        model(input_ids)
    for h in hooks:
        h.remove()

    return act_scales

def load_calib_dataset(dataset_name_or_dir: str,
                    config_name: Optional[str] = None,
                    split: Optional[str] = None,
                    key: Optional[str] = None,
                    trust_remote_code=True,
                    **kwargs):
    dataset = load_from_disk(dataset_name_or_dir)

    return [dataset["text"][0]]
