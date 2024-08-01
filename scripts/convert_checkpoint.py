import torch
import traceback
from transformers import AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed

from tensorrt_llm._utils import release_gc
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.mapping import Mapping

from config import BuildParam
from quantization import LLaMAForCausalLMWrapper


def args_to_quantization(
    build_type: str,
    use_int8kv: bool,
    sq_value: float,
) -> QuantConfig:
    q_config = QuantConfig()
    q_config.exclude_modules = ["lm_head"]

    if build_type == "w8a16":
        q_config.quant_algo = QuantAlgo.W8A16
    elif build_type == "w4a16":
        q_config.quant_algo = QuantAlgo.W4A16
    elif build_type == "w8a8":
        q_config.quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN

    if use_int8kv:
        q_config.kv_cache_quant_algo = QuantAlgo.INT8

    if sq_value is not None:
        q_config.smoothquant_val = sq_value

    return q_config


def convert(
    p: BuildParam,
    use_int8kv: bool,
    model_dir: str,
    output_dir: str,
    calib_dataset: str = None,
    num_samplers: int = 512,
    sq_value: float = None,
) -> None:
    q_config = args_to_quantization(p.build_type, use_int8kv, sq_value)
    if p.build_type == "w8a8" or use_int8kv:
        assert calib_dataset is not None, "Please set calib_dataset when build type is w8a8 or int8kv."
        mapping = Mapping(world_size=p.tp_size * p.pp_size,
                          rank=-1,
                          tp_size=p.tp_size,
                          pp_size=p.pp_size)
        LLaMAForCausalLMWrapper.quantize(model_dir,
                                  output_dir,
                                  mapping=mapping,
                                  quant_config=q_config,
                                  calib_dataset=calib_dataset,
                                  num_samplers=num_samplers)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                        device_map="auto",
                                                        torch_dtype="auto",
                                                        trust_remote_code=True)

        def convert_and_save_rank(rank, tp_size, pp_size, output_dir) -> None:
            mapping = Mapping(world_size=tp_size * pp_size,
                              rank=rank,
                              tp_size=tp_size,
                              pp_size=pp_size)
            llama = LLaMAForCausalLMWrapper.from_hugging_face(
                model_dir if hf_model is None else hf_model,
                mapping=mapping,
                quant_config=q_config)
            llama.save_checkpoint(output_dir, save_config=(rank == 0))
            del llama

        # tp_size * pp_size threads to export engine
        max_workers = min(torch.cuda.device_count(), p.tp_size * p.pp_size)
        with ThreadPoolExecutor(max_workers=max_workers) as t:
            futures = [
                t.submit(convert_and_save_rank, rank, p.tp_size, p.pp_size,
                         output_dir) for rank in range(p.tp_size * p.pp_size)
            ]
            expections = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    expections.append(e)
            assert len(
                expections
            ) == 0, "Checkpoint conversion failed, please check error log."
        release_gc()
