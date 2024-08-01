import os
import copy
import torch
import traceback

from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
from tensorrt_llm.builder import BuildConfig, Engine, build
from tensorrt_llm.models.modeling_utils import PretrainedConfig 
from tensorrt_llm.models.llama.model import LLaMAForCausalLM

from config import BuildParam


def build_model(model_dir: str, build_config: BuildConfig,
                model_config: PretrainedConfig, rank: int) -> Engine:
    

    rank_config = copy.deepcopy(model_config)
    rank_config.set_rank(rank)
    model = LLaMAForCausalLM.from_checkpoint(model_dir, config=rank_config) 

    build_config = copy.deepcopy(build_config)
    return build(model, build_config)


def build_and_save(model_dir, output_dir, build_config,
                         model_config, rank):
    torch.cuda.set_device(rank)

    engine = build_model(model_dir, build_config, model_config, rank)
    assert engine is not None
    engine.save(output_dir)


def export(p: BuildParam, model_dir: str, output_dir: str) -> None:
    build_config = BuildConfig.from_dict({
        "max_input_len": p.max_input_length,
        "max_seq_len": p.max_input_length + p.max_output_length,
        "max_batch_size": p.max_batch_size,
        "max_beam_width": p.max_beam_width,
        "max_num_tokens": p.max_batch_size * p.max_input_length,
        "opt_num_tokens": p.max_batch_size * p.max_beam_width,
    })
    model_config = PretrainedConfig.from_json_file(
        os.path.join(model_dir, "config.json"))

    # tp_size * pp_size threads to export engine
    max_workers = min(torch.cuda.device_count(), p.tp_size * p.pp_size)
    if max_workers == 1:
        for rank in range(p.tp_size * p.pp_size):
            build_and_save(model_dir, output_dir, build_config, model_config,
                        rank)
    else:
        with ProcessPoolExecutor(mp_context=get_context("spawn"), max_workers=max_workers) as e:
            futures = [
                e.submit(build_and_save, model_dir, output_dir, build_config,
                         model_config, rank) for rank in range(p.tp_size * p.pp_size)
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
