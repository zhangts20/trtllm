import os
import argparse

from tempfile import TemporaryDirectory
from convert_checkpoint import convert
from export_engine import export
from config import BuildParam


def main(p: BuildParam, root_dir: str) -> None:
    # whether use int8kv or not
    use_int8kv = "int8kv" in p.build_type
    p.build_type = p.build_type.split(",")[0]
    assert p.build_type in ["fp16", "w8a16", "w4a16", "w8a8"]

    # input model directory
    model_dir = os.path.join(root_dir, p.model_name)

    # rename output directory
    output_name = f"{p.model_name}-tp{p.tp_size}-pp{p.pp_size}-{p.max_batch_size}-{p.max_input_length}-{p.max_output_length}"
    output_dir = os.path.join(root_dir, output_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    temp_dir = TemporaryDirectory(dir=root_dir)
    # 0. Convert Weights
    convert(p, use_int8kv, model_dir, output_dir=temp_dir.name)

    # 1. Export Engine
    export(p, model_dir=temp_dir.name, output_dir=output_dir)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir",
                        type=str,
                        required=True,
                        help="The root directory of model.")
    parser.add_argument("--model-name",
                        type=str,
                        required=True,
                        help="The model name to build.")
    parser.add_argument("--build-type",
                        type=str,
                        default="fp16",
                        choices=[
                            "fp16", "fp16,int8kv", "w8a16", "w8a16,int8kv",
                            "w4a16", "w4a16,int8kv", "w8a8", "w8a8,int8kv"
                        ],
                        help="The quantization of generated model.")
    parser.add_argument("--tp-size",
                        type=int,
                        default=1,
                        help="The tensor parallel.")
    parser.add_argument("--pp-size",
                        type=int,
                        default=1,
                        help="The pipeline parallel.")
    parser.add_argument("--max-bs",
                        type=int,
                        default=8,
                        help="The max batch of output of generated model.")
    parser.add_argument("--max-beam",
                        type=int,
                        default=1,
                        help="The max beam width of generated model.")
    parser.add_argument("--max-in",
                        type=int,
                        default=1024,
                        help="The max length of input of generated model.")
    parser.add_argument("--max-out",
                        type=int,
                        default=512,
                        help="The max length of output of generated model.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_args()
    print("Build Parameters:")
    print("================ Argument ================")
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))
    print("==========================================")

    build_param = {
        "model_name": args.model_name,
        "build_type": args.build_type,
        "tp_size": args.tp_size,
        "pp_size": args.pp_size,
        "max_batch_size": args.max_bs,
        "max_beam_width": args.max_beam,
        "max_input_length": args.max_in,
        "max_output_length": args.max_out,
    }
    main(BuildParam(**build_param), root_dir=args.root_dir)
