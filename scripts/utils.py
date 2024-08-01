import os
import shutil

from typing import List


def copy_tokenizer(input_dir: str, output_dir: str):
    # copy tokenizer*
    for filename in os.listdir(input_dir):
        if filename.startswith("tokenizer"):
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(output_dir, filename)
            shutil.copy(src_path, dst_path)
    # copy special_tokens_map.json
    src_path = os.path.join(input_dir, "special_tokens_map.json")
    dst_path = os.path.join(output_dir, "special_tokens_map.json")
    shutil.copy(src_path, dst_path)
