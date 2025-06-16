import json
import sys
import glob
import os
from pathlib import Path
from typing import List

import numpy as np
import zstandard as zstd
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
import utils.packed_dataset as packed_dataset


def prepare(
    source_path: str,
    tokenizer_path: str,
    destination_path: str,
    chunk_size: int,
    split: str="train",
) -> None:
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    filenames = glob.glob(source_path, recursive=True)
    
    if not filenames:
        raise RuntimeError(f"No files found at {source_path}.")

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_slimpajama",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.eos_token_ids,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                    continue # we don't want to include the github data
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))


if __name__ == "__main__":
    prepare(
        source_path="data/slimpajama_sample/raw/*",
        tokenizer_path="Qwen/Qwen3-0.6B",
        destination_path="data/slimpajama_sample/tokenized",
        chunk_size=2049 * 1024,
        split="train",
    )