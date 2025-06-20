import argparse
import json
import sys
import glob
import os
import time
from multiprocessing import Process, cpu_count
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

# Filename for SlimPajama
slimpajama_sets = {
    "train": "train/chunk*/*",
    "validation": "validation/chunk*/*",
    "test": "test/chunk*/*",
}


def prepare(
    filenames: List[str],
    tokenizer_name_or_path: str,
    destination_path: str,
    chunk_size: int,
    split: str="train",
    process_id: int = 0
) -> None:

    Path(destination_path).mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_slimpajama_{process_id}",  # Use process_id to differentiate builders
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

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def main(
    source_path: str,
    tokenizer_name_or_path: str,
    destination_path: str,
    chunk_size: int = 2049 * 1024,
    split: str="train",
    percentage: float = 1.0,
) -> None:

    filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
    filenames = filenames[:int(len(filenames) * percentage)]
    
    num_processes = cpu_count() 
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(
            target=prepare, 
            args=(
                list(subset), 
                tokenizer_name_or_path, 
                destination_path, 
                chunk_size, 
                split, 
                i
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--destination_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=2049 * 1024)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--percentage", type=float, default=1.0)
    args = parser.parse_args()
    main(
        source_path=args.source_path, 
        tokenizer_name_or_path=args.tokenizer_name_or_path, 
        destination_path=args.destination_path, 
        chunk_size=args.chunk_size, 
        split=args.split, 
        percentage=args.percentage
    )