#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import warnings

warnings.filterwarnings("ignore")
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import resource

resource.setrlimit(resource.RLIMIT_NOFILE, (10000, 10000))

import sys
import torch
import fire
import time
import json
import warnings
from typing import Tuple
import random
from pathlib import Path
from llama import ModelArgs, Transformer, Tokenizer, Llama


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLama:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[0]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    random_seed = random.randint(1, 65534)
    torch.manual_seed(random_seed)
    print(f"Seed: {random_seed:5d}")
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model = model.to("mps")

    generator = Llama(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 1):

    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    try:
        while True:
            queryInputs = [input("Enter your LLaMA prompt: ")]
            print("Thinking...")
            queryTime = time.time()
            results = generator.generate(
                queryInputs, max_gen_len=max_seq_len, temperature=temperature, top_p=top_p
            )
            print(f"\nInferred in {time.time() - queryTime:.2f} seconds")
            print("==================================\n")
    except KeyboardInterrupt:
        sys.exit()


if __name__ == "__main__":
    fire.Fire(main)
