#!/usr/bin/env python3

import os

import sys
import fire
import time

from llama import ModelArgs, Transformer, Tokenizer, Llama, Generator

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 1):

    llama = Llama.build(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    try:
        while True:
            queryInputs = [input("Enter your LLaMA prompt: ")]
            queryInputs = [llama.tokenizer.encode(x, bos=True, eos=False) for x in queryInputs]

            print("Thinking...")

            queryTime = time.time()
            generator = Generator(llama.model, llama.tokenizer,
                queryInputs, max_gen_len=max_seq_len, temperature=temperature, top_p=top_p
            )
            tokenx = []
            while None != (next_token := generator.get_next_token()):
               text = llama.tokenizer.decode(next_token.tolist())
               print(text, end="", flush=True)

            print(f"\nInferred in {time.time() - queryTime:.2f} seconds")
            print("==================================\n")
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    fire.Fire(main)
