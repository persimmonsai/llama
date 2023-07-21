# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch

import traceback
from typing import List
from llama.model import Transformer
from llama.tokenizer import Tokenizer
import os


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
    ) -> "Llama":
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
        if torch.backends.mps.is_available():
            torch.set_default_tensor_type(torch.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)
        if torch.backends.mps.is_available():
            model = model.to("mps")

        generator = Llama(model, tokenizer)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")
        return generator

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        decoded = [None] * bsz
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            decoded.append(self.tokenizer.decode(t))
            for i, t in enumerate(tokens.tolist()):
                t = t[: min(cur_pos, len(prompt_tokens[i]) + max_gen_len)]
                try:
                    t = t[: t.index(self.tokenizer.eos_id)]
                except ValueError:
                    pass
                try:
                    d = self.tokenizer.decode(t)
                    print(d[len(decoded[i - 1]) :], end="", flush=True)
                    decoded[i] = d
                    if d.endswith('\n\n'):
                        return decoded
                except IndexError:
                    traceback.print_exc()
                    print(t)
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
