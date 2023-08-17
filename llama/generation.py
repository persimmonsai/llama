# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class Generator:
    def __init__(
        self,
        model,
        tokenizer,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        use_repetition_penalty = True,
        repetition_penalty_range: int = 1024,
        repetition_penalty_slope: float = 0.7,
        repetition_penalty: float = 1.15,
        logprobs: bool = False,
        echo: bool = False,
    ):
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.use_repetition_penalty = use_repetition_penalty
        self.repetition_penalty_range = repetition_penalty_range
        self.repetition_penalty_slope = repetition_penalty_slope
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.logprobs = logprobs
        self.echo = echo
        self.model = model
        self.tokenizer = tokenizer
        self.bsz = len(prompt_tokens)
        params = self.model.params
        assert self.bsz <= params.max_batch_size, (self.bsz, params.max_batch_size)

        self.min_prompt_len = min(len(t) for t in prompt_tokens)
        self.max_prompt_len = max(len(t) for t in prompt_tokens)
        assert self.max_prompt_len <= params.max_seq_len
        self.total_len = min(params.max_seq_len, self.max_gen_len + self.max_prompt_len)

        self.pad_id = self.tokenizer.pad_id
        self.tokens = torch.full((self.bsz, self.total_len), self.pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            self.tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        if logprobs:
            self.token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        self.prev_pos = 0
        self.cur_pos = self.min_prompt_len
        self.eos_reached = torch.tensor([False] * self.bsz)

    def get_next_token(self):
        input_text_mask = self.tokens != self.pad_id
        self.cur_pos += 1
        if self.cur_pos >= self.total_len: return None

        logits = self.model.forward(self.tokens[:, self.prev_pos:self.cur_pos], self.prev_pos)
        if self.logprobs:
            self.token_logprobs[:, self.prev_pos + 1 : self.cur_pos + 1] = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=self.tokens[:, self.prev_pos + 1 : self.cur_pos + 1],
                reduction="none",
                ignore_index=self.pad_id,
            )
        if self.temperature > 0:
            if self.use_repetition_penalty:
                next_token_scores = apply_top_p(logits[:, -1], self.top_p)
                next_token_scores = apply_temperature(next_token_scores, self.temperature)
                next_token_scores = apply_advanced_repetition_penalty(
                    self.tokens[:, :self.cur_pos],
                    next_token_scores,
                    self.repetition_penalty_range,
                    self.repetition_penalty_slope,
                    self.repetition_penalty
                )
                next_token_scores = torch.nn.functional.softmax(
                    next_token_scores, dim = -1
                )
                next_token = torch.multinomial(
                    next_token_scores, num_samples = 1
                ).squeeze(1)
            else:
                probs = torch.softmax(logits[:, -1] / self.temperature, dim=-1)
                next_token = sample_top_p(probs, self.top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        if next_token[0] == self.tokenizer.eos_id: return None
        # only replace token if prompt has already been generated
#        next_token = torch.where(
#            input_text_mask[:, self.cur_pos], self.tokens[:, self.cur_pos], next_token
#        )
        self.tokens[:, self.cur_pos] = next_token
        self.eos_reached |= (~input_text_mask[:, self.cur_pos]) & (
            next_token == self.tokenizer.eos_id
        )
        self.prev_pos = self.cur_pos
        if all(self.eos_reached):
                return None

        return next_token




class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
    ) -> "Llama":
        if not torch.distributed.is_initialized():
            if torch.backends.mps.is_available():
                torch.distributed.init_process_group("gloo")
            else:
                torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(list(Path(ckpt_dir).glob('*.pth'))+list(Path(ckpt_dir).glob('*.pt')))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if 'model' in checkpoint: checkpoint = checkpoint['model']
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        if torch.backends.mps.is_available():
            torch.set_default_tensor_type(torch.HalfTensor)
        elif torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args)

        for layer_id in range(params['n_layers']):
            for wkey in ['wq', 'wk', 'wv']:
                key = f'layers.{layer_id}.attention.{wkey}.weight'
                wq_weight = torch.chunk(checkpoint[key], chunks=params['n_heads'], dim=0)

                del checkpoint[key]
                for i, weight in enumerate(wq_weight, start=0):
                    checkpoint[f'layers.{layer_id}.attention.{wkey}.{i}.weight'] = weight

        model.load_state_dict(checkpoint, strict=False)
        if torch.backends.mps.is_available():
            model = model.to("mps")
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        use_repetition_penalty = True,
        repetition_penalty_range: int = 1024,
        repetition_penalty_slope: float = 0.7,
        repetition_penalty: float = 1.15,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz)
        input_text_mask = tokens != pad_id
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            if temperature > 0:
                if use_repetition_penalty:
                    next_token_scores = apply_top_p(logits[:, -1], top_p)
                    next_token_scores = apply_temperature(next_token_scores, temperature)
                    next_token_scores = apply_advanced_repetition_penalty(
                        tokens[:, :cur_pos],
                        next_token_scores,
                        repetition_penalty_range,
                        repetition_penalty_slope,
                        repetition_penalty
                    )
                    next_token_scores = torch.nn.functional.softmax(
                        next_token_scores, dim = -1
                    )
                    next_token = torch.multinomial(
                        next_token_scores, num_samples = 1
                    ).squeeze(1)
                else:
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
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"raw": t, "generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        for dialog in dialogs:
            if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                    }
                ] + dialog
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {"generation": {"role": "assistant", "content": self.tokenizer.decode(t)}}
            for t in generation_tokens
        ]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def apply_temperature(scores, tempt):
    scores = scores / tempt
    return scores


def apply_top_p(scores, top_p, filter_value = -float("Inf"), min_tokens_to_keep=1):
    sorted_logits, sorted_indices = torch.sort(scores, descending = False)
    cumulative_probs = sorted_logits.softmax(dim = -1).cumsum(dim = -1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, filter_value)
    return scores


def apply_advanced_repetition_penalty(
    input_ids, scores, penalty_range, penalty_slope, penalty
):
    penalty_range = int(penalty_range)
    clipped_penalty_range = min(input_ids.shape[-1], penalty_range)

    if penalty != 1.0:
        if penalty_range > 0:
            if clipped_penalty_range < input_ids.shape[1]:
                input_ids = input_ids[..., -clipped_penalty_range:]

            if penalty_slope != 0:
                _penalty = (
                    torch.arange(
                        penalty_range, dtype = scores.dtype, device = scores.device
                    )
                    / (penalty_range - 1)
                ) * 2.0 - 1
                _penalty = (penalty_slope * _penalty) / (
                    1 + torch.abs(_penalty) * (penalty_slope - 1)
                )
                _penalty = 1 + ((_penalty + 1) / 2).unsqueeze(0) * (penalty - 1)
                penalty = _penalty[..., -clipped_penalty_range:]

        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score <= 0, score * penalty, score / penalty)
        scores.scatter_(1, input_ids, score)
    return scores
