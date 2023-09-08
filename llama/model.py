# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn

from torch.nn.utils import skip_init
import sys
import os

complex_device = torch.device("cpu")
device = complex_device
if torch.cuda.is_available():
    device = torch.device("cuda")
    complex_device = device
    torch.set_default_device(device)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.set_default_device(device)

def fairscale_row_parallel_linear(d1, d2, layer_id): return RowParallelLinear(d1, d2, bias=False, input_is_parallel=True, init_method=lambda x: x)

def fairscale_column_parallel_linear(d1, d2, layer_id): return ColumnParallelLinear(d1, d2, bias=False, gather_output=False, init_method=lambda x: x)

def fairscale_parallel_embedding(d1, d2, layer_id): return ParallelEmbedding(d1, d2, init_method=lambda x: x)

def test_parallel_linear1(d1, d2, layer_id): return skip_init(nn.Linear, d1, d2, bias=False)

def test_parallel_embedding1(d1, d2, layer_id): return skip_init(nn.Embedding, d1, d2)

import numpy as np

def debug_one(a, name, start_pos = None, layer_id = None, head = None):
    d = '/tmp/mdebug'
    if not os.path.exists(d): os.mkdir(d)
    
    if(start_pos is not None):
        d = d + '/S' + str(start_pos)
        if not os.path.exists(d): os.mkdir(d)
    if(layer_id is not None):
        d = d + '/L' + str(layer_id)
        if not os.path.exists(d): os.mkdir(d)
    if(head is not None):
        d = d + '/H' + str(head)
        if not os.path.exists(d): os.mkdir(d)
    np.save(d + '/' + name + '.npy', a.cpu().numpy())

def debug_two(r, a, name, start_pos = None, layer_id = None, head = None):
    debug_one(r, name + '-r', start_pos, layer_id, head)
    debug_one(a, name + '-a', start_pos, layer_id, head)

    return r

def debug_mult_output(r, a, b, name, start_pos = None, layer_id = None, head = None):
    debug_one(r, name + '-r', start_pos, layer_id, head)
    debug_one(a, name + '-a', start_pos, layer_id, head)
    debug_one(b, name + '-b', start_pos, layer_id, head)

    return r;

def debug_mult(a, b, name, start_pos, layer_id, head):
    r = torch.matmul(a, b)
    debug_mult_output(r, a, b, name, start_pos, layer_id, head);
    return r

def debug_stacked_mult(a, b, name, start_pos, layer_id):
    return torch.stack([ debug_mult(a[head], b[head], name, start_pos, layer_id, head) for head in range(len(a))])

class test_parallel_linear(nn.Linear):
    def __init__(self, d1, d2, name, layer_id, head):
        self.layer_id = layer_id
        self.name = name
        self.head = head
        self.start_pos = None
        super().__init__(d1, d2, bias=False, device='meta')
        self.to_empty(device=device)
    def forward(self, x):
#        return x.matmul(self.weight.t())
        return debug_mult_output(super().forward(x), x, self.weight, self.name, self.start_pos, self.layer_id, self.head)

class test_parallel_embedding(nn.Embedding):
    def __init__(self, d1, d2, name):
        super().__init__(d1, d2, device='meta')
        self.to_empty(device=device)
        self.name = name
    def forward(self, x, start_pos):
        return debug_mult_output(super().forward(x), x, self.weight, self.name, start_pos)

use_fairscale = False

if use_fairscale:
    wqParallelLinear = fairscale_column_parallel_linear
    wkParallelLinear = fairscale_column_parallel_linear
    wvParallelLinear = fairscale_column_parallel_linear
    woParallelLinear = fairscale_row_parallel_linear
    w1ParallelLinear = fairscale_column_parallel_linear
    w2ParallelLinear = fairscale_row_parallel_linear
    w3ParallelLinear = fairscale_column_parallel_linear
    outputParallelLinear = fairscale_column_parallel_linear
    parallelEmbedding = fairscale_parallel_embedding
else:
    wqParallelLinear = test_parallel_linear
    wkParallelLinear = test_parallel_linear
    wvParallelLinear = test_parallel_linear
    woParallelLinear = test_parallel_linear
    w1ParallelLinear = test_parallel_linear
    w2ParallelLinear = test_parallel_linear
    w3ParallelLinear = test_parallel_linear
    outputParallelLinear = test_parallel_linear
    parallelEmbedding = test_parallel_embedding

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, name: str = None, layer: int = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.layer = layer
        self.name = name

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, start_pos = None):
        output = self._norm(x.float()).type_as(x)
        r = output * self.weight
        return debug_mult_output(r, output, self.weight, self.name, start_pos, self.layer)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=complex_device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    start_pos = None, layer_id = None, head = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.to(freqs_cis.device)
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(x.ndim-1)
    debug_two(x_out, x, 'rot-emb', start_pos, layer_id, head)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id

        self.wq = nn.ModuleList([wqParallelLinear(
            args.dim,
            self.head_dim,
            'wq', layer_id, head,
        ) for head in range(args.n_heads)])

        self.wk = nn.ModuleList([wkParallelLinear(
            args.dim,
            self.head_dim,
            'wk', layer_id, head,
        ) for head in range(self.n_kv_heads)])

        self.wv = nn.ModuleList([wvParallelLinear(
            args.dim,
            self.head_dim,
            'wv', layer_id, head,
        ) for head in range(self.n_kv_heads)])

        self.wo = woParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            'wo', layer_id, None
        )

        self.cache_k = torch.zeros(
            (
                self.n_local_kv_heads,
                args.max_batch_size,
                args.max_seq_len,
                self.head_dim,
            )
        ).to(device)
        self.cache_v = torch.zeros(
            (
                self.n_local_kv_heads,
                args.max_batch_size,
                args.max_seq_len,
                self.head_dim,
            )
        ).to(device)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        for wq in self.wq: wq.start_pos = start_pos
        for wk in self.wk: wk.start_pos = start_pos
        for wv in self.wv: wv.start_pos = start_pos

        self.wo.start_pos = start_pos

        xq = torch.stack([apply_rotary_emb(self.wq[h](x), freqs_cis=freqs_cis, start_pos=start_pos, layer_id=self.layer_id, head=h) for h in range(len(self.wq))], dim=2)
        xk = torch.stack([apply_rotary_emb(self.wk[h](x), freqs_cis=freqs_cis, start_pos=start_pos, layer_id=self.layer_id, head=h) for h in range(len(self.wk))], dim=2)
        xv = torch.stack([wv(x) for wv in self.wv], dim=2)

        xq = xq.permute(2, 0, 1, 3)
        xk = xk.permute(2, 0, 1, 3)
        xv = xv.permute(2, 0, 1, 3)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:, :bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, :bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:, :bsz, : start_pos + seqlen]
        values = self.cache_v[:, :bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        keys = keys.transpose(2,3)
        scores = debug_stacked_mult(xq, keys, 'at-scores', start_pos, self.layer_id) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = debug_stacked_mult(scores, values, 'at-out', start_pos, self.layer_id)
        output = output.transpose(0, 1)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        layer_id: int,
    ):
        super().__init__()
        self.layer_id = layer_id
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = w1ParallelLinear(
            dim, hidden_dim, 'w1', layer_id, None
        )
        self.w2 = w2ParallelLinear(
            hidden_dim, dim, 'w2', layer_id, None
        )
        self.w3 = w3ParallelLinear(
            dim, hidden_dim, 'w3', layer_id, None
        )

    def forward(self, x, start_pos):
        self.w1.start_pos = start_pos
        self.w2.start_pos = start_pos
        self.w3.start_pos = start_pos

        silu_w1_x = F.silu(self.w1(x))
        w3_x = self.w3(x)
        r = silu_w1_x * w3_x
        debug_mult_output(r, silu_w1_x, w3_x, 'ff-out', start_pos, self.layer_id)
        return self.w2(r)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, layer_id)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            layer_id=layer_id,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, name='att-norm', layer=layer_id)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, name='ffn-norm', layer=layer_id)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention.forward(
            self.attention_norm(x, start_pos), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h, start_pos), start_pos)
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = parallelEmbedding(
            params.vocab_size, params.dim, 'tok-emb'
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps, name='out-norm')
        self.output = outputParallelLinear(
            params.dim, params.vocab_size, 'out', None, None
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens, start_pos)
        #self.freqs_cis = self.freqs_cis.float().to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        debug_one(freqs_cis, 'freqs-cis', start_pos)

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=freqs_cis.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, (mask.to(device) if mask is not None else mask))
        h = self.norm(h)
        output = self.output(h).float()
        return output
