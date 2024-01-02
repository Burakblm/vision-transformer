import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelArgs:
    block_size: int = 1024
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class SeqEmbeding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.block_size is not None

        self.args = args

        self.tok_embedding = nn.Embedding(args.vocab_size, args.n_embd)
        self.pos_embedding = nn.Embedding(args.block_size, args.n_embd)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long)
        idx = idx.to(torch.long)
        token_embd = self.tok_embedding(idx)
        pos_embd = self.pos_embedding(pos)
        x = self.drop(token_embd + pos_embd)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w_q = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.w_k = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.w_v = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.to_out = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.register_buffer("bias", torch.tril(torch.ones(args.block_size, args.block_size))
                            .view(1, 1, args.block_size, args.block_size))

        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout

    def forward(self, q, k, v, mask = False):
        B, T, C = q.size()

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.to_out(y))

        return y


class CasualSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.layernorm = LayerNorm(args.n_embd, bias=args.bias)

    def forward(self, x: torch.Tensor):
        attn = self.attention(
            q=x,
            k=x,
            v=x,
            mask=True
        )

        return self.layernorm(attn)

class CrossAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.layernom = LayerNorm(args.n_embd, bias=args.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        attn = self.attention(
            q = x,
            k = x,
            v = y,
            mask=False
        )
        
        return self.layernom(attn)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.to_fi = nn.Linear(args.n_embd, 4 * args.n_embd, bias=args.bias)
        self.to_out = nn.Linear(args.n_embd * 4, args.n_embd, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.to_fi(x)
        x = self.gelu(x)
        x = self.to_out(x)
        x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attetion = CasualSelfAttention(args)
        self.cross_attetnion = CrossAttention(args)
        self.ff = FeedForward(args)

    def forward(self, inputs):
        in_seq, out_seq = inputs

        out_seq = self.self_attetion(out_seq)
        out_seq = self.cross_attetnion(out_seq, in_seq)
        out_seq = self.ff(out_seq)

        return out_seq

