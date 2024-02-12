import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass


@dataclass
class ModelArgs:
    block_size: int = 1024
    vocab_size: int = 32000
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    image_size: int = 384
    patch_size: int = 32
    chans: int = 3


class PatchEmbed(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.img_size =  args.image_size
        self.patch_size = args.patch_size
        self.n_patch = (args.image_size // args.patch_size) ** 2
        self.proj = nn.Conv2d(
            args.chans, # images channel
            args.n_embd, # embedding_dim = 768
            kernel_size=args.patch_size, #
            stride=args.patch_size
        )
        self.img_pos_embedding = nn.Parameter(
            torch.randn(1, (args.image_size // args.patch_size) ** 2 + 1, args.n_embd)
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, args.n_embd))

    def forward(self, x):
        x = self.proj(x) # (batch_size, n_embed, x_patch, y_patch)
        x = x.flatten(2) # (batch_size, n_embed, patch_size)
        x = x.transpose(1, 2) # (batch_size, patch_size, n_embed)

        b, n, _ = x.shape # (batch_size, patch_size, n_embed)
        x = torch.cat([self.cls_token, x], dim=1)
        x += self.img_pos_embedding[:, :(n + 1)]
        return x

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

class LayerNorm(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(args.n_embd))
        self.bias = nn.Parameter(torch.zeros(args.n_embd)) if args.bias else None
    
    def forward(self, inpt: torch.Tensor):
        return F.layer_norm(inpt, self.weight.shape, self.weight, self.bias, 1e-5)


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

        q = q.view(q.size(0), q.size(1), self.n_head, q.size(2) // self.n_head).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.n_head, k.size(2) // self.n_head).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.n_head, v.size(2) // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(q.size(0), -1, C)
        y = self.resid_dropout(self.to_out(y))

        return y


class GlobalSelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.layernorm = LayerNorm(args)

    def forward(self, x: torch.Tensor):
        attn = self.attention(
            q=x,
            k=x,
            v=x,
        )
        x = x + attn
        x = self.layernorm(x)

        return x


class CasualSelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.layernorm = LayerNorm(args)

    def forward(self, x: torch.Tensor):
        attn = self.attention(
            q=x,
            k=x,
            v=x,
            mask=True
        )
        x = x + attn
        x = self.layernorm(x)

        return x


class CrossAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.layernom = LayerNorm(args)

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        attn = self.attention(
            q = x,
            k = context,
            v = context,
        )
        x = x + attn
        x = self.layernom(x)

        return x

class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.to_fi = nn.Linear(args.n_embd, 4 * args.n_embd, bias=args.bias)
        self.to_out = nn.Linear(args.n_embd * 4, args.n_embd, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
        self.gelu = nn.GELU()
        self.layernom = LayerNorm(args)

    def forward(self, x: torch.Tensor):
        x = self.to_fi(x)
        x = self.gelu(x)
        x = self.to_out(x)
        x = x + self.dropout(x)
        x = self.layernom(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attention = GlobalSelfAttention(args)
        self.ffn = FeedForward(args)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.n_layer
        self.patch_embedding = PatchEmbed(args)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(args) for _ in range(self.num_layers)
        ])

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x)

        return x
    

class DecoderLayer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.casual_self_attention = CasualSelfAttention(args)
        self.cross_attention = CrossAttention(args)
        self.ffn = FeedForward(args)

    def forward(self, x, context):
        x = self.casual_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        x = self.ffn(x)
        return x


class Decoder(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_layers = args.n_layer
        self.seq_embedding = SeqEmbeding(args)
        self.dropout = nn.Dropout(args.dropout)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(args) for _ in range(self.num_layers)
        ])

    def forward(self, x, context):
        x = self.seq_embedding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, context)

        return x

class Vit(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.output = nn.Linear(args.n_embd, args.vocab_size, bias=False)


    def forward(self, inputs, targets=None):
        context, x = inputs

        context = self.encoder(context) # (batch_size, context_len, d_model)
        x = self.decoder(x, context) # (batch_size, target_len, d_model)

        if targets is not None:
            targets = targets.to(torch.long)
            logits = self.output(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(x)
            loss = None

        return logits, loss
    
    def generate(self, inputs, max_new_token):
        context, x = inputs
        idx = x
        context = self.encoder(context) # (batch_size, context_len, d_model)

        for i in range(max_new_token):
            x = self.decoder(idx, context) # (batch_size, target_len, d_model)
            logits = self.output(x)
            logits = logits[:, -1, :]
            props = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(props, num_samples=1)
            _, idx_next = torch.topk(props, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
