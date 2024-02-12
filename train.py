import torch
import torch.nn as nn

from dataclasses import dataclass

from model import Vit
from deneme import train_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

model = Vit(ModelArgs)
model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

train_data = train_dataset()
image, text = train_data

for iter in range(40):

    logits, loss = model(train_data, text)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())
