from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import einops
from dataclasses import dataclass
from model import Vit

@dataclass
class ModelArgs:
    block_size: int = 101
    vocab_size: int = 32000
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    image_size: int = 300
    patch_size: int = 30
    chans: int = 3

model = Vit(ModelArgs)

# PIL ile resmi açma
image_path = "/Users/burakbulama/Desktop/Vision Transformer/cat.jpeg"
pil_image = Image.open(image_path)


transform = transforms.Compose([
    transforms.ToTensor(),
])

input_tensor = transform(pil_image)
input_tensor = input_tensor.unsqueeze(0)

text_inp = torch.tensor([[12,34,75,84]])
text_out = torch.tensor([[42,34,76,84]])
print(text_inp.size())

inpt = (input_tensor, text_inp)

logits, loss = model(inpt, text_out)
