from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import einops
from dataclasses import dataclass
from model import Vit


def train_dataset():
    # PIL ile resmi açma
    image_path = "/Users/burakbulama/Desktop/Vision Transformer/cat.jpeg"
    pil_image = Image.open(image_path)


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_tensor = transform(pil_image)
    input_tensor = input_tensor.unsqueeze(0)

    text_inp = torch.tensor([[12,34,75,84]])
    text_out = torch.tensor([[12,34,75,84]])
    text_bos_tok = torch.tensor([[0]])
    inpt = (input_tensor, text_inp)
    return inpt

"""
inpt = (input_tensor, text_inp)

logits, loss = model(inpt, text_out)
print(loss.item())

inpt = (input_tensor, text_bos_tok)
out = model.generate(inpt, 20)
print(out)
"""
