import torch
from PIL import Image
import numpy as np
import torchvision as tv

def save_image(pic, path):
    grid = tv.utils.make_grid(pic, nrow=8, padding=2)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


