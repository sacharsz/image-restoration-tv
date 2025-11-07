#%%
# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.util import random_noise
from numpy.fft import fft2, ifft2
import torch
from ipywidgets import interact
import ipywidgets as widgets
from skimage.metrics import structural_similarity as ssim
# %%
# 1 Charger une image propre
u0 = img_as_float(data.camera())  # ou data.coins(), data.astronaut() pour tester d'autres
M, N = u0.shape

# 2 Ajouter un bruit gaussien
sigma = 0.5
v = random_noise(u0, mode='gaussian', var=sigma**2)

# 3 Afficher l'image originale et bruitée
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(u0, cmap='gray')
axes[0].set_title("Image originale")
axes[0].axis('off')

axes[1].imshow(v, cmap='gray')
axes[1].set_title(f"Image bruitée (σ={sigma})")
axes[1].axis('off')

plt.show()

# %%
