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
'''
Implémenter un débruitage d'image, avec régularisation de Tychonov obtenu de deux manières différentes :
1. solution explicite (calculée en Fourier)
2. solution approchée obtenue par optimisation automatique avec Pytorch.
'''
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
### 1.2. Fonctions utilitaires
def rmse_np(ref, est):
    return np.sqrt(np.mean((ref - est) ** 2))


def ssim_np(ref, est):
    return ssim(ref, est, data_range=1.0)

def psnr_np(ref, est):
    rmse = rmse_np(ref, est)
    if rmse == 0:
        return np.inf
    return 20 * np.log10(1.0 / rmse)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Tipo inesperado: {type(x)}")

def show_images(imgs, titles=None, cmap='gray', figsize=(12,4)):
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=figsize)
    if n == 1: axs = [axs]
    for i, im in enumerate(imgs):
        axs[i].imshow(to_numpy(im), cmap=cmap)
        if titles: axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.show()

# %%
#2. Methodologie
### 2.1. Débruitage par solution explicite en Fourier

### cod pour faire tikhonov avec fourier
### obtenu du code de jpynb envoye pour Arthur

def tikhonov_fourier(v, lam):
    """Solution Tikhonov explicite avec conditions périodiques."""
    M, N = v.shape
    xi = np.arange(M)
    zeta = np.arange(N)
    xi, zeta = np.meshgrid(xi, zeta, indexing='ij')
    L_hat = 4 * (np.sin(np.pi * xi / M) ** 2 + np.sin(np.pi * zeta / N) ** 2)
    v_hat = fft2(v)
    u_hat = v_hat / (1 + 2 * lam * L_hat)
    u_rec = np.real(ifft2(u_hat))
    return u_rec

# %%
#### 2.1.3. Résultats qualitatifs

#### test et visualisation
# Tester pour différentes valeurs de λ
lambdas = [0.1, 1, 2, 3, 5]

fig, axes = plt.subplots(1, len(lambdas)+1, figsize=(16, 5))
axes[0].imshow(v, cmap='gray')
axes[0].set_title("Image bruitée")
axes[0].axis('off')

for i, lam in enumerate(lambdas):
    u_fourier = tikhonov_fourier(v, lam)
    axes[i+1].imshow(u_fourier, cmap='gray')
    axes[i+1].set_title(f"λ = {lam}")
    axes[i+1].axis('off')

plt.show()
# %%
#### 2.1.4. Résultats quantitatifs avec PSNR and SSIM

# ------------------
### RQ : WE SHOULD ADD GRAPHS HERE
# ------------------
for lam in lambdas:
    u_rec = tikhonov_fourier(v, lam)
    print(f"λ={lam:>4} → PSNR = {psnr_np(u0, u_rec):.2f} dB")


for lam in lambdas:
    u_rec = tikhonov_fourier(v, lam)
    print(f"λ={lam:>4} → SSIM = {ssim_np(u0, u_rec):.2f}") #1 = perfect


for lam in lambdas:
    u_rec = tikhonov_fourier(v, lam)
    print(f"λ={lam:>4} → RMSE = {rmse_np(u0, u_rec):.2f}") # the lower the better

# %%
### 2.2. Solution approchée (PyTorch)

#### 2.2.2. Implémentation

# a) Définition des gradients discrets

def gradient_x(u):
    gx = torch.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    return gx

def gradient_y(u):
    gy = torch.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gy

# b) Methode appliquant la descente de gradient à une fonction reg_fn entrée en parametre

def optimize_image(v, lam=0.05, lr=0.5, n_iter=200, reg_fn=None, record_loss=False):
    """
    Résout : min_u  0.5‖u - v‖² + λ * reg_fn(u)
    """
    assert reg_fn is not None, "Tu dois donner une fonction de régularisation reg_fn(u)"

    u = v.detach().clone().float().requires_grad_(True)
    opt = torch.optim.SGD([u], lr=lr)
    losses = []

    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)

        fidelity = 0.5 * torch.sum((u - v)**2)
        reg = lam * reg_fn(u)
        loss = fidelity + reg #fonction à minimiser

        loss.backward()
        opt.step()

        with torch.no_grad():
            u.clamp_(0, 1)

        if record_loss:
            losses.append(loss.item())

    return (u.detach(), losses) if record_loss else u.detach()

# c) Application de la descente de gradient avec Tychonov comme fonction de régularisation
def tikhonov_reg(u):
    """Régularisation de Tikhonov : ||∇u||²"""
    gx, gy = gradient_x(u), gradient_y(u)
    return (gx**2 + gy**2).sum()

v_torch = torch.tensor(v, dtype=torch.float32)
lam = 1
lr = 1/(1+16*lam)

u_rec, loss_values = optimize_image(v_torch, lam=lam, lr = lr, reg_fn=tikhonov_reg, record_loss=True)

plt.figure(figsize=(5,3))
plt.plot(loss_values)
plt.title("Évolution de la fonction de coût")
plt.xlabel("Itérations")
plt.ylabel("J(u)")
plt.grid(True)
plt.show()

# Comparaison visuelle avec la solution de Fourier

#Image restaurée par descente de gradient avec Tychonov comme fonction de régularisation
v_torch = torch.tensor(v, dtype=torch.float32)
lam = 1
lr_tych = 1/(1+16*lam)
u_rec, loss_values = optimize_image(v_torch, lam, lr_tych, reg_fn=tikhonov_reg, record_loss=True)
u_tych = to_numpy(u_rec)

#Image restaurée par solution explicite avec Fourier
u_fourier = tikhonov_fourier(v, lam)

show_images(
    [v, u_tych, u_fourier],
    titles=["Image bruitée", f"PyTorch (λ={lam})", f"Fourier (λ={lam})"]
)

# ------------------
### RQ : ICI L'IMAGE DE LA DIFFERENCE ENTRE LES DEUX EST BIZARRE JE TROUVE, FAUDRAIT QUE
### REGARDE DE NOUVEAU
# ------------------

#Explicitation de la différence entre les deux
im = u_fourier - u_tych
show_images([im], titles=["PyTorch - Fourier"])


# Visualisation avec variation de paramètres

# rango de valores de lambda
lambdas = [0.01, 0.02, 0.2, 0.5, 1, 2, 5, 10, 30]
lrs = [1/(1+16*lam) for lam in lambdas]

psnr_fourier_list = []
psnr_pytorch_list = []
lam = 1
for lr in lrs:
    # solución Fourier
    u_fourier = tikhonov_fourier(v, lam)
    psnr_fourier_list.append(psnr_np(u0, u_fourier))

    # solution gradient descent Tychonov
    v_torch = torch.tensor(v, dtype=torch.float32)
    n_iter = 500
    u_pytorch, _ = optimize_image(v_torch, lam, lr, n_iter, reg_fn=tikhonov_reg, record_loss=True)
    u_pytorch_np = u_pytorch.detach().numpy()
    psnr_pytorch_list.append(psnr_np(u0, u_pytorch_np))

# graficar PSNR vs lambda
plt.figure(figsize=(8,5))
plt.plot(lrs, psnr_fourier_list, marker='o', label='Fourier')
plt.plot(lrs, psnr_pytorch_list, marker='s', label='PyTorch')
plt.xlabel('Lambda')
#plt.xscale('log')
plt.ylabel('PSNR [dB]')
plt.title('Comparaison du PSNR pour différentes valeurs de lr')
plt.legend()
plt.grid(True)
plt.show()

# %%

# Application de la descente de gradient avec Total Variation comme fonction de régularisation

def tv_reg(u, eps = 0.01):
    gx, gy = gradient_x(u), gradient_y(u)
    return torch.sqrt(gx**2 + gy**2 + eps**2).sum()

v_torch = torch.tensor(v, dtype=torch.float32)
lam = 1
eps = 0.0001
lr = 2/(1+8*lam/eps)

u_rec, loss_values = optimize_image(v_torch, lam, lr, n_iter = 22000, reg_fn=lambda u: tv_reg(u, eps=eps), record_loss=True)

plt.figure(figsize=(5,3))
plt.plot(loss_values)
plt.title("Évolution de la fonction de coût")
plt.xlabel("Itérations")
plt.ylabel("J(u)")
plt.grid(True)
plt.show()



u_tv = to_numpy(u_rec)
show_images([v, u_tv])

# %%
#### Debluring 

# Load a blur kernel
kt = torch.tensor(np.loadtxt('kernels/kernel8.txt'))
# kt = np.loadtxt('kernels/levin7.txt')
(m,n) = kt.shape

plt.imshow(kt)
plt.title('Blur kernel')
plt.show()

# Embed the kernel in a MxN image, and put center at pixel (0,0)
k = torch.zeros((M,N))
k[0:m,0:n] = kt/torch.sum(kt)
k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
fk = fft2(k)


fk = fk.to(dtype=torch.complex64)

# %%
# Compute the degraded image v = k*u0 + w  (convolution with periodic boundary conditions)


sigma = 0.02

# asegurarte que u0 y fk sean tensores float32
u0_torch = torch.tensor(u0, dtype=torch.float32)
fk_torch = torch.tensor(fk, dtype=torch.complex64)

# convolución periódica
u0_fft = fft2(u0_torch)
conv_u0 = ifft2(u0_fft * fk_torch)

conv_u0 = torch.real(conv_u0)
v = conv_u0 + sigma * torch.randn_like(conv_u0)
v = v.clamp(0.0, 1.0)

plt.figure(dpi=100)
plt.imshow(v.detach().cpu().numpy(), cmap='gray')
plt.title('Image dégradée')
plt.axis('off')
plt.show()
# %%
