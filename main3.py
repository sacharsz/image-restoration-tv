#%%
# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.util import random_noise
import torch
from ipywidgets import interact
import ipywidgets as widgets
from skimage.metrics import structural_similarity as ssim
from torch.fft import fft2, ifft2, fftshift, ifftshift
#%% 
# IMAGES RELATED METHODS
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

#%%
# METRICS METHODS

def rmse_np(ref, est):
    return np.sqrt(np.mean((ref - est) ** 2))

def mse_np(ref, est):
    return np.mean((ref - est)**2)

def ssim_np(ref, est):
    return ssim(ref, est, data_range=1.0)

def mae_np(ref, est):
    return np.mean(np.abs(ref - est))

def psnr_np(ref, est):
    rmse = rmse_np(ref, est)
    if rmse == 0:
        return np.inf
    return 20 * np.log10(1.0 / rmse)


# %%
# GRADIENT METHODS
def gradient_x(u):
    gx = torch.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    return gx

def gradient_y(u):
    gy = torch.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gy

#%%
# REGULARISATION FONCTIONS
def tikhonov_reg(u):
    """Régularisation de Tikhonov : ||∇u||²"""
    gx, gy = gradient_x(u), gradient_y(u)
    return (gx**2 + gy**2).sum()

def tv_reg(u, eps = 0.001):
    gx, gy = gradient_x(u), gradient_y(u)
    return torch.sqrt(gx**2 + gy**2 + eps**2).sum()

# Applies the gradient descent 
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
        # fidelity = 0.5 * torch.mean((u - v)**2)
        reg = lam * reg_fn(u)

        loss = fidelity + reg #fonction à minimiser

        loss.backward()
        opt.step()

        with torch.no_grad():
            u.clamp_(0, 1)

        if record_loss:
            losses.append(loss.item())

    return (u.detach(), losses) if record_loss else u.detach()



#%%
# DEBLURING 

#%%
def periodic_conv2d(u, k):
    """Periodic (FFT-based) 2D convolution. u, k: (M,N) float32 torch tensors."""
    
        # --- convertit u en torch.float32 si besoin ---
    if isinstance(u, np.ndarray):
        u = torch.tensor(u, dtype=torch.float32)
    else:
        u = u.detach().float()

    # --- convertit k en torch.float32 si besoin ---
    if isinstance(k, np.ndarray):
        k = torch.tensor(k, dtype=torch.float32)
    else:
        k = k.detach().float()

    # passage en complexe pour la FFT
    u_f = fft2(u)
    k_f = fft2(k, s=u.shape)

    conv = ifft2(u_f * k_f)
    return torch.real(conv)

#%%
# IMAGE LOADING 
u0 = img_as_float(data.camera())  # ou data.coins(), data.astronaut() pour tester d'autres
M, N = u0.shape
show_images([u0])
# %%
# LOAD BLUR KERNEL
kt = torch.tensor(np.loadtxt('kernels/kernel8.txt'))
(m,n) = kt.shape
show_images([kt], "Blur kerbel")

# Embed the kernel in a MxN image, and put center at pixel (0,0)
k = torch.zeros((M,N))
k[0:m,0:n] = kt/torch.sum(kt)
k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
fk = fft2(k)
fk = fk.to(dtype=torch.complex64)

# %%
# GENERATION OF THE BLURED IMAGE
sigma = 0.5  # écart-type du bruit

conv_u0 = periodic_conv2d(u0, k)
noise = sigma * torch.randn_like(conv_u0)
v = (conv_u0 + noise).clamp(0.0, 1.0)  # image floutée + bruit, bornée dans [0,1]

plt.figure(dpi=100)
plt.imshow(to_numpy(v), cmap='gray')
plt.title('Image dégradée (blur + bruit)')
plt.axis('off')
plt.show()
# %%
# LOSS FUNCTION

def deblur_tikhonov_loss(u: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         lam: float) -> torch.Tensor:
    """
    J(u) = 0.5 ||k * u - v||^2 + λ ||∇u||^2 (Tikhonov).
    """
    conv_u = periodic_conv2d(u, k)
    fidelity = 0.5 * torch.sum((conv_u - v)**2)
    reg = lam * tikhonov_reg(u)
    return fidelity + reg


def deblur_tv_loss(u: torch.Tensor,
                   k: torch.Tensor,
                   v: torch.Tensor,
                   lam: float,
                   eps_tv: float = 1e-3) -> torch.Tensor:
    """
    J(u) = 0.5 ||k * u - v||^2 + λ TV_eps(u).
    """
    conv_u = periodic_conv2d(u, k)
    fidelity = 0.5 * torch.sum((conv_u - v)**2)
    reg = lam * tv_reg(u, eps=eps_tv)
    return fidelity + reg
# %%
