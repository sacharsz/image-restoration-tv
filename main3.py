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
#%% METHODOLOGY / COMPUTATION - DENOISING BLOCK

#%% DATA LOADING AND NOISY IMAGE

u0 = img_as_float(data.camera())  # clean image
M, N = u0.shape
sigma = 0.5
v = random_noise(u0, mode='gaussian', var=sigma**2)

show_images([u0, v], titles=["Original", f"Noisy σ={sigma}"])

# Tikhonov Fourier
lambdas = [0.1, 0.3, 0.5, 1, 2, 3, 4, 5]
tikh_results = [(lam, tikhonov_fourier(v, lam)) for lam in lambdas]
best_lambda = 2

# Tikhonov PyTorch
v_torch = torch.tensor(v, dtype=torch.float32)
lam_tyk = 2
lr_tyk = 1/(1 + 16*lam_tyk)
u_tyk, loss_tyk = optimize_image(v_torch, lam_tyk, lr_tyk, n_iter=500, reg_fn=tikhonov_reg, record_loss=True)

# Total Variation PyTorch
lam_tv = 0.7
eps = 1e-3
lr_tv = 2/(1 + 8*lam_tv/eps)
u_tv, loss_tv = optimize_image(v_torch, lam_tv, lr_tv, n_iter=7000, reg_fn=lambda u: tv_reg(u, eps), record_loss=True)

# Metrics across λ for TV
lambdas_tv = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 1.5, 2, 2.5]
lrs_tv = [2/(1 + 8*lam/eps) for lam in lambdas_tv]
psnr_tv, ssim_tv, mse_tv, mae_tv = [], [], [], []
for lam, lr in zip(lambdas_tv, lrs_tv):
    u_tmp = optimize_image(v_torch, lam, lr, n_iter=6000, reg_fn=lambda u: tv_reg(u, eps), record_loss=False)
    u_np = u_tmp.detach().cpu().numpy()
    psnr_tv.append(psnr_np(u0, u_np))
    ssim_tv.append(ssim_np(u0, u_np))
    mse_tv.append(np.mean((u0-u_np)**2))
    mae_tv.append(np.mean(np.abs(u0-u_np)))
psnr_tv_n = np.array(psnr_tv)/max(psnr_tv)
ssim_tv_n = np.array(ssim_tv)/max(ssim_tv)
mse_tv_n  = np.array(mse_tv)/max(mse_tv)
mae_tv_n  = np.array(mae_tv)/max(mae_tv)

#%% PLOTTING - DENOISING BLOCK

# Tikhonov Fourier qualitative
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (lam, u_img) in zip(axes.flatten(), tikh_results):
    ax.imshow(u_img, cmap='gray')
    ax.set_title(f"λ = {lam}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# Tikhonov PyTorch cost
plt.figure(figsize=(5,3))
plt.plot(loss_tyk)
plt.title("Tikhonov cost evolution")
plt.xlabel("Iterations")
plt.ylabel("J(u)")
plt.grid(True)
plt.show()

# Tikhonov Fourier vs PyTorch
u_tyk_np = to_numpy(u_tyk)
u_fourier = tikhonov_fourier(v, lam_tyk)
show_images([v, u_tyk_np, u_fourier, u_fourier-u_tyk_np],
            titles=["Noisy", f"PyTorch λ={lam_tyk}", f"Fourier λ={lam_tyk}", "Difference"])

# Total Variation cost
plt.figure(figsize=(5,3))
plt.plot(loss_tv)
plt.title("Total Variation cost evolution")
plt.xlabel("Iterations")
plt.ylabel("J(u)")
plt.grid(True)
plt.show()

# TV metrics vs λ
plt.figure(figsize=(8,4))
plt.plot(lambdas_tv, psnr_tv_n, marker='o', label='PSNR')
plt.plot(lambdas_tv, ssim_tv_n, marker='s', label='SSIM')
plt.plot(lambdas_tv, mse_tv_n, marker='^', label='MSE')
plt.plot(lambdas_tv, mae_tv_n, marker='d', label='MAE')
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Normalized metrics')
plt.title('Total Variation – Normalized metrics vs λ')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Comparison side-by-side
u_tyk_np = to_numpy(u_tyk)
u_tv_np = to_numpy(u_tv)
fig, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].imshow(u_tyk_np, cmap='gray')
axes[0].set_title(f"Tikhonov λ={lam_tyk}")
axes[0].axis('off')
axes[1].imshow(u_tv_np, cmap='gray')
axes[1].set_title(f"Total Variation λ={lam_tv}")
axes[1].axis('off')
plt.tight_layout()
plt.show()


#%%
# DEBLURING 

#%%
def periodic_conv2d(u, k):
    """Periodic (FFT-based) 2D convolution. u, k: (M,N) float32 torch tensors."""
    uf = fft2(u)
    kf = fft2(k)
    return torch.real(ifft2(uf * kf))

#%%
# IMAGE LOADING 
u0 = img_as_float(data.camera())  # ou data.coins(), data.astronaut() pour tester d'autres
M, N = u0.shape
show_images([u0])
# %%
# LOAD BLUR KERNEL
kt = torch.tensor(np.loadtxt('kernels/kernel8.txt'))

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

