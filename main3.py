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

# ---------------------------------------------------------------------

# DENOISING

#%% DATA LOADING AND NOISY IMAGE

u0 = img_as_float(data.camera())  # clean image
M, N = u0.shape
sigma = 0.5
v = random_noise(u0, mode='gaussian', var=sigma**2)
v_torch = torch.tensor(v, dtype=torch.float32)

show_images([u0, v], titles=["Original", f"Noisy σ={sigma}"])
#%%
# DENOISING - TYKHONOV ANALYTIC

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

#%%
# TEST WITH SEVERAL LAMBDAS
lambdas_tyk = [0.1, 0.3, 0.5, 1, 2, 3, 4, 5]
lrs_tyk = [1 / (1 + 16 * lam) for lam in lambdas_tyk] 
results = [(lam, tikhonov_fourier(v_torch, lam)) for lam in lambdas_tyk]


# visualisation
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, (lam, u_img) in zip(axes.flatten(), results):
    ax.imshow(u_img, cmap='gray')
    ax.set_title(f"λ = {lam}")
    ax.axis('off')

plt.tight_layout()
plt.show()
#%%
# DENOISING - COMPARAISON BETWEEN TYKHONOV FOURIER AND PYTORCH TO CHECK CONSISTANCY

psnr_fourier_list = []
psnr_pytorch_list = []
ssim_fourier_list = []
ssim_pytorch_list = []
mse_fourier_list = []
mse_pytorch_list = []
mae_fourier_list = []
mae_pytorch_list = []

v_torch_base = torch.tensor(v, dtype=torch.float32)

for lam, lr in zip(lambdas_tyk, lrs_tyk):

    # --- Fourier ---
    u_fourier = tikhonov_fourier(v_torch_base, lam)

    # conversion en numpy
    if isinstance(u_fourier, torch.Tensor):
        u_fourier_np = u_fourier.detach().cpu().numpy()
    else:
        u_fourier_np = np.array(u_fourier)

    psnr_fourier_list.append(psnr_np(u0, u_fourier_np))
    ssim_fourier_list.append(ssim(u0, u_fourier_np, data_range=1.0))
    mse_fourier_list.append(np.mean((u0 - u_fourier_np) ** 2))
    mae_fourier_list.append(np.mean(np.abs(u0 - u_fourier_np)))

    # --- PyTorch / descente de gradient ---
    v_torch = v_torch_base.clone()
    n_iter = 500
    u_pytorch, _ = optimize_image(v_torch, lam, lr, n_iter, reg_fn=tikhonov_reg, record_loss=True)

    u_pytorch_np = u_pytorch.detach().cpu().numpy()
    psnr_pytorch_list.append(psnr_np(u0, u_pytorch_np))
    ssim_pytorch_list.append(ssim(u0, u_pytorch_np, data_range=1.0))
    mse_pytorch_list.append(np.mean((u0 - u_pytorch_np) ** 2))
    mae_pytorch_list.append(np.mean(np.abs(u0 - u_pytorch_np)))


# --- conversion numpy ---
psnr_fourier = np.array(psnr_fourier_list)
ssim_fourier = np.array(ssim_fourier_list)
mse_fourier  = np.array(mse_fourier_list)
mae_fourier  = np.array(mae_fourier_list)

psnr_pytorch = np.array(psnr_pytorch_list)
ssim_pytorch = np.array(ssim_pytorch_list)
mse_pytorch  = np.array(mse_pytorch_list)
mae_pytorch  = np.array(mae_pytorch_list)

# --- normalisation ---
psnr_fourier_n = psnr_fourier / psnr_fourier.max()
ssim_fourier_n = ssim_fourier / ssim_fourier.max()
mse_fourier_n  = mse_fourier  / mse_fourier.max()
mae_fourier_n  = mae_fourier  / mae_fourier.max()

psnr_pytorch_n = psnr_pytorch / psnr_pytorch.max()
ssim_pytorch_n = ssim_pytorch / ssim_pytorch.max()
mse_pytorch_n  = mse_pytorch  / mse_pytorch.max()
mae_pytorch_n  = mae_pytorch  / mae_pytorch.max()


# --- affichage sur un seul graphique ---
plt.figure(figsize=(10, 6))

# Fourier
plt.plot(lambdas_tyk, psnr_fourier_n, marker='o', linestyle='-',  label='PSNR Fourier')
plt.plot(lambdas_tyk, ssim_fourier_n, marker='s', linestyle='-',  label='SSIM Fourier')
plt.plot(lambdas_tyk, mse_fourier_n,  marker='^', linestyle='-',  label='MSE Fourier')
plt.plot(lambdas_tyk, mae_fourier_n,  marker='d', linestyle='-',  label='MAE Fourier')

# PyTorch
plt.plot(lambdas_tyk, psnr_pytorch_n, marker='o', linestyle='--', label='PSNR PyTorch')
plt.plot(lambdas_tyk, ssim_pytorch_n, marker='s', linestyle='--', label='SSIM PyTorch')
plt.plot(lambdas_tyk, mse_pytorch_n,  marker='^', linestyle='--', label='MSE PyTorch')
plt.plot(lambdas_tyk, mae_pytorch_n,  marker='d', linestyle='--', label='MAE PyTorch')

plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Métriques normalisées')
plt.title('PSNR, SSIM, MSE et MAE normalisés en fonction de $\lambda$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# DENOISING - QUALITATIVE COMPARISON BETWEEN FOURIER AND PYTORCH

# This are the values that gives the best results considering analytic and quantitative analysis 
lam = lambdas_tyk[4]
lr_tyk = lrs_tyk[4]
u_rec, loss_values = optimize_image(v_torch, lam, lr_tyk,
                                    reg_fn=tikhonov_reg, record_loss=True)
u_tych = to_numpy(u_rec)

# Image restaurée par solution explicite avec Fourier
u_fourier = tikhonov_fourier(v_torch, lam)

# Différence
im_diff = u_fourier - u_tych

# Show everything on the same line
show_images(
    [v, u_tych, u_fourier, im_diff],
    titles=[
        "Image bruitée",
        f"PyTorch (λ={lam})",
        f"Fourier (λ={lam})",
        "Différence (F - PT)"
    ]  # ensure one single row
)
# Print statistics for each image to check their ranges:
for img, name in zip([v, u_tych, u_fourier], ["Noisy", "PyTorch", "Fourier"]):
    arr = to_numpy(img)
    print(f"{name} min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}, shape: {arr.shape}, dtype: {arr.dtype}")

#%%
# DENOISING - TOTAL VARIATION

# TEST WITH SEVERAL LAMBDAS
lambdas_TV = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 1.5, 2, 2.5]
eps = 0.001
lrs_TV = [2/(1 + 8*lam/eps) for lam in lambdas_TV]



# visualisation
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

results = []
psnr_tv, ssim_tv, mse_tv, mae_tv = [], [], [], []

for lam, lr in zip(lambdas_TV, lrs_TV):
    u_t = optimize_image(
        v_torch, lam, lr,
        n_iter=6000,
        reg_fn=lambda u: tv_reg(u, eps),
        record_loss=False
    )
    u_np = u_t.detach().cpu().numpy()
    results.append((lam, u_np))
    psnr_tv.append(psnr_np(u0, u_np))
    ssim_tv.append(ssim(u0, u_np, data_range=1.0))
    mse_tv.append(np.mean((u0 - u_np)**2))
    mae_tv.append(np.mean(np.abs(u0 - u_np)))

# visualisation
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, (lam, u_img) in zip(axes.flatten(), results):
    ax.imshow(u_img, cmap='gray')
    ax.set_title(f"λ = {lam}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# --- conversion en numpy ---
psnr_tv = np.array(psnr_tv)
ssim_tv = np.array(ssim_tv)
mse_tv  = np.array(mse_tv)
mae_tv  = np.array(mae_tv)

# --- normalisation (0–1) par le max de chaque métrique ---
psnr_tv_n = psnr_tv / psnr_tv.max()
ssim_tv_n = ssim_tv / ssim_tv.max()
mse_tv_n  = mse_tv  / mse_tv.max()
mae_tv_n  = mae_tv  / mae_tv.max()

# === plot unique : toutes les métriques normalisées vs lambda ===
plt.figure(figsize=(8, 4))

plt.plot(lambdas_TV, psnr_tv_n, marker='o', label='PSNR (normalisé)')
plt.plot(lambdas_TV, ssim_tv_n, marker='s', label='SSIM (normalisé)')
plt.plot(lambdas_TV, mse_tv_n,  marker='^', label='MSE (normalisé)')
plt.plot(lambdas_TV, mae_tv_n,  marker='d', label='MAE (normalisé)')

plt.xscale('log')  # optionnel, tu peux enlever si tu veux un axe linéaire
plt.xlabel(r'$\lambda$')
plt.ylabel('Métriques normalisées')
plt.title('TV – PSNR, SSIM, MSE et MAE normalisés en fonction de $\lambda$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



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
# DEBLURING FUNCTION

def run_deblur(method: str,
               lam: float,
               k: torch.Tensor,
               v: torch.Tensor,
               eps_tv: float = 1e-3,
               n_iter: int = 2000):
    
        # on travaille sur des copies détachées
    k_torch = k.detach().float()
    v_torch = v.detach().float()

    # variable à optimiser, initialisée par l'image floutée
    u = v_torch.clone().requires_grad_(True)

    # --- estimation de ||A^T A|| par la FFT du noyau ---
    H = fft2(k_torch, s=v_torch.shape)
    norm_A2 = (H.abs() ** 2).max().item()  # ≈ ||A^T A||

    if method == 'tikh':
        # L ≈ ||A^T A|| + 16 λ (Tikhonov)
        L = norm_A2 + 16.0 * lam
        lr = 0.9 / L
        loss_fn = lambda U: deblur_tikhonov_loss(U, k_torch, v_torch, lam)

    elif method == 'tv':
        # L ≈ ||A^T A|| + 8 λ / ε (TV)
        L = norm_A2 + 8.0 * lam / eps_tv
        lr = 1.9 / L
        loss_fn = lambda U: deblur_tv_loss(U, k_torch, v_torch, lam, eps_tv)

    else:
        raise ValueError("method must be 'tikh' or 'tv'")

    opt = torch.optim.SGD([u], lr=lr)
    loss_values = []

    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(u)
        loss.backward()
        opt.step()

        with torch.no_grad():
            u.clamp_(0.0, 1.0)

        loss_values.append(loss.item())

    restored_np = to_numpy(u)
    return restored_np, loss_values, lr

#%%
# IMAGE LOADING 
u0 = img_as_float(data.camera())
M, N = u0.shape
show_images([u0])
# %%
# LOAD BLUR KERNEL
kt = torch.tensor(np.loadtxt('kernels/kernel8.txt'))
m, n = kt.shape

# --- Embed du noyau dans une image (M, N) et centrage en (0,0) ---
k = torch.zeros((M, N), dtype=torch.float32)
k[0:m, 0:n] = kt / kt.sum()
k = torch.roll(k, shifts=(-m // 2, -n // 2), dims=(0, 1))

plt.figure()
plt.imshow(kt, cmap='gray')
plt.title('Blur kernel')
plt.axis('off')
plt.show()

# --- Embed du noyau dans une image (M, N) et centrage en (0,0) ---
k = torch.zeros((M, N), dtype=torch.float32)
k[0:m, 0:n] = kt / kt.sum()
k = torch.roll(k, shifts=(-m // 2, -n // 2), dims=(0, 1))

# %%
# GENERATION OF THE BLURED IMAGE
sigma = 0.02  # écart-type du bruit

conv_u0 = periodic_conv2d(u0, k)
noise = sigma * torch.randn_like(conv_u0)
v = (conv_u0 + noise).clamp(0.0, 1.0)  # image floutée + bruit, bornée dans [0,1]

plt.figure(dpi=100)
plt.imshow(to_numpy(v), cmap='gray')
plt.title('Image dégradée (blur + bruit)')
plt.axis('off')
plt.show()
# %%
# DEBLURING - TYCHONOV STUDY

#%%
# --- Étude quantitative pour Tikhonov ---

lambdas_tikh = [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10]
psnr_tikh, ssim_tikh, mse_tikh, mae_tikh = [], [], [], []
restored_tikh_images = []  # pour affichage ultérieur

for lam in lambdas_tikh:
    u_rest, loss_vals, lr_used = run_deblur(
        method='tikh',
        lam=lam,
        k=k,
        v=v,
        n_iter=200
    )
    restored_tikh_images.append((lam, u_rest, lr_used))

    psnr_tikh.append(psnr_np(u0, u_rest))
    ssim_tikh.append(ssim_np(u0, u_rest))
    mse_tikh.append(mse_np(u0, u_rest))
    mae_tikh.append(mae_np(u0, u_rest))

psnr_tikh = np.array(psnr_tikh)
ssim_tikh = np.array(ssim_tikh)
mse_tikh  = np.array(mse_tikh)
mae_tikh  = np.array(mae_tikh)

# Normalisation pour affichage
psnr_tikh_n = psnr_tikh / psnr_tikh.max()
ssim_tikh_n = ssim_tikh / ssim_tikh.max()
mse_tikh_n  = mse_tikh  / mse_tikh.max()
mae_tikh_n  = mae_tikh  / mae_tikh.max()

plt.figure(figsize=(8, 4))
plt.plot(lambdas_tikh, psnr_tikh_n, marker='o', label='PSNR (norm.)')
plt.plot(lambdas_tikh, ssim_tikh_n, marker='s', label='SSIM (norm.)')
plt.plot(lambdas_tikh, mse_tikh_n,  marker='^', label='MSE (norm.)')
plt.plot(lambdas_tikh, mae_tikh_n,  marker='d', label='MAE (norm.)')
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Métriques normalisées')
plt.title('Deblurring Tikhonov – métriques normalisées en fonction de λ')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# --- Affichage qualitatif Tikhonov (λ variés) ---

plt.figure(figsize=(16, 3))

# Original
plt.subplot(1, len(restored_tikh_images) + 2, 1)
plt.imshow(u0, cmap='gray')
plt.title("Original")
plt.axis("off")

# Blur
plt.subplot(1, len(restored_tikh_images) + 2, 2)
plt.imshow(to_numpy(v), cmap='gray')
plt.title("Blurred")
plt.axis("off")

# Différents λ
for i, (lam, img, lr_used) in enumerate(restored_tikh_images):
    plt.subplot(1, len(restored_tikh_images) + 2, i + 3)
    plt.imshow(img, cmap='gray')
    plt.title(f"λ={lam}\nlr={lr_used:.1e}", fontsize=8)
    plt.axis("off")

plt.suptitle("Deblurring Tikhonov – influence de λ", fontsize=13)
plt.tight_layout()
plt.show()


# %%
# DEBLURING - TV STUDY

#%%
# --- Étude en λ pour TV (deblurring) ---

lambdas_tv = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
eps_tv = 1e-3
psnr_tv, ssim_tv, mse_tv, mae_tv = [], [], [], []
restored_tv_images = []

for lam in lambdas_tv:
    u_rest, loss_vals, lr_used = run_deblur(
        method='tv',
        lam=lam,
        k=k,
        v=v,
        eps_tv=eps_tv,
        n_iter=1500  # TV converge plus lentement
    )
    restored_tv_images.append((lam, u_rest, lr_used))

    psnr_tv.append(psnr_np(u0, u_rest))
    ssim_tv.append(ssim_np(u0, u_rest))
    mse_tv.append(mse_np(u0, u_rest))
    mae_tv.append(mae_np(u0, u_rest))

psnr_tv = np.array(psnr_tv)
ssim_tv = np.array(ssim_tv)
mse_tv  = np.array(mse_tv)
mae_tv  = np.array(mae_tv)

psnr_tv_n = psnr_tv / psnr_tv.max()
ssim_tv_n = ssim_tv / ssim_tv.max()
mse_tv_n  = mse_tv  / mse_tv.max()
mae_tv_n  = mae_tv  / mae_tv.max()

plt.figure(figsize=(8, 4))
plt.plot(lambdas_tv, psnr_tv_n, marker='o', label='PSNR (norm.)')
plt.plot(lambdas_tv, ssim_tv_n, marker='s', label='SSIM (norm.)')
plt.plot(lambdas_tv, mse_tv_n,  marker='^', label='MSE (norm.)')
plt.plot(lambdas_tv, mae_tv_n,  marker='d', label='MAE (norm.)')
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Métriques normalisées')
plt.title('Deblurring TV – métriques normalisées en fonction de λ')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
#%%
# --- Affichage qualitatif TV ---

plt.figure(figsize=(16, 3))

# Original
plt.subplot(1, len(restored_tv_images) + 2, 1)
plt.imshow(u0, cmap='gray')
plt.title("Original")
plt.axis("off")

# Blur
plt.subplot(1, len(restored_tv_images) + 2, 2)
plt.imshow(to_numpy(v), cmap='gray')
plt.title("Blurred")
plt.axis("off")

# TV pour différents λ
for i, (lam, img, lr_used) in enumerate(restored_tv_images):
    plt.subplot(1, len(restored_tv_images) + 2, i + 3)
    plt.imshow(img, cmap='gray')
    plt.title(f"λ={lam}\nlr={lr_used:.1e}", fontsize=8)
    plt.axis("off")

plt.suptitle("Deblurring TV – influence de λ", fontsize=13)
plt.tight_layout()
plt.show()

#%%
# --- Choix de paramètres "optimaux" (à partir des études précédentes) ---

lam_tikh_best = 3.0
lam_tv_best   = 0.3
eps_tv_best   = 0.005

rest_tikh, loss_tikh, lr_tikh = run_deblur(
    method='tikh',
    lam=lam_tikh_best,
    k=k,
    v=v,
    n_iter=1500
)

rest_tv,  loss_tv,  lr_tv  = run_deblur(
    method='tv',
    lam=lam_tv_best,
    k=k,
    v=v,
    eps_tv=eps_tv_best,
    n_iter=1500
)

# métriques
psnr_tikh_b = psnr_np(u0, rest_tikh)
mse_tikh_b  = mse_np(u0, rest_tikh)
mae_tikh_b  = mae_np(u0, rest_tikh)
ssim_tikh_b = ssim_np(u0, rest_tikh)

psnr_tv_b = psnr_np(u0, rest_tv)
mse_tv_b  = mse_np(u0, rest_tv)
mae_tv_b  = mae_np(u0, rest_tv)
ssim_tv_b = ssim_np(u0, rest_tv)

print("\n=== Comparaison quantitative – deblurring ===")
print(f"Tikhonov (λ={lam_tikh_best}):")
print(f"  PSNR = {psnr_tikh_b:.2f} dB")
print(f"  MSE  = {mse_tikh_b:.4f}")
print(f"  MAE  = {mae_tikh_b:.4f}")
print(f"  SSIM = {ssim_tikh_b:.3f}")

print(f"\nTotal Variation (λ={lam_tv_best}, ε={eps_tv_best}):")
print(f"  PSNR = {psnr_tv_b:.2f} dB")
print(f"  MSE  = {mse_tv_b:.4f}")
print(f"  MAE  = {mae_tv_b:.4f}")
print(f"  SSIM = {ssim_tv_b:.3f}")

# affichage images
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(u0, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(to_numpy(v), cmap='gray')
plt.title("Blurred")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(rest_tikh, cmap='gray')
plt.title(f"Tikhonov\nλ={lam_tikh_best}, lr={lr_tikh:.1e}")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(rest_tv, cmap='gray')
plt.title(f"TV\nλ={lam_tv_best}, ε={eps_tv_best}, lr={lr_tv:.1e}")
plt.axis("off")

plt.tight_layout()
plt.show()
