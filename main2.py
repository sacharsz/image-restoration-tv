#%% [markdown]
# # 0. Preliminaires

#%%
#!pip install numpy matplotlib scikit-image torch ipywidgets

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

#%% [markdown]
# # 1. Introduction
#
# #### Implémenter un débruitage d'image, avec régularisation de Tychonov obtenu de deux manières différentes :
# 1. solution explicite (calculée en Fourier)
# 2. solution approchée obtenue par optimisation automatique avec Pytorch.

#%% [markdown]

#%% [markdown]
# On cherche à minimiser la fonction:
#
# $$
# J(u) = \frac{1}{2} \|u - v\|^2 + \lambda \|\nabla u\|^2
# $$
#
# où \( 𝒗 \) est l'image bruitée et \( λ \) le paramètre de régularisation.  
#
# La différence avec la méthode analytique est que l'on approche la solution en utilisant une **descente de gradient** implémentée avec **PyTorch**, particulièrement utile lorsque la solution analytique n'est pas accessible.

#%% [markdown]
# ### 1.1. Chargement d'image et ajoute de bruit gaussien

#%%
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

#%% [markdown]
# On observe que le bruit gaussien altère fortement les zones homogènes de l'image.  
# Notre objectif sera de réduire ce bruit tout en préservant les contours importants.

#%% [markdown]
# ### 1.2. Fonctions utilitaires

#%%
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

def show_images(imgs, titles=None, cmap='gray', figsize=(12,4), vmin=0, vmax=1):
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=figsize)
    if n == 1: axs = [axs]
    for i, im in enumerate(imgs):
        axs[i].imshow(to_numpy(im), cmap=cmap)
        if titles: axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.show()

#%% [markdown]
# #2. Methodologie

#%% [markdown]
# ### 2.1. Solution analytique (Fourier)

#%% [markdown]
# #### 2.1.1. Principe
# Pour débruiter une image en utilisant la régularisation de Tychonov, on cherche à minimiser :
#
# $$
# \frac{1}{2} \| u - v \|_2^2 + λ \| ∇ u \|_2^2
# $$
#
# où \(v\) est l'image bruitée et \(∇u\) les dérivées discrètes de l'image.  
#
# Avec des conditions aux bords périodiques, la solution \($u^{*}$ \) peut s'écrire explicitement dans le domaine de Fourier :
#
# $$
# \hat{u}^*(\xi, \zeta) = \frac{\hat{v}(\xi, \zeta)}{1 + 2 \lambda \hat{L}(\xi, \zeta)}
# $$
#
# avec
#
# $$
# \hat{L}(\xi, \zeta) = 4 \left( \sin^2\left(\frac{\pi \xi}{M}\right) + \sin^2\left(\frac{\pi \zeta}{N}\right) \right)
# $$
#
# Cette formule agit comme un **filtre passe-bas** : plus \(λ\) est grand, plus les hautes fréquences sont atténuées, ce qui lisse l'image.
#
#
# Notez que la solution de Fourier est **immédiate**, ce qui sera vu plus tard dans la comparaison des temps.

#%% [markdown]
# #### 2.1.2. Implémentation

#%%
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

#%% [markdown]
# #### 2.1.3. Résultats qualitatifs
#
# - Pour **λ petit (0.1)** → le débruitage est faible, l'image reste bruitée.  
# - Pour **λ grand (20)** → le bruit disparaît mais les contours deviennent flous.  
# - Pour **λ intermédiaire (1 à 5)** → bon compromis entre suppression du bruit et conservation des contours.  
#
# Cette évolution correspond à ce que prédit la régularisation de Tychonov : plus λ augmente, plus les hautes fréquences sont atténuées, et l'image devient plus lissée.

#%%
#### test
lambdas = [0.1, 0.3, 0.5, 1, 2, 3, 4, 5]
results = [(lam, tikhonov_fourier(v, lam)) for lam in lambdas]


#### visualisation
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, (lam, u_img) in zip(axes.flatten(), results):
    ax.imshow(u_img, cmap='gray')
    ax.set_title(f"λ = {lam}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# save as a file
output_filename = "tikhonov_lambdas_qualitative.png"
fig.savefig(output_filename, dpi=300)
print(f"Saved as '{output_filename}'")

#%% [markdown]
# #### 2.1.4. Résultats quantitatifs avec PSNR and SSIM

#%%
#### metrics table
best_lambda = 2  # ganador

def mae_np(a, b):
    return np.mean(np.abs(a - b))


# header
print(f"{'λ':>6} | {'PSNR':>7} | {'SSIM':>6} | {'RMSE':>6} | {'MAE':>6}")
print(f"{'':>6} | {'(↑)':>7} | {'(↑)':>6} | {'(↓)':>6} | {'(↓)':>6}")
print("-"*60)

for lam in lambdas:
    u_rec = tikhonov_fourier(v, lam)
    psnr_val = psnr_np(u0, u_rec)
    ssim_val = ssim_np(u0, u_rec)
    rmse_val = rmse_np(u0, u_rec)
    mae_val  = mae_np(u0, u_rec)

    mark = "  <--- best" if lam == best_lambda else ""
    print(f"{lam:>6} | {psnr_val:>7.2f} | {ssim_val:>6.3f} | "
          f"{rmse_val:>6.3f} | {mae_val:>6.3f}{mark}")

#%% [markdown]
# Le PSNR permet de quantifier la qualité de la restauration.  
# On remarque qu'il atteint souvent un maximum pour une valeur intermédiaire de λ,
# correspondant au meilleur compromis entre bruit et netteté.

#%% [markdown]
# ### 2.2. Solution approchée (PyTorch)

#%% [markdown]
# #### 2.2.1. Principe
# On cherche à résoudre le même problème de débruitage :
#
# $$
# \min_u \; \|u - v\|^2 + \lambda \|\nabla u\|^2
# $$
#
# soit la fonctionnelle :
#
# $$
# J(u) = \frac{1}{2} \|u - v\|^2 + \lambda \|\nabla u\|^2
# $$
#
#
# La différence avec la solution analytique est que nous allons **approcher la solution** numériquement avec **descente de gradient** en utilisant **PyTorch**.  
#
# Cette méthode est utile quand la solution exacte n'est pas accessible (par exemple si \(A\) n'est plus l'identité).

#%% [markdown]
# #### 2.2.2. Implémentation

#%% [markdown]
# a) Définition des gradients discrets

#%%
def gradient_x(u):
    gx = torch.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    return gx

def gradient_y(u):
    gy = torch.zeros_like(u)
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gy

#%% [markdown]
# b) Methode appliquant la descente de gradient à une fonction reg_fn entrée en parametre

#%%
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

#%% [markdown]
# c) Application de la descente de gradient avec Tychonov comme fonction de régularisation

#%%
def tikhonov_reg(u):
    """Régularisation de Tikhonov : ||∇u||²"""
    gx, gy = gradient_x(u), gradient_y(u)
    return (gx**2 + gy**2).sum()


v_torch = torch.tensor(v, dtype=torch.float32)
lam = 2
lr = 1/(1+16*lam)

u_rec, loss_values = optimize_image(v_torch, lam=lam, lr = lr, n_iter=85, reg_fn=tikhonov_reg, record_loss=True)

plt.figure(figsize=(5,3))
plt.plot(loss_values)
plt.title("Évolution de la fonction de coût")
plt.xlabel("Itérations")
plt.ylabel("J(u)")
plt.grid(True)
plt.show()

# === compute convergence iteration ===
loss_arr = np.array(loss_values)
smooth_d = np.convolve(np.abs(np.diff(loss_arr)), np.ones(50)/50, mode='valid')

thr = 1e-1
idx = np.where(smooth_d < thr)[0]

print("Convergence at iteration:", idx[0] if len(idx)>0 else "not detected")

#%% [markdown]
# d) Comparaison visuelle avec la solution de Fourier

#%%
# Image restaurée par descente de gradient avec Tychonov
v_torch = torch.tensor(v, dtype=torch.float32)
lam = 2
lr_tych = 1/(1 + 16*lam)
u_rec, loss_values = optimize_image(v_torch, lam, lr_tych,
                                    reg_fn=tikhonov_reg, record_loss=True)
u_tych = to_numpy(u_rec)

# Image restaurée par solution explicite avec Fourier
u_fourier = tikhonov_fourier(v, lam)

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

#%%
# Print statistics for each image to check their ranges:
for img, name in zip([v, u_tych, u_fourier], ["Noisy", "PyTorch", "Fourier"]):
    arr = to_numpy(img)
    print(f"{name} min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}, shape: {arr.shape}, dtype: {arr.dtype}")

#%% [markdown]
# #### Visualisation avec variation des parametres

#%%
# valeurs de lambda
lambdas = [0.01, 0.02, 0.2, 0.5, 1, 2, 5, 10, 30]
lrs = [1 / (1 + 16 * lam) for lam in lambdas] 

psnr_fourier_list = []
psnr_pytorch_list = []
ssim_fourier_list = []
ssim_pytorch_list = []
mse_fourier_list = []
mse_pytorch_list = []
mae_fourier_list = []
mae_pytorch_list = []

v_torch_base = torch.tensor(v, dtype=torch.float32)

for lam, lr in zip(lambdas, lrs):

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
    u_pytorch, _ = optimize_image(v_torch, lam, lr, n_iter,
                                  reg_fn=tikhonov_reg, record_loss=True)

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
plt.plot(lambdas, psnr_fourier_n, marker='o', linestyle='-',  label='PSNR Fourier')
plt.plot(lambdas, ssim_fourier_n, marker='s', linestyle='-',  label='SSIM Fourier')
plt.plot(lambdas, mse_fourier_n,  marker='^', linestyle='-',  label='MSE Fourier')
plt.plot(lambdas, mae_fourier_n,  marker='d', linestyle='-',  label='MAE Fourier')

# PyTorch
plt.plot(lambdas, psnr_pytorch_n, marker='o', linestyle='--', label='PSNR PyTorch')
plt.plot(lambdas, ssim_pytorch_n, marker='s', linestyle='--', label='SSIM PyTorch')
plt.plot(lambdas, mse_pytorch_n,  marker='^', linestyle='--', label='MSE PyTorch')
plt.plot(lambdas, mae_pytorch_n,  marker='d', linestyle='--', label='MAE PyTorch')

plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Métriques normalisées')
plt.title('PSNR, SSIM, MSE et MAE normalisés en fonction de $\lambda$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ...

#%% [markdown]
# Application de la descente de gradient avec Total Variation comme fonction de régularisation

#%%
def tv_reg(u, eps = 0.001):
    gx, gy = gradient_x(u), gradient_y(u)
    return torch.sqrt(gx**2 + gy**2 + eps**2).sum()

v_torch = torch.tensor(v, dtype=torch.float32)
lam = 0.7
eps = 0.001
# obs for the report.. Learning Rate changes (new regularisation function), can talk about this in the annexe maybe
lr = 2/(1+8*lam/eps)

u_rec, loss_values = optimize_image(v_torch, lam, lr, n_iter = 6000, reg_fn=lambda u: tv_reg(u, eps=eps), record_loss=True)

plt.figure(figsize=(5,3))
plt.plot(loss_values)
plt.title("Évolution de la fonction de coût")
plt.xlabel("Itérations")
plt.ylabel("J(u)")
plt.grid(True)
plt.show()


# === compute convergence iteration ===
loss_arr = np.array(loss_values)
smooth_d = np.convolve(np.abs(np.diff(loss_arr)), np.ones(50)/50, mode='valid')
thr = 1e-1
idx = np.where(smooth_d < thr)[0]
print("Convergence at iteration:", idx[0] if len(idx)>0 else "not detected")

#%%
lambdas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 1.5, 2, 2.5]
eps = 0.001
lrs = [2/(1 + 8*lam/eps) for lam in lambdas]

psnr_tv, ssim_tv, mse_tv, mae_tv = [], [], [], []

for lam, lr in zip(lambdas, lrs):
    u_t = optimize_image(
        v_torch, lam, lr,
        n_iter=6000,
        reg_fn=lambda u: tv_reg(u, eps),
        record_loss=False
    )
    u_np = u_t.detach().cpu().numpy()
    psnr_tv.append(psnr_np(u0, u_np))
    ssim_tv.append(ssim(u0, u_np, data_range=1.0))
    mse_tv.append(np.mean((u0 - u_np)**2))
    mae_tv.append(np.mean(np.abs(u0 - u_np)))

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

plt.plot(lambdas, psnr_tv_n, marker='o', label='PSNR (normalisé)')
plt.plot(lambdas, ssim_tv_n, marker='s', label='SSIM (normalisé)')
plt.plot(lambdas, mse_tv_n,  marker='^', label='MSE (normalisé)')
plt.plot(lambdas, mae_tv_n,  marker='d', label='MAE (normalisé)')

plt.xscale('log')  # optionnel, tu peux enlever si tu veux un axe linéaire
plt.xlabel(r'$\lambda$')
plt.ylabel('Métriques normalisées')
plt.title('TV – PSNR, SSIM, MSE et MAE normalisés en fonction de $\lambda$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#%%
##### this is to plot both methods for the report .. we repeat the last tests but with the already explored and chosen parameters

# PyTorch tensors
v_torch = torch.tensor(v, dtype=torch.float32)
eps = 0.001

# --- Tikhonov ---
lam_tyk = 2
lr_tyk = 1 / (1 + 16*lam_tyk)
u_tyk= optimize_image(
    v_torch, lam_tyk, lr_tyk,
    n_iter=500,
    reg_fn=tikhonov_reg,
    record_loss=False
)
u_tyk_np = u_tyk.detach().cpu().numpy()

# --- Total Variation ---
lam_tv = 0.7
lr_tv = 1 / (1 + (8*lam_tv/eps))
u_tv = optimize_image(
    v_torch, lam_tv, lr_tv,
    n_iter=7000,
    reg_fn=lambda u: tv_reg(u, eps),
    record_loss=False
)
u_tv_np = u_tv.detach().cpu().numpy()

# --- show side by side ---
import matplotlib.pyplot as plt

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
#### this is code done to CONFIRM that the chosen parameters were ok.
# we finally decided that lambda=2 for tykhonov
# and lambda=0.7 for total variation is okay
# OBSERVATION. IMPORTANT. THIS ONLY WORKS FOR A CHOSEN SIGMA, maybe it would be good to try if these choices are the same for other sigmas (like images with less noise blabla)

pass
"""
v_torch = torch.tensor(v, dtype=torch.float32)

# ---- parameter grids ----
# tikhonov λs: small to large
lams_tyk = [0.01, 0.1, 0.5, 1, 2, 5, 10]

# TV λs and εs: wide range
lams_tv  = [0.01, 0.1, 0.3, 0.5, 0.7, 1, 2]
eps_list = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]

# ---- storage ----
results_tyk = {}
results_tv  = {}

# ---- Tikhonov ----
for lam in lams_tyk:
    lr = 1 / (1 + 16*lam)
    print(f"Running Tikhonov: lam={lam}")
    u = optimize_image(
        v_torch, lam, lr,
        n_iter=500,
        reg_fn=tikhonov_reg,
        record_loss=False
    )
    results_tyk[lam] = u.detach().cpu().numpy().squeeze()

# ---- Total Variation ----
for lam in lams_tv:
    for eps in eps_list:
        lr = 1 / (1 + (8*lam/eps))
        print(f"Running TV: lam={lam}, eps={eps}")
        u = optimize_image(
            v_torch, lam, lr,
            n_iter=5000,
            reg_fn=lambda u: tv_reg(u, eps),
            record_loss=False
        )
        results_tv[(lam, eps)] = u.detach().cpu().numpy().squeeze()

# ---- plot Tikhonov ----
fig, axes = plt.subplots(1, len(lams_tyk), figsize=(20,4))
for i, lam in enumerate(lams_tyk):
    axes[i].imshow(results_tyk[lam], cmap='gray')
    axes[i].set_title(f"λ={lam}")
    axes[i].axis('off')
plt.suptitle("Tikhonov Denoising")
plt.tight_layout()
plt.show()

# ---- plot TV ----
fig, axes = plt.subplots(len(lams_tv), len(eps_list), figsize=(18,12))
for i, lam in enumerate(lams_tv):
    for j, eps in enumerate(eps_list):
        axes[i,j].imshow(results_tv[(lam, eps)], cmap='gray')
        axes[i,j].set_title(f"λ={lam}, eps={eps}", fontsize=8)
        axes[i,j].axis('off')
plt.suptitle("Total Variation Denoising")
plt.tight_layout()
plt.show()
"""

#%%
# this code was only used to print the metrics to include in the report. shouldnt include it in the code we send
pass
"""
# Compute metrics
psnr_tyk = psnr_np(u0, u_tyk_np)
mse_tyk  = np.mean((u0 - u_tyk_np)**2)
mae_tyk  = np.mean(np.abs(u0 - u_tyk_np))
ssim_tyk = ssim(u0, u_tyk_np, data_range=1.0)

psnr_tv = psnr_np(u0, u_tv_np)
mse_tv  = np.mean((u0 - u_tv_np)**2)
mae_tv  = np.mean(np.abs(u0 - u_tv_np))
ssim_tv = ssim(u0, u_tv_np, data_range=1.0)

# Print in LaTeX-ready format
print("\\begin{tabular}{lcccc}")
print("\\hline")
print("\\textbf{Method} & \\textbf{PSNR [dB]} & \\textbf{MSE} & \\textbf{MAE} & \\textbf{SSIM} \\\\")
print("\\hline")
print(f"Tikhonov ($\\lambda=2$) & {psnr_tyk:.2f} & {mse_tyk:.4f} & {mae_tyk:.4f} & {ssim_tyk:.3f} \\\\")
print(f"Total Variation ($\\lambda=0.7$, $\\epsilon=0.001$) & {psnr_tv:.2f} & {mse_tv:.4f} & {mae_tv:.4f} & {ssim_tv:.3f} \\\\")
print("\\hline")
print("\\end{tabular}")
"""

#%%
# this code is to start the debluring part
# it was literally copied and pasted from the code arthur sended us. thats why it redefines functions and imports libraries blabla#
# we should clean it like see which parts are already defined

#########import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import torch
print(torch.__version__)

pi = torch.pi
def periodic_conv2d(u, k):
    """Periodic (FFT-based) 2D convolution. u, k: (M,N) float32 torch tensors."""
    uf = fft2(u)
    kf = fft2(k)
    return torch.real(ifft2(uf * kf))

def rgb2gray(u):
    return 0.2989 * u[:,:,0] + 0.5870 * u[:,:,1] + 0.1140 * u[:,:,2]

def str2(chars):
    return "{:.2f}".format(chars)

def psnr(uref,ut,M=1):
    mse = np.sqrt(np.mean((np.array(uref)-np.array(ut))**2))
    return 20*np.log10(M/mse)

def optim(f,niter=1000,lr=0.1):
    u = torch.randn(M,N, requires_grad=True)
    optimu = torch.optim.SGD([u], lr=lr)
    losslist = []
    for it in range(niter):
        loss = f(u)
        losslist.append(loss.detach())
        optimu.zero_grad()
        loss.backward()
        optimu.step()
    return u.detach(),losslist

# viewimage
import tempfile
import IPython
from skimage.transform import rescale

def viewimage(im, normalize=True,z=2,order=0,titre='',displayfilename=False):
    imin= np.array(im).astype(np.float32)
    channel_axis = 2 if len(u0.shape)>2 else None
    imin = rescale(imin, z, order=order, channel_axis=channel_axis)
    if normalize:
        imin-=imin.min()
        if imin.max()>0:
            imin/=imin.max()
    else:
        imin=imin.clip(0,255)/255
    imin=(imin*255).astype(np.uint8)
    filename=tempfile.mktemp(titre+'.png')
    if displayfilename:
        print (filename)
    plt.imsave(filename, imin, cmap='gray')
    IPython.display.display(IPython.display.Image(filename))


# alternative viewimage if the other one does not work:
def Viewimage(im,dpi=100,cmap='gray'):
    plt.figure(dpi=dpi)
    if cmap is None:
        plt.imshow(np.array(im))
    else:
        plt.imshow(np.array(im),cmap=cmap)
    plt.axis('off')
    plt.show()

# !wget https://perso.telecom-paristech.fr/aleclaire/mva/tpdeblur.zip
# !unzip tpdeblur.zip

#%% [markdown]
# # A) Deblurring with Tychonov and $\mathsf{TV}_\varepsilon$ regularizations

#%% [markdown]
# <br/>In this practical session, you have to fill the code at places marked ``### ... ###``

#%%
# Open the image u0y did at the begining
M,N = u0.shape

viewimage(u0)

#%%
# Load a blur kernel
kt = torch.tensor(np.loadtxt('kernels/kernel8.txt'))
# kt = np.loadtxt('kernels/levin7.txt')
(m,n) = kt.shape

plt.imshow(kt, cmap='gray')
plt.title('Blur kernel')
plt.axis('off')
plt.show()

# Embed the kernel in a MxN image, and put center at pixel (0,0)
k = torch.zeros((M,N))
k[0:m,0:n] = kt/torch.sum(kt)
k = torch.roll(k,(-int(m/2),-int(n/2)),(0,1))
fk = fft2(k)


fk = fk.to(dtype=torch.complex64)

#%%
# Compute the degraded image v = k*u0 + w  (convolution with periodic boundary conditions)


#sigma = 0.02
sigma = 0.1

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
plt.imsave('blurred_image.png', v.cpu().numpy(), cmap='gray')

#%%
# defines functions to start the  debluring process

# ---------- 3. loss functions for deblurring (as in your snippet) ----------
def deblur_tikhonov_loss(u, k, v, lam):
    conv_u = periodic_conv2d(u, k)
    fidelity = 0.5 * torch.sum((conv_u - v)**2)
    reg = lam * tikhonov_reg(u)
    return fidelity + reg

def deblur_tv_loss(u, k, v, lam, eps_tv=1e-3):
    conv_u = periodic_conv2d(u, k)
    fidelity = 0.5 * torch.sum((conv_u - v)**2)
    reg = lam * tv_reg(u, eps=eps_tv)
    return fidelity + reg

# ---------- 4. optimization runner function (keeps your style) ----------
def run_deblur(method='tv',
               lam=1.0,
               eps_tv=1e-3,
               n_iter=20000,
               k=None,
               v=None):
    """method: 'tv' or 'tikh'
       k : noyau de convolution (torch tensor)
       v : image floutée (torch tensor)
    """
    # fallback sur les variables globales si pas données
    if k is None:
        raise ValueError("k peut pas être None")
    else:
        k_torch = k.clone().detach().float()

    if v is None:
        raise ValueError("v peut pas être None")
    else:
        v_torch = v.clone().detach().float()

    # variable à optimiser
    guess = v_torch.clone().detach().float().requires_grad_(True)

    # ----- calcul de ||A^T A|| via FFT du noyau k -----
    H = torch.fft.fft2(k_torch, s=v_torch.shape[-2:])
    norm_A2 = (H.abs() ** 2).max().item()   # ≈ ||A^T A||

    if method == 'tv':
        # L ≈ ||A^T A|| + 8 λ / ε
        L = norm_A2 + 8.0 * lam / eps_tv
        lr = 1.9 / L
        loss_func = lambda U: deblur_tv_loss(U, k_torch, v_torch, lam, eps_tv=eps_tv)

    elif method == 'tikh':
        # L ≈ ||A^T A|| + 16 λ
        L = norm_A2 + 16.0 * lam
        lr = 0.9 / L
        loss_func = lambda U: deblur_tikhonov_loss(U, k_torch, v_torch, lam)

    else:
        raise ValueError("method must be 'tv' or 'tikh'")

    opt = torch.optim.SGD([guess], lr=lr)
    loss_values = []

    for i in range(n_iter):
        opt.zero_grad(set_to_none=True)
        loss = loss_func(guess)
        loss.backward()
        opt.step()
        with torch.no_grad():
            guess.clamp_(0, 1)

        loss_values.append(loss.item())

    restored = guess.detach().cpu().numpy()
    return restored, loss_values, lr



#%%
# Étude quantitative de l'influence de λ pour le deblurring Tikhonov


lambdas_tik = [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10]
#lambdas_tik = [0.01, 0.03, 0.05, 0.07, 0.1]
#lambdas_tik = [0.005, 0.007, 0.01, 0.03, 0.05]
psnr_tik, ssim_tik, mse_tik, mae_tik = [], [], [], []

for lam in lambdas_tik:
    # on peut adapter n_iter si nécessaire
    rest_tik_lam, loss_tik_lam, lr_tik_lam = run_deblur(
        method='tikh',
        lam=lam,
        n_iter=200,
        k=k,
        v=v
    )
    u_np = rest_tik_lam  # run_deblur renvoie déjà un numpy

    psnr_tik.append(psnr_np(u0, u_np))
    ssim_tik.append(ssim(u0, u_np, data_range=1.0))
    mse_tik.append(np.mean((u0 - u_np) ** 2))
    mae_tik.append(np.mean(np.abs(u0 - u_np)))

psnr_tik = np.array(psnr_tik)
ssim_tik = np.array(ssim_tik)
mse_tik  = np.array(mse_tik)
mae_tik  = np.array(mae_tik)

# Normalisation (0–1) par le max de chaque métrique
psnr_tik_n = psnr_tik / psnr_tik.max()
ssim_tik_n = ssim_tik / ssim_tik.max()
mse_tik_n  = mse_tik  / mse_tik.max()
mae_tik_n  = mae_tik  / mae_tik.max()

plt.figure(figsize=(8, 4))
plt.plot(lambdas_tik, psnr_tik_n, marker='o', label='PSNR (norm.)')
plt.plot(lambdas_tik, ssim_tik_n, marker='s', label='SSIM (norm.)')
plt.plot(lambdas_tik, mse_tik_n,  marker='^', label='MSE (norm.)')
plt.plot(lambdas_tik, mae_tik_n,  marker='d', label='MAE (norm.)')
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Métriques normalisées')
plt.title('Deblurring Tikhonov – métriques normalisées en fonction de $\lambda$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Affichage de la fonction de coût pour un seul λ

lam_plot = 0.05   # <-- choisis ici le lambda que tu veux visualiser
n_iter_plot = 200

_, loss_plot, lr_plot = run_deblur(
    method='tikh',
    lam=lam_plot,
    n_iter=n_iter_plot,
    k=k,
    v=v
)

iters = np.arange(len(loss_plot))

plt.figure(figsize=(7, 4))
plt.plot(iters, loss_plot)
plt.xlabel("Itérations")
plt.ylabel("Loss")
plt.title(f"Tikhonov – Fonction de coût pour λ={lam_plot} (lr={lr_plot:.2e})")
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Etude qualitatif : images restaurées cote à cote pour différents λ

lambdas_show = [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10]
n_iter_show = 200

rest_images = []

for lam in lambdas_show:
    restored, _, lr_tmp = run_deblur(
        method='tikh',
        lam=lam,
        n_iter=n_iter_show,
        k=k,
        v=v
    )
    rest_images.append((lam, restored, lr_tmp))

# --- création de la figure ---
plt.figure(figsize=(16, 3))

# --- colonne 1 : Original ---
plt.subplot(1, len(rest_images) + 2, 1)
plt.imshow(u0, cmap='gray')
plt.title("Original")
plt.axis("off")

# --- colonne 2 : Floutée ---
plt.subplot(1, len(rest_images) + 2, 2)
plt.imshow(v, cmap='gray')
plt.title("Blurred")
plt.axis("off")

# --- colonnes suivantes : λ différents ---
for i, (lam, img, lr) in enumerate(rest_images):
    plt.subplot(1, len(rest_images) + 2, i + 3)
    plt.imshow(img, cmap='gray')
    plt.title(f"λ={lam}\nlr={lr:.1e}", fontsize=8)
    plt.axis("off")

plt.suptitle("Deblurring Tikhonov – Influence qualitative de λ", fontsize=13)
plt.tight_layout()
plt.show()

#%%
# Étude quantitative de l'influence de λ pour le deblurring TV

lambdas_tv = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]
eps_tv = 0.001
psnr_tv, ssim_tv, mse_tv, mae_tv = [], [], [], []

# on stocke ici les images restaurées pour chaque λ
rest_tv_images = []   # liste de tuples (lambda, image, lr)

for lam in lambdas_tv:
    rest_tv_lam, loss_tv_lam, lr_tv_lam = run_deblur(
        method='tv',
        lam=lam,
        eps_tv=eps_tv,
        n_iter=1500,      # TV a besoin de plus d'itérations
        k=k,
        v=v
    )
    u_np = rest_tv_lam  # numpy

    # métriques
    psnr_tv.append(psnr_np(u0, u_np))
    ssim_tv.append(ssim(u0, u_np, data_range=1.0))
    mse_tv.append(np.mean((u0 - u_np) ** 2))
    mae_tv.append(np.mean(np.abs(u0 - u_np)))

    # stockage pour usage ultérieur (affichage qualitatif)
    rest_tv_images.append((lam, u_np, lr_tv_lam))

psnr_tv = np.array(psnr_tv)
ssim_tv = np.array(ssim_tv)
mse_tv  = np.array(mse_tv)
mae_tv  = np.array(mae_tv)

# Normalisation
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
plt.title('Deblurring TV – métriques normalisées en fonction de $\lambda$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Affichage qualitatif TV : Original + Floue + différentes valeurs de λ

plt.figure(figsize=(16, 3))

# Colonne 1 : Original
plt.subplot(1, len(rest_tv_images) + 2, 1)
plt.imshow(u0, cmap='gray')
plt.title("Original")
plt.axis("off")

# Colonne 2 : Floue
plt.subplot(1, len(rest_tv_images) + 2, 2)
plt.imshow(v, cmap='gray')
plt.title("Blurred")
plt.axis("off")

# Colonnes suivantes : résultats TV pour chaque λ
for i, (lam, img, lr_tv_lam) in enumerate(rest_tv_images):
    plt.subplot(1, len(rest_tv_images) + 2, i + 3)
    plt.imshow(img, cmap='gray')
    plt.title(f"λ={lam}\nlr={lr_tv_lam:.1e}", fontsize=8)
    plt.axis("off")

plt.suptitle("Deblurring TV – Influence qualitative de λ", fontsize=13)
plt.tight_layout()
plt.show()

#%%
# tries different parameters to choose which ones work best for the deblurring part

# ---- choose the parameter grid ----
lams = [0.03, 0.05]
eps_list = [0.005, 0.01, 0.015]

results = {}

# ---- run TV deblurring for each (λ, ε) ----
for lam in lams:
    for eps in eps_list:
        print(f"Running TV: lam={lam}, eps={eps}")
        rest, loss, lr_used = run_deblur(
            method='tv',
            lam=lam,
            eps_tv=eps,
            n_iter=1500,
            k = k, 
            v = v
        )

        # rest is already numpy → just squeeze
        results[(lam, eps)] = rest.squeeze()

# ---- plot ----
fig, axes = plt.subplots(len(lams), len(eps_list), figsize=(12, 10))

for i, lam in enumerate(lams):
    for j, eps in enumerate(eps_list):
        ax = axes[i, j]
        ax.imshow(results[(lam, eps)], cmap='gray')
        ax.set_title(f"λ={lam}, eps={eps}")
        ax.axis('off')

plt.tight_layout()
plt.show()

#%%
# ---------- 5. run both methods ----------
# Tikhonov (faster convergence)
lam_tik = 0.5
rest_tik, loss_tik, lr_tik = run_deblur(method='tikh', lam=lam_tik, n_iter=1500, k = k, v = v)

# TV (needs more iterations)
lam_tv = 0.03
eps_tv = 0.005
rest_tv, loss_tv, lr_tv = run_deblur(method='tv', lam=lam_tv, eps_tv=eps_tv, n_iter=1500, k = k, v = v)

# save results
plt.imsave('deblur_tikhonov.png', rest_tik, cmap='gray')
plt.imsave('deblur_tv.png', rest_tv, cmap='gray')

#%%
# === Affichage des images : Original, Floutée, Tikhonov, TV ===

plt.figure(figsize=(14, 4))

# Colonne 1 : Original
plt.subplot(1, 4, 1)
plt.imshow(u0, cmap='gray')
plt.title("Original")
plt.axis("off")

# Colonne 2 : Floutée
plt.subplot(1, 4, 2)
plt.imshow(v, cmap='gray')
plt.title("Blurred")
plt.axis("off")

# Colonne 3 : Tikhonov
plt.subplot(1, 4, 3)
plt.imshow(rest_tik, cmap='gray')
plt.title(f"Tikhonov\nλ={lam_tik}, lr={lr_tik:.2e}")
plt.axis("off")

# Colonne 4 : TV
plt.subplot(1, 4, 4)
plt.imshow(rest_tv, cmap='gray')
plt.title(f"TV\nλ={lam_tv}, ε={eps_tv}, lr={lr_tv:.2e}")
plt.axis("off")

plt.tight_layout()
plt.show()


#%%
# === Courbes des deux fonctions de perte ===

iters_tik = np.arange(len(loss_tik))
iters_tv  = np.arange(len(loss_tv))

plt.figure(figsize=(8, 4))
plt.plot(iters_tik, loss_tik, label='Tikhonov')
plt.plot(iters_tv,  loss_tv,  label='TV')
plt.xlabel('Itérations')
plt.ylabel('Loss')
plt.title('Évolution des fonctions de coût')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
