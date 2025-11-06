# ============================================================
# 1. DEBRUITAGE
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage import img_as_float

device = 'cpu'

# ------------------------------------------------------------
# utils
# ------------------------------------------------------------
def to_torch(x): return torch.tensor(x, dtype=torch.float32, device=device)
def to_numpy(x): return x.detach().cpu().numpy()

def show_images(imgs, titles=None, cmap='gray'):
    n = len(imgs)
    plt.figure(figsize=(4*n, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(to_numpy(imgs[i]), cmap=cmap)
        if titles: plt.title(titles[i])
        plt.axis('off')
    plt.show()

# synthetic clean + noisy image
u_true = to_torch(img_as_float(shepp_logan_phantom()))
v_noisy = u_true + 0.1 * torch.randn_like(u_true)

show_images([u_true, v_noisy], ["u (true)", "v (noisy)"])


# ============================================================
# 1.1 METHODS
# ============================================================

# ------------------------------------------------------------
# 1.1.1 Tikhonov analytique (Fourier)
# ------------------------------------------------------------

def tikhonov_fourier(v, lam):
    ny, nx = v.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    L2 = (fx**2 + fy**2)
    H = 1 / (1 + lam * (4 * np.pi**2) * L2)
    V = np.fft.fft2(to_numpy(v))
    U = np.fft.ifft2(V * H)
    return to_torch(np.real(U)), H

lam = 0.5
u_tik_fourier, H = tikhonov_fourier(v_noisy, lam)
show_images([v_noisy, u_tik_fourier], ["noisy", f"Tikhonov Fourier λ={lam}"])
print("PSNR:", psnr(to_numpy(u_true), to_numpy(u_tik_fourier)))

# optional: show filter response
plt.imshow(np.fft.fftshift(H), cmap='magma')
plt.title("Filter response H(f)")
plt.colorbar(); plt.show()


# ------------------------------------------------------------
# 1.1.2 Tikhonov numérique (descente de gradient)
# ------------------------------------------------------------

def tikhonov_reg(u, v, lam):
    grad_y = u[1:, :] - u[:-1, :]
    grad_x = u[:, 1:] - u[:, :-1]
    return 0.5 * torch.sum((u - v)**2) + 0.5 * lam * (torch.sum(grad_x**2) + torch.sum(grad_y**2))

def gradient_descent_tikhonov(v, lam, lr=0.2, n_iter=100):
    u = v.clone().requires_grad_(True)
    losses = []
    for i in range(n_iter):
        loss = tikhonov_reg(u, v, lam)
        loss.backward()
        with torch.no_grad():
            u -= lr * u.grad
            u.grad.zero_()
        losses.append(loss.item())
    return u.detach(), losses

u_tik_gd, losses_tik = gradient_descent_tikhonov(v_noisy, lam=0.5, lr=0.3, n_iter=100)
plt.plot(losses_tik); plt.title("Convergence Tikhonov GD"); plt.xlabel("iteration"); plt.ylabel("loss"); plt.show()

show_images([v_noisy, u_tik_gd], ["noisy", "Tikhonov GD"])
print("PSNR:", psnr(to_numpy(u_true), to_numpy(u_tik_gd)))


# ------------------------------------------------------------
# 1.1.3 Variation Totale (TV)
# ------------------------------------------------------------

def tv_reg(u, v, lam, eps=1e-3):
    grad_y = u[1:, :] - u[:-1, :]
    grad_x = u[:, 1:] - u[:, :-1]
    tv_norm = torch.sum(torch.sqrt(grad_x**2 + grad_y**2 + eps))
    return 0.5 * torch.sum((u - v)**2) + lam * tv_norm

def gradient_descent_tv(v, lam, lr=0.1, n_iter=100):
    u = v.clone().requires_grad_(True)
    losses = []
    for i in range(n_iter):
        loss = tv_reg(u, v, lam)
        loss.backward()
        with torch.no_grad():
            u -= lr * u.grad
            u.grad.zero_()
        losses.append(loss.item())
    return u.detach(), losses

u_tv, losses_tv = gradient_descent_tv(v_noisy, lam=0.05, lr=0.2, n_iter=100)
plt.plot(losses_tv); plt.title("Convergence TV"); plt.xlabel("iteration"); plt.ylabel("loss"); plt.show()

show_images([v_noisy, u_tv], ["noisy", "TV λ=0.05"])
print("PSNR:", psnr(to_numpy(u_true), to_numpy(u_tv)))


# ------------------------------------------------------------
# 1.1.3.1 Inpainting avec TV
# ------------------------------------------------------------

mask = torch.ones_like(u_true)
mask[50:100, 70:130] = 0
v_incomplete = v_noisy * mask

def inpainting_tv(v, mask, lam, lr=0.1, n_iter=200):
    u = v.clone().requires_grad_(True)
    for _ in range(n_iter):
        loss = tv_reg(u*mask + v*(1-mask), v, lam)
        loss.backward()
        with torch.no_grad():
            u -= lr * u.grad
            u.grad.zero_()
    return u.detach()

u_inpaint = inpainting_tv(v_incomplete, mask, lam=0.1)
show_images([v_incomplete, u_inpaint], ["incomplete", "inpainted (TV)"])


# ============================================================
# 1.2 PARAMÈTRES : Impact & Choix
# ============================================================

lams = [0.01, 0.1, 0.5, 1.0]
results = []
for l in lams:
    u_t, _ = gradient_descent_tikhonov(v_noisy, lam=l, lr=0.3, n_iter=100)
    results.append((l, u_t, psnr(to_numpy(u_true), to_numpy(u_t))))
show_images([r[1] for r in results], [f"λ={r[0]} | PSNR={r[2]:.2f}" for r in results])
print("PSNR by λ:", [(r[0], round(r[2],2)) for r in results])


# ============================================================
# 1.3 CONCLUSION
# ============================================================

def metrics_table(u_true, denoisers):
    print(f"{'Method':25s} {'PSNR':>8s} {'SSIM':>8s}")
    for name, u in denoisers.items():
        print(f"{name:25s} {psnr(to_numpy(u_true), to_numpy(u)):.3f} {ssim(to_numpy(u_true), to_numpy(u)):.3f}")

denoisers = {
    "Tikhonov Fourier": u_tik_fourier,
    "Tikhonov GD": u_tik_gd,
    "TV": u_tv
}
metrics_table(u_true, denoisers)

# visualize final comparison
show_images([u_true, v_noisy, u_tik_fourier, u_tik_gd, u_tv],
            ["true", "noisy", "Tikhonov Fourier", "Tikhonov GD", "TV"])

print("""
- TV yields best perceptual results (edges preserved, higher SSIM).
- Tikhonov (Fourier or GD) gives smoother but blurrier outputs.
- PSNR slightly higher for Tikhonov; SSIM higher for TV.
→ Next step: go to DEFLOUAGE (deblurring with A known).
""")

# -*- coding: utf-8 -*-
"""
Sections:
2. DEBLOUAGE (Déflouage / Partial Reconstruction)
    2.1 Methods
        2.1.1 Define operator A (Gaussian blur)
        2.1.2 Wiener deconvolution (Fourier baseline)
        2.1.3 Tikhonov deconvolution in Fourier (with Laplacian prior)
        2.1.4 Gradient-based reconstruction (known A) - Tikhonov & TV (PyTorch)
    2.2 Parameters (impact & choice) - show images
    2.3 Conclusion (PSNR / SSIM / which is best)
3. RESULTS & SUMMARY
    - summary tables, plots and zoom crops
"""

# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim
import torch
import pandas as pd

# === Utilities ===
def rmse(ref, est):
    return np.sqrt(np.mean((ref - est) ** 2))

def psnr(ref, est):
    e = rmse(ref, est)
    return np.inf if e == 0 else 20 * np.log10(1.0 / e)

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

def show_images(imgs, titles=None, cmap='gray', figsize=(12,3), suptitle=None):
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axs = [axs]
    for ax, im, t in zip(axs, imgs, titles or ['']*n):
        ax.imshow(np.clip(to_numpy(im),0,1), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(t)
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    plt.show()

def crop(img, center=None, size=120):
    arr = to_numpy(img)
    h, w = arr.shape
    if center is None:
        cy, cx = h//2, w//2
    else:
        cy, cx = center
    half = size // 2
    y0, y1 = max(0, cy-half), min(h, cy+half)
    x0, x1 = max(0, cx-half), min(w, cx+half)
    return arr[y0:y1, x0:x1]

# === Prepare test image and noisy blurred observation (consistent baseline) ===
u0 = img_as_float(data.camera())  # reference clean image in [0,1]
np.random.seed(0)
sigma_noise = 0.02  # observation noise level for blur experiment
# We'll create blurred observation for deflouage experiments below.

# ================================================================
# 2. DEBLURING
# ================================================================
# -*- coding: utf-8 -*-
"""
Sections:
2. DEBLOUAGE (Déflouage / Partial Reconstruction)
    2.1 Methods
        2.1.1 Define operator A (Gaussian blur)
        2.1.2 Wiener deconvolution (Fourier baseline)
        2.1.3 Tikhonov deconvolution in Fourier (with Laplacian prior)
        2.1.4 Gradient-based reconstruction (known A) - Tikhonov & TV (PyTorch)
    2.2 Parameters (impact & choice) - show images
    2.3 Conclusion (PSNR / SSIM / which is best)
3. RESULTS & SUMMARY
    - summary tables, plots and zoom crops
"""

# === Imports ===
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import structural_similarity as ssim
import torch
import pandas as pd

# === Utilities ===
def rmse(ref, est):
    return np.sqrt(np.mean((ref - est) ** 2))

def psnr(ref, est):
    e = rmse(ref, est)
    return np.inf if e == 0 else 20 * np.log10(1.0 / e)

def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

def show_images(imgs, titles=None, cmap='gray', figsize=(12,3), suptitle=None):
    n = len(imgs)
    fig, axs = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axs = [axs]
    for ax, im, t in zip(axs, imgs, titles or ['']*n):
        ax.imshow(np.clip(to_numpy(im),0,1), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(t)
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    plt.show()

def crop(img, center=None, size=120):
    arr = to_numpy(img)
    h, w = arr.shape
    if center is None:
        cy, cx = h//2, w//2
    else:
        cy, cx = center
    half = size // 2
    y0, y1 = max(0, cy-half), min(h, cy+half)
    x0, x1 = max(0, cx-half), min(w, cx+half)
    return arr[y0:y1, x0:x1]

# === Prepare test image and noisy blurred observation (consistent baseline) ===
u0 = img_as_float(data.camera())  # reference clean image in [0,1]
np.random.seed(0)
sigma_noise = 0.02  # observation noise level for blur experiment
# We'll create blurred observation for deflouage experiments below.

# ================================================================
# 2. DEBLOUAGE
# ================================================================

# --------------------------
# 2.1 METHODS
# --------------------------

# 2.1.1 Define operator A - gaussian blur kernel (numpy + torch variants)
def gaussian_kernel_numpy(size=9, sigma=2.0):
    ax = np.linspace(-(size-1)/2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2*sigma**2))
    k = k / np.sum(k)
    return k

def pad_kernel_to_image(kernel, image_shape):
    kh, kw = kernel.shape
    H, W = image_shape
    pad = np.zeros((H, W), dtype=np.float32)
    pad[:kh, :kw] = kernel  # top-left
    # roll to center kernel
    pad = np.roll(np.roll(pad, -kh//2, axis=0), -kw//2, axis=1)
    return pad

def apply_A_torch(u_torch, kernel_torch):
    pad = kernel_torch.shape[0]//2
    u = u_torch.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
    k = kernel_torch.unsqueeze(0).unsqueeze(0)  # 1x1xkxk
    v = torch.nn.functional.conv2d(u, k, padding=pad)
    return v.squeeze(0).squeeze(0)

# Create kernels of different blur strength to experiment with
kernels = {
    'sigma=1.0': gaussian_kernel_numpy(size=9, sigma=1.0),
    'sigma=2.0': gaussian_kernel_numpy(size=9, sigma=2.0),
    'sigma=3.0': gaussian_kernel_numpy(size=9, sigma=3.0)
}

# show example kernel and its spectrum
example_sigma = 'sigma=2.0'
k_example = kernels[example_sigma]
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(k_example, cmap='gray'); plt.title(f'Kernel (spatial) {example_sigma}'); plt.axis('off')
plt.subplot(1,2,2)
Kfft = np.abs(fftshift(fft2(pad_kernel_to_image(k_example, u0.shape))))
plt.imshow(np.log(Kfft + 1e-8), cmap='magma'); plt.title('log |FFT(kernel)|'); plt.axis('off')
plt.show()

# Prepare blurred + noisy observation using the chosen kernel
kernel_used = k_example
u_true = u0.copy()
v_blur = np.real(ifft2(fft2(u_true) * fft2(pad_kernel_to_image(kernel_used, u_true.shape))))  # convolution
v_blur = v_blur + sigma_noise * np.random.randn(*v_blur.shape)  # add small noise
v_blur = np.clip(v_blur, 0, 1)
show_images([u_true, v_blur], ["Original", "Blurred + noise"], figsize=(10,4))

# 2.1.2 Wiener deconvolution (Fourier baseline)
def wiener_deconvolution(observed, kernel, K=1e-3):
    # observed, kernel: numpy arrays, kernel smaller than image
    Hpad = pad_kernel_to_image(kernel, observed.shape)
    H = fft2(Hpad)
    V = fft2(observed)
    H_conj = np.conj(H)
    denom = (H * H_conj) + K
    U = (H_conj / denom) * V
    u_rec = np.real(ifft2(U))
    return np.clip(u_rec, 0, 1)

u_wiener = wiener_deconvolution(v_blur, kernel_used, K=1e-3)
print("Wiener PSNR:", psnr(u_true, u_wiener), "SSIM:", ssim(u_true, u_wiener))

# show wiener result
show_images([v_blur, u_wiener], ["Blurred", "Wiener deconv (baseline)"])

# 2.1.3 Tikhonov deconvolution in Fourier (with Laplacian prior)
def laplacian_spectrum(shape):
    # discrete Laplacian spectrum for periodic BC: L_hat(ω) = 4[sin^2(pi kx/N) + sin^2(pi ky/M)]
    M, N = shape
    xi = np.arange(M)[:, None]
    zeta = np.arange(N)[None, :]
    L_hat = 4 * (np.sin(np.pi * xi / M)**2 + np.sin(np.pi * zeta / N)**2)
    return L_hat

def tikhonov_deconv_fourier(observed, kernel, lam):
    Hpad = pad_kernel_to_image(kernel, observed.shape)
    H = fft2(Hpad)
    V = fft2(observed)
    L = laplacian_spectrum(observed.shape)
    # filter = conj(H) / (|H|^2 + lam * L_spectral)
    denom = (np.abs(H)**2) + (lam * L)
    # avoid division by zero small values
    denom = denom + 1e-12
    U = (np.conj(H) / denom) * V
    u_rec = np.real(ifft2(U))
    return np.clip(u_rec, 0, 1)

# test Tikhonov Fourier for a few lambda
lams = [1e-4, 1e-3, 1e-2, 1e-1]
tikhonov_fourier_results = {}
for lam in lams:
    r = tikhonov_deconv_fourier(v_blur, kernel_used, lam)
    tikhonov_fourier_results[lam] = r
    print(f"Tikhonov Fourier λ={lam}: PSNR={psnr(u_true, r):.2f}, SSIM={ssim(u_true, r):.4f}")

# display a few
show_images([v_blur, tikhonov_fourier_results[lams[1]], tikhonov_fourier_results[lams[2]]],
            ["Blurred", f"Tikh-Fourier λ={lams[1]}", f"Tikh-Fourier λ={lams[2]}"])

# 2.1.4 Gradient-based reconstruction with known A (PyTorch)
# We'll implement two versions: Tikhonov (quadratic on gradients) and TV (isotropic)
def gaussian_kernel_torch(kernel_numpy):
    return torch.tensor(kernel_numpy, dtype=torch.float32)

kernel_torch = gaussian_kernel_torch(kernel_used)
u_true_t = torch.tensor(u_true, dtype=torch.float32)
v_blur_t = torch.tensor(v_blur, dtype=torch.float32)

# gradient operators (torch)
def grad_x(u):
    return torch.cat((u[:, 1:] - u[:, :-1], torch.zeros(u.shape[0], 1, device=u.device)), dim=1)

def grad_y(u):
    return torch.cat((u[1:, :] - u[:-1, :], torch.zeros(1, u.shape[1], device=u.device)), dim=0)

def tikhonov_reg_torch(u):
    gx = grad_x(u)
    gy = grad_y(u)
    return (gx**2 + gy**2).mean()

def tv_reg_torch(u, eps=1e-6):
    gx = grad_x(u)
    gy = grad_y(u)
    return torch.sqrt(gx**2 + gy**2 + eps).mean()

def reconstruct_with_A_torch(v_t, kernel_t, lam=0.05, lr=0.02, n_iter=300, reg='tikh'):
    # v_t: torch HxW, kernel_t: torch kxk
    u = v_t.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    losses = []
    for i in range(n_iter):
        opt.zero_grad()
        Av = apply_A_torch(u, kernel_t)
        fidelity = 0.5 * ((Av - v_t)**2).mean()
        reg_term = lam * (tikhonov_reg_torch(u) if reg=='tikh' else tv_reg_torch(u))
        loss = fidelity + reg_term
        loss.backward()
        opt.step()
        with torch.no_grad():
            u.clamp_(0, 1)
        losses.append(float(loss.item()))
    return u.detach(), losses

# run reconstructions (Tikhonov-GD and TV-GD)
u_rec_tikh_torch, losses_rec_tikh = reconstruct_with_A_torch(v_blur_t, kernel_torch, lam=0.05, lr=0.02, n_iter=400, reg='tikh')
u_rec_tv_torch, losses_rec_tv = reconstruct_with_A_torch(v_blur_t, kernel_torch, lam=0.05, lr=0.02, n_iter=400, reg='tv')

u_rec_tikh = to_numpy(u_rec_tikh_torch)
u_rec_tv = to_numpy(u_rec_tv_torch)

# visualize reconstructions
show_images([u_true, v_blur, u_rec_tikh, u_rec_tv, u_wiener],
            ["True", "Blurred", "Tikhonov (GD)", "TV (GD)", "Wiener (Fourier)"], figsize=(15,3))

# plot reconstruction losses
plt.figure(figsize=(6,4))
plt.plot(losses_rec_tikh, label='Tikhonov (GD)')
plt.plot(losses_rec_tv, label='TV (GD)')
plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.title('Reconstruction loss evolution'); plt.legend(); plt.grid(True); plt.show()

# --------------------------
# 2.2 PARAMETERS: impact & choice (show images)
# --------------------------
# Evaluate PSNR/SSIM for various lambdas for each method (Wiener is K param)
lambdas_eval = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
records = []
for lam in lambdas_eval:
    # Tikhonov Fourier
    r_tikf = tikhonov_deconv_fourier(v_blur, kernel_used, lam)
    # Tikhonov GD (shorter runs to keep time reasonable)
    u_tik_gd_t, _ = reconstruct_with_A_torch(v_blur_t, kernel_torch, lam=lam, lr=0.02, n_iter=250, reg='tikh')
    r_tik_gd = to_numpy(u_tik_gd_t)
    # TV GD
    u_tv_gd_t, _ = reconstruct_with_A_torch(v_blur_t, kernel_torch, lam=lam, lr=0.02, n_iter=250, reg='tv')
    r_tv_gd = to_numpy(u_tv_gd_t)
    # Wiener baseline (we keep K small fixed)
    r_wien = wiener_deconvolution(v_blur, kernel_used, K=1e-3)
    # metrics
    records.append({
        'lambda': lam,
        'PSNR_Tikh_Fourier': psnr(u_true, r_tikf),
        'SSIM_Tikh_Fourier': ssim(u_true, r_tikf),
        'PSNR_Tikh_GD': psnr(u_true, r_tik_gd),
        'SSIM_Tikh_GD': ssim(u_true, r_tik_gd),
        'PSNR_TV_GD': psnr(u_true, r_tv_gd),
        'SSIM_TV_GD': ssim(u_true, r_tv_gd),
        'PSNR_Wiener': psnr(u_true, r_wien),
        'SSIM_Wiener': ssim(u_true, r_wien)
    })
    print(f"lambda={lam}: TikhF {records[-1]['PSNR_Tikh_Fourier']:.2f} dB, TikhGD {records[-1]['PSNR_Tikh_GD']:.2f}, TV {records[-1]['PSNR_TV_GD']:.2f}")

df_params = pd.DataFrame(records)
print("\nParameter sweep results (excerpt):")
print(df_params[['lambda','PSNR_Tikh_Fourier','PSNR_Tikh_GD','PSNR_TV_GD','PSNR_Wiener']])

# visualize PSNR vs lambda
plt.figure(figsize=(6,4))
plt.plot(df_params['lambda'], df_params['PSNR_Tikh_Fourier'], marker='o', label='Tikhonov Fourier')
plt.plot(df_params['lambda'], df_params['PSNR_Tikh_GD'], marker='x', label='Tikhonov GD')
plt.plot(df_params['lambda'], df_params['PSNR_TV_GD'], marker='s', label='TV GD')
plt.xscale('log')
plt.xlabel('λ (log scale)'); plt.ylabel('PSNR [dB]'); plt.title('PSNR vs λ (reconstruction)'); plt.legend(); plt.grid(True); plt.show()

# Show effect of kernel sigma (impact of blur severity)
sigma_list = [1.0, 2.0, 3.0]
sigma_records = []
for s in sigma_list:
    k = gaussian_kernel_numpy(size=9, sigma=s)
    # recreate blurred image
    vb = np.real(ifft2(fft2(u_true) * fft2(pad_kernel_to_image(k, u_true.shape))))
    vb = vb + sigma_noise * np.random.randn(*vb.shape)
    vb = np.clip(vb, 0, 1)
    # Wiener
    w = wiener_deconvolution(vb, k, K=1e-3)
    # Tikhonov Fourier with a fixed lam
    tf = tikhonov_deconv_fourier(vb, k, lam=1e-2)
    # Tikhonov GD quick
    k_torch = gaussian_kernel_torch(k)
    vb_t = torch.tensor(vb, dtype=torch.float32)
    tg, _ = reconstruct_with_A_torch(vb_t, k_torch, lam=0.05, lr=0.02, n_iter=200, reg='tikh')
    tvg, _ = reconstruct_with_A_torch(vb_t, k_torch, lam=0.05, lr=0.02, n_iter=200, reg='tv')
    sigma_records.append({
        'sigma': s,
        'PSNR_Blurred': psnr(u_true, vb),
        'PSNR_Wiener': psnr(u_true, w),
        'PSNR_Tikh_Fourier': psnr(u_true, tf),
        'PSNR_Tikh_GD': psnr(u_true, to_numpy(tg)),
        'PSNR_TV_GD': psnr(u_true, to_numpy(tvg))
    })
    print(f"sigma={s}: blurred PSNR {sigma_records[-1]['PSNR_Blurred']:.2f}, Wiener {sigma_records[-1]['PSNR_Wiener']:.2f}")

df_sigma = pd.DataFrame(sigma_records)
print("\nBlur severity sweep (PSNR):")
print(df_sigma)

# --------------------------
# 2.3 CONCLUSION (within code we compute summarized metrics)
# --------------------------
# Pick one representative lambda result to compare methods side-by-side (from earlier)
lam_choice = 1e-2
u_tikh_f_chosen = tikhonov_deconv_fourier(v_blur, kernel_used, lam_choice)
# recompute GD reconstructions for the chosen lambda with more iterations
u_tikh_gd_ch_t, _ = reconstruct_with_A_torch(v_blur_t, kernel_torch, lam=lam_choice, lr=0.02, n_iter=400, reg='tikh')
u_tv_gd_ch_t, _ = reconstruct_with_A_torch(v_blur_t, kernel_torch, lam=lam_choice, lr=0.02, n_iter=400, reg='tv')
u_tikh_gd_ch = to_numpy(u_tikh_gd_ch_t)
u_tv_gd_ch = to_numpy(u_tv_gd_ch_t)
u_wien_ch = u_wiener

summary = [
    ('Blurred (observed)', v_blur, psnr(u_true, v_blur), ssim(u_true, v_blur)),
    ('Wiener (Fourier)', u_wien_ch, psnr(u_true, u_wien_ch), ssim(u_true, u_wien_ch)),
    ('Tikhonov (Fourier)', u_tikh_f_chosen, psnr(u_true, u_tikh_f_chosen), ssim(u_true, u_tikh_f_chosen)),
    ('Tikhonov (GD)', u_tikh_gd_ch, psnr(u_true, u_tikh_gd_ch), ssim(u_true, u_tikh_gd_ch)),
    ('TV (GD)', u_tv_gd_ch, psnr(u_true, u_tv_gd_ch), ssim(u_true, u_tv_gd_ch)),
]

df_summary = pd.DataFrame([{'method': name, 'PSNR': p, 'SSIM': s} for name,_,p,s in summary])
print("\n--- Reconstruction summary (chosen λ = {:.1e}) ---".format(lam_choice))
print(df_summary)

# show side-by-side
show_images([s[1] for s in summary],
            [f"{s[0]}\nPSNR={s[2]:.2f}, SSIM={s[3]:.3f}" for s in summary],
            figsize=(16,3), suptitle="Comparison of reconstruction methods")

# zoom crop to inspect details
center = (300, 300)
show_images([crop(s[1], center=center, size=120) for s in summary],
            [f"{s[0]}\nPSNR={s[2]:.2f}" for s in summary],
            figsize=(16,3), suptitle="Zoom (center) - details")

# ================================================================
# 3. RESULTS & SUMMARY
# ================================================================
# Save tables (optional)
# df_params.to_csv('recon_params_sweep.csv', index=False)
# df_sigma.to_csv('recon_sigma_sweep.csv', index=False)
# df_summary.to_csv('recon_summary.csv', index=False)

# Final concise textual summary (printed)
print("\nFINAL CONCLUSIONS (automatically generated):")
best_by_psnr = df_summary.iloc[df_summary['PSNR'].argmax()]['method']
best_by_ssim = df_summary.iloc[df_summary['SSIM'].argmax()]['method']
print(f"- Best method by PSNR: {best_by_psnr}")
print(f"- Best method by SSIM: {best_by_ssim}")
print("""
Observations:
- Wiener (Fourier) is fast and can give reasonable restoration when blur is mild and SNR is OK.
- Tikhonov (Fourier) gives a principled deconvolution with a Laplacian prior: parameter λ controls smoothing vs amplification of high frequencies.
- Gradient-based Tikhonov (GD) and TV (GD) allow flexible priors and constrained optimization (e.g., non-negativity) but are slower.
- TV yields better edge preservation (higher SSIM) while Tikhonov may get higher PSNR in some cases due to smoother energy distribution.
- Parameter selection (λ, Wiener K) is crucial. Use PSNR/SSIM sweeps or cross-validation on synthetic data to pick λ.
Next steps / Perspectives:
- Implement automatic λ selection (L-curve, generalized cross-validation) or plug-and-play priors.
- Try blind deconvolution if A is unknown, or deep-learning based deblurring for greater performance.
""")
