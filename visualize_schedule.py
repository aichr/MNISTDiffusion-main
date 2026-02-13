"""Visualize the diffusion noise schedule and forward diffusion process.

Run:
    python visualize_schedule.py

Generates `figures/noise_schedule.png` with four panels:
  1. Beta schedule (noise variance per step)
  2. Alpha_cumprod (signal retained) & sqrt coefficients
  3. Signal-to-noise ratio (SNR) in dB
  4. Forward diffusion on a sample MNIST digit at selected timesteps
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST


# ── Cosine variance schedule (same as model.py) ──────────────────────────

def cosine_variance_schedule(timesteps, epsilon=0.008):
    steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
    f_t = torch.cos(
        ((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5
    ) ** 2
    betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
    return betas


# ── Linear variance schedule (for comparison) ────────────────────────────

def linear_variance_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


# ── Forward diffusion at arbitrary t (closed form) ───────────────────────

def forward_diffusion(x_0, t, noise, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod):
    """q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"""
    a = sqrt_alpha_cumprod[t]
    b = sqrt_one_minus_alpha_cumprod[t]
    return a * x_0 + b * noise


def main():
    timesteps = 1000

    # ── Compute schedules ─────────────────────────────────────────────
    betas_cos = cosine_variance_schedule(timesteps)
    betas_lin = linear_variance_schedule(timesteps)

    alphas_cos = 1.0 - betas_cos
    alphas_lin = 1.0 - betas_lin

    alpha_cumprod_cos = torch.cumprod(alphas_cos, dim=0)
    alpha_cumprod_lin = torch.cumprod(alphas_lin, dim=0)

    sqrt_alpha_cumprod_cos = torch.sqrt(alpha_cumprod_cos)
    sqrt_one_minus_cos = torch.sqrt(1.0 - alpha_cumprod_cos)

    t_axis = np.arange(timesteps)

    # ── Load one MNIST sample ─────────────────────────────────────────
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # [0,1] -> [-1,1]
    ])
    dataset = MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    sample_img, _ = dataset[0]  # shape: (1, 28, 28)
    noise = torch.randn_like(sample_img)

    # Timesteps to visualize forward diffusion
    demo_steps = [0, 50, 100, 200, 400, 600, 800, 999]

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.suptitle("Diffusion Noise Schedule Visualization (T=1000)", fontsize=16, fontweight="bold")

    gs = fig.add_gridspec(3, 2)

    # --- Panel 1: Beta schedule ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_axis, betas_cos.numpy(), label="Cosine (this model)", linewidth=1.5)
    ax1.plot(t_axis, betas_lin.numpy(), label="Linear (DDPM)", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Timestep t")
    ax1.set_ylabel(r"$\beta_t$")
    ax1.set_title(r"Beta Schedule — noise added per step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Alpha_cumprod and sqrt coefficients ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_axis, alpha_cumprod_cos.numpy(), label=r"$\bar{\alpha}_t$ (cosine)", linewidth=1.5)
    ax2.plot(t_axis, alpha_cumprod_lin.numpy(), label=r"$\bar{\alpha}_t$ (linear)", linewidth=1.5, alpha=0.7)
    ax2.plot(
        t_axis,
        sqrt_alpha_cumprod_cos.numpy(),
        "--",
        label=r"$\sqrt{\bar{\alpha}_t}$ (signal coeff)",
        linewidth=1.2,
    )
    ax2.plot(
        t_axis,
        sqrt_one_minus_cos.numpy(),
        "--",
        label=r"$\sqrt{1 - \bar{\alpha}_t}$ (noise coeff)",
        linewidth=1.2,
    )
    ax2.set_xlabel("Timestep t")
    ax2.set_title(r"Cumulative Signal & Noise Coefficients")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # --- Panel 3: SNR in dB ---
    ax3 = fig.add_subplot(gs[1, 0])
    snr_cos = alpha_cumprod_cos / (1.0 - alpha_cumprod_cos + 1e-8)
    snr_lin = alpha_cumprod_lin / (1.0 - alpha_cumprod_lin + 1e-8)
    ax3.plot(t_axis, 10 * torch.log10(snr_cos).numpy(), label="Cosine", linewidth=1.5)
    ax3.plot(t_axis, 10 * torch.log10(snr_lin).numpy(), label="Linear", linewidth=1.5, alpha=0.7)
    ax3.set_xlabel("Timestep t")
    ax3.set_ylabel("SNR (dB)")
    ax3.set_title("Signal-to-Noise Ratio over time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)

    # --- Panel 4: Cosine vs Linear alpha_bar side by side ---
    ax4 = fig.add_subplot(gs[1, 1])
    diff = alpha_cumprod_cos - alpha_cumprod_lin
    ax4.fill_between(t_axis, diff.numpy(), alpha=0.4, label="Cosine - Linear")
    ax4.plot(t_axis, diff.numpy(), linewidth=1.2)
    ax4.set_xlabel("Timestep t")
    ax4.set_ylabel(r"$\bar{\alpha}_{cos} - \bar{\alpha}_{lin}$")
    ax4.set_title("Cosine schedule preserves signal longer in early steps")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)

    # --- Panel 5: Forward diffusion on a sample image ---
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_title("Forward Diffusion: adding noise to a digit at selected timesteps", pad=10)
    ax5.axis("off")

    n_demo = len(demo_steps)
    inner_gs = gs[2, :].subgridspec(1, n_demo, wspace=0.05)

    for idx, step in enumerate(demo_steps):
        x_t = forward_diffusion(
            sample_img, step, noise, sqrt_alpha_cumprod_cos, sqrt_one_minus_cos
        )
        # Convert from [-1,1] to [0,1] for display
        img_display = (x_t.squeeze().numpy() + 1.0) / 2.0
        img_display = np.clip(img_display, 0.0, 1.0)

        ax = fig.add_subplot(inner_gs[0, idx])
        ax.imshow(img_display, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={step}\n" + r"$\bar{\alpha}$" + f"={alpha_cumprod_cos[step]:.3f}", fontsize=9)
        ax.axis("off")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/noise_schedule.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
