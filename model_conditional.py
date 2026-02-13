import math

import torch
import torch.nn as nn
from tqdm import tqdm

from unet_conditional import ConditionalUnet


class MNISTConditionalDiffusion(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        time_embedding_dim=256,
        timesteps=1000,
        base_dim=32,
        dim_mults=(1, 2, 4, 8),
        num_classes=10,
        cfg_dropout=0.1,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.null_label = num_classes
        self.cfg_dropout = cfg_dropout

        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        self.model = ConditionalUnet(
            timesteps=timesteps,
            time_embedding_dim=time_embedding_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            base_dim=base_dim,
            dim_mults=dim_mults,
            num_classes=num_classes,
        )

    def forward(self, x, noise, labels):
        # x: NCHW, labels: N
        labels = labels.to(x.device, dtype=torch.long)
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        x_t = self._forward_diffusion(x, t, noise)

        if self.training and self.cfg_dropout > 0.0:
            drop_mask = torch.rand_like(labels.float()) < self.cfg_dropout
            labels = labels.clone()
            labels[drop_mask] = self.null_label

        pred_noise = self.model(x_t, t, labels)
        return pred_noise

    @torch.no_grad()
    def sampling(
        self,
        n_samples,
        labels,
        guidance_scale=3.0,
        clipped_reverse_diffusion=True,
        device="cuda",
    ):
        x_t = torch.randn(
            (n_samples, self.in_channels, self.image_size, self.image_size),
            device=device,
        )
        labels = self._prepare_labels(labels, n_samples, device)

        for i in tqdm(
            range(self.timesteps - 1, -1, -1),
            desc="Conditional Sampling",
        ):
            noise = torch.randn_like(x_t)
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            pred = self._predict_noise(x_t, t, labels, guidance_scale)

            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise, pred)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise, pred)

        x_t = (x_t + 1.0) / 2.0  # [-1,1] to [0,1]
        return x_t

    def _prepare_labels(self, labels, n_samples, device):
        if isinstance(labels, int):
            labels = torch.full(
                (n_samples,),
                labels,
                device=device,
                dtype=torch.long,
            )
        elif isinstance(labels, (list, tuple)):
            labels = torch.tensor(labels, device=device, dtype=torch.long)
        else:
            labels = labels.to(device=device, dtype=torch.long)

        if labels.shape[0] != n_samples:
            raise ValueError(
                f"labels must have shape ({n_samples},), got {tuple(labels.shape)}"
            )
        if labels.min().item() < 0 or labels.max().item() >= self.num_classes:
            raise ValueError(
                "labels must be within "
                f"[0, {self.num_classes - 1}] for conditional sampling."
            )
        return labels

    @torch.no_grad()
    def _predict_noise(self, x_t, t, labels, guidance_scale):
        if guidance_scale <= 0.0:
            return self.model(x_t, t, labels)

        null_labels = torch.full_like(labels, self.null_label)
        pred_uncond = self.model(x_t, t, null_labels)
        pred_cond = self.model(x_t, t, labels)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(
            0,
            timesteps,
            steps=timesteps + 1,
            dtype=torch.float32,
        )
        f_t = torch.cos(
            ((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5
        ) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        return (
            self.sqrt_alphas_cumprod.gather(-1, t).reshape(
                x_0.shape[0], 1, 1, 1
            )
            * x_0
            + self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(
                x_0.shape[0], 1, 1, 1
            )
            * noise
        )

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise, pred):
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = (
            self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(
                x_t.shape[0], 1, 1, 1
            )
        )

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred
        )

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            std = 0.0

        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise, pred):
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(
            x_t.shape[0], 1, 1, 1
        )
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)

        x_0_pred = torch.sqrt(1.0 / alpha_t_cumprod) * x_t - torch.sqrt(
            1.0 / alpha_t_cumprod - 1.0
        ) * pred
        x_0_pred.clamp_(-1.0, 1.0)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(
                x_t.shape[0], 1, 1, 1
            )
            mean = (
                (
                    beta_t
                    * torch.sqrt(alpha_t_cumprod_prev)
                    / (1.0 - alpha_t_cumprod)
                )
                * x_0_pred
                + (
                    (1.0 - alpha_t_cumprod_prev)
                    * torch.sqrt(alpha_t)
                    / (1.0 - alpha_t_cumprod)
                )
                * x_t
            )
            std = torch.sqrt(
                beta_t * (1.0 - alpha_t_cumprod_prev) / (1.0 - alpha_t_cumprod)
            )
        else:
            mean = (beta_t / (1.0 - alpha_t_cumprod)) * x_0_pred
            std = 0.0

        return mean + std * noise
