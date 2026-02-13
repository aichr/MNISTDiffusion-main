import argparse
import math
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import MNISTDiffusion
from utils import ExponentialMovingAverage


def create_mnist_dataloader(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [0,1] to [-1,1]
        ]
    )

    train_dataset = MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess,
    )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Diffusion Model on MNIST")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckpt", type=str,
                        help="define checkpoint path", default="")
    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument("--model_base_dim", type=int,
                        help="base dim of Unet", default=64)
    parser.add_argument("--timesteps", type=int,
                        help="sampling steps of DDPM", default=1000)
    parser.add_argument(
        "--model_ema_steps", type=int, help="ema model evaluation interval", default=10
    )
    parser.add_argument(
        "--model_ema_decay", type=float, help="ema model decay", default=0.995
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=10,
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="set to normal sampling method without clip x_0 which could yield unstable samples",
    )
    args = parser.parse_args()

    return args


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(args):
    device = get_device()
    train_dataloader = create_mnist_dataloader(
        batch_size=args.batch_size,
        image_size=28,
    )
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
    ).to(device)

    # torchvision ema setting
    # https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(
        model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * len(train_dataloader),
        pct_start=0.25,
        anneal_strategy="cos",
    )
    loss_fn = nn.MSELoss(reduction='mean')

    # load checkpoint
    if args.ckpt:
        ckpt = torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for step, (image, _) in enumerate(train_dataloader):
            image = image.to(device)
            noise = torch.randn_like(image)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            if step % args.log_freq == 0:
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}],"
                    f"Step[{step}/{len(train_dataloader)}],"
                    f"loss:{loss.detach().cpu().item():.5f},"
                    f"lr:{scheduler.get_last_lr()[0]:.5f}"
                )
        ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
        torch.save(ckpt, f"{results_dir}/steps_{global_steps:0>8}.pt")

        model_ema.eval()
        samples = model_ema.module.sampling(
            args.n_samples,
            clipped_reverse_diffusion=not args.no_clip,
            device=device,
        )
        save_image(
            samples,
            f"{results_dir}/steps_{global_steps:0>8}.png",
            nrow=int(math.sqrt(args.n_samples)),
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
