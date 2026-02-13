import argparse
import math
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model_conditional import MNISTConditionalDiffusion
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
        description="Train conditional diffusion model on MNIST."
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default="", help="checkpoint path")
    parser.add_argument("--n_samples", type=int, default=36)
    parser.add_argument("--sample_label", type=int, default=1)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help=(
            "classifier-free guidance scale during sampling "
            "(0 disables guidance)"
        ),
    )
    parser.add_argument("--model_base_dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument(
        "--cfg_dropout",
        type=float,
        default=0.1,
        help="dropout probability of label conditioning in training",
    )
    parser.add_argument("--model_ema_steps", type=int, default=10)
    parser.add_argument("--model_ema_decay", type=float, default=0.995)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="use unclipped reverse process for sampling",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="force CPU execution"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help=(
            "name for this training run "
            "(outputs go to results_conditional/<run_name>)"
        ),
    )
    return parser.parse_args()


def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(args):
    if args.sample_label < 0 or args.sample_label >= args.num_classes:
        raise ValueError(
            "--sample_label must be in "
            f"[0, {args.num_classes - 1}], got {args.sample_label}"
        )

    device = get_device(force_cpu=args.cpu)
    train_dataloader = create_mnist_dataloader(
        batch_size=args.batch_size,
        image_size=28,
    )
    model = MNISTConditionalDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
        num_classes=args.num_classes,
        cfg_dropout=args.cfg_dropout,
    ).to(device)

    adjust = args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(
        model, device=device, decay=1.0 - alpha
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = OneCycleLR(
        optimizer,
        args.lr,
        total_steps=args.epochs * len(train_dataloader),
        pct_start=0.25,
        anneal_strategy="cos",
    )
    loss_fn = nn.MSELoss(reduction="mean")

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    if args.run_name:
        run_name = args.run_name
    elif args.ckpt:
        run_name = os.path.basename(os.path.dirname(args.ckpt))
        if not run_name:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    else:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    results_dir = os.path.join("results_conditional", run_name)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving checkpoints and samples to: {results_dir}")

    for epoch in range(args.epochs):
        model.train()
        for step, (image, label) in enumerate(train_dataloader):
            image = image.to(device)
            label = label.to(device)
            noise = torch.randn_like(image)

            pred = model(image, noise, label)
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
            n_samples=args.n_samples,
            labels=args.sample_label,
            guidance_scale=args.guidance_scale,
            clipped_reverse_diffusion=not args.no_clip,
            device=device,
        )
        save_image(
            samples,
            (
                f"{results_dir}/steps_{global_steps:0>8}_"
                f"label_{args.sample_label}.png"
            ),
            nrow=int(math.sqrt(args.n_samples)),
        )


if __name__ == "__main__":
    main(parse_args())
