"""
utils/gan_utils.py
==================
DCGAN model definitions, training helpers, and image generation utilities
for the Logo Detection pipeline.
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# ── Architecture constants ─────────────────────────────────────────────────────
NZ  = 100   # Latent vector dimensionality
NGF = 64    # Generator feature map base width
NDF = 64    # Discriminator feature map base width
NC  = 3     # Channels (RGB)
IMG_SIZE = 64  # Spatial resolution of GAN output


# ── Weight initialisation ──────────────────────────────────────────────────────
def weights_init(m: nn.Module) -> None:
    """Initialise Conv and BatchNorm weights per the DCGAN paper."""
    cname = m.__class__.__name__
    if 'Conv' in cname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in cname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ── Generator ──────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    DCGAN Generator: maps a latent vector Z → RGB image of size 64×64.

    Architecture (transpose-conv ladder):
        Z(NZ×1×1) → NGF*8×4×4 → NGF*4×8×8 → NGF*2×16×16
                   → NGF×32×32 → NC×64×64
    """
    def __init__(self, nz: int = NZ, ngf: int = NGF, nc: int = NC):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 — 1×1 → 4×4
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            # Block 2 — 4×4 → 8×8
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            # Block 3 — 8×8 → 16×16
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            # Block 4 — 16×16 → 32×32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # Output — 32×32 → 64×64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Discriminator ──────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    DCGAN Discriminator: maps NC×64×64 image → scalar real/fake probability.
    """
    def __init__(self, nc: int = NC, ndf: int = NDF):
        super().__init__()
        self.net = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32×32 → 16×16
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16×16 → 8×8
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8×8 → 4×4
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → 1×1
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Dataset ────────────────────────────────────────────────────────────────────
class LogoDataset(Dataset):
    """
    Dataset that loads logo images for GAN training.
    All images are resized to IMG_SIZE × IMG_SIZE and normalised to [-1, 1].
    """
    def __init__(self, img_dir: str, img_size: int = IMG_SIZE):
        self.paths = (
            list(Path(img_dir).glob('*.jpg')) +
            list(Path(img_dir).glob('*.png'))
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


# ── Training step helpers ──────────────────────────────────────────────────────
def train_discriminator(
    netD: Discriminator,
    netG: Generator,
    optimD: torch.optim.Optimizer,
    criterion: nn.Module,
    real_imgs: torch.Tensor,
    device: torch.device,
    nz: int = NZ,
) -> Tuple[float, torch.Tensor]:
    """One discriminator update step. Returns (loss, fake_imgs)."""
    bs = real_imgs.size(0)
    netD.zero_grad()

    # Real pass
    label = torch.full((bs,), 1.0, device=device)
    out   = netD(real_imgs.to(device))
    loss_real = criterion(out, label)
    loss_real.backward()

    # Fake pass
    noise = torch.randn(bs, nz, 1, 1, device=device)
    fake  = netG(noise)
    label.fill_(0.0)
    out   = netD(fake.detach())
    loss_fake = criterion(out, label)
    loss_fake.backward()
    optimD.step()

    return (loss_real + loss_fake).item(), fake


def train_generator(
    netD: Discriminator,
    netG: Generator,
    optimG: torch.optim.Optimizer,
    criterion: nn.Module,
    fake_imgs: torch.Tensor,
    device: torch.device,
) -> float:
    """One generator update step. Returns loss value."""
    bs = fake_imgs.size(0)
    netG.zero_grad()
    label = torch.full((bs,), 1.0, device=device)
    out   = netD(fake_imgs)
    loss  = criterion(out, label)
    loss.backward()
    optimG.step()
    return loss.item()


# ── Image generation ───────────────────────────────────────────────────────────
def generate_images(
    netG: Generator,
    device: torch.device,
    n: int = 64,
    nz: int = NZ,
    upscale_to: int = 640,
) -> List[np.ndarray]:
    """
    Generate `n` synthetic images with the trained Generator.
    Returns list of uint8 RGB numpy arrays at `upscale_to` × `upscale_to`.
    """
    netG.eval()
    images: List[np.ndarray] = []
    with torch.no_grad():
        batches = math.ceil(n / 64)
        for b in range(batches):
            bs    = min(64, n - len(images))
            noise = torch.randn(bs, nz, 1, 1, device=device)
            fakes = netG(noise).cpu()
            for j in range(bs):
                t   = fakes[j] * 0.5 + 0.5          # [-1,1] → [0,1]
                arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                arr = cv2.resize(arr, (upscale_to, upscale_to),
                                 interpolation=cv2.INTER_CUBIC)
                images.append(arr)
    return images


def save_generated_images(
    images: List[np.ndarray],
    out_dir: str,
    n_classes: int,
    prefix: str = 'gen',
) -> List[Path]:
    """
    Save generated images and write pseudo YOLO labels (centred bbox).
    Returns list of saved image paths.
    """
    out_dir  = Path(out_dir)
    img_dir  = out_dir / 'images'
    lbl_dir  = out_dir / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for i, img_rgb in enumerate(images):
        cls_idx = i % n_classes
        fname   = f'{prefix}_{i:05d}.jpg'
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_dir / fname), img_bgr)

        lbl_path = lbl_dir / fname.replace('.jpg', '.txt')
        # Centred pseudo-label: object covers 80% of the image
        lbl_path.write_text(
            f'{cls_idx} 0.500000 0.500000 0.800000 0.800000\n')
        saved.append(img_dir / fname)

    return saved


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_loss_curve(
    g_losses: List[float],
    d_losses: List[float],
    save_path: Optional[str] = None,
    title: str = 'DCGAN Training Loss',
) -> None:
    """Plot Generator and Discriminator loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs  = range(1, len(g_losses) + 1)
    ax.plot(epochs, g_losses, label='Generator',     color='#e63946', lw=2)
    ax.plot(epochs, d_losses, label='Discriminator', color='#457b9d', lw=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_image_grid(
    images: List[np.ndarray],
    nrow: int = 8,
    save_path: Optional[str] = None,
    title: str = 'Generated Samples',
) -> None:
    """Plot a grid of RGB images."""
    n   = min(len(images), nrow * nrow)
    nco = min(nrow, n)
    nro = math.ceil(n / nco)
    fig, axes = plt.subplots(nro, nco, figsize=(nco * 1.5, nro * 1.5))
    for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
        ax.axis('off')
    for i, ax in enumerate(axes.flat if hasattr(axes, 'flat') else [axes]):
        if i < n:
            ax.imshow(images[i])
    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def save_checkpoint(netG: Generator, netD: Discriminator,
                    optimG, optimD, epoch: int, path: str) -> None:
    torch.save({
        'epoch':        epoch,
        'netG':         netG.state_dict(),
        'netD':         netD.state_dict(),
        'optimG':       optimG.state_dict(),
        'optimD':       optimD.state_dict(),
    }, path)


def load_checkpoint(netG: Generator, netD: Discriminator,
                    optimG, optimD, path: str,
                    device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    netG.load_state_dict(ckpt['netG'])
    netD.load_state_dict(ckpt['netD'])
    optimG.load_state_dict(ckpt['optimG'])
    optimD.load_state_dict(ckpt['optimD'])
    return ckpt['epoch']
