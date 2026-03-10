r"""
Variational Autoencoder (VAE) with reparameterization trick on MNIST.

Mathematical formulation:
    ELBO (Evidence Lower Bound):
        $\mathcal{L}(\theta, \phi; \mathbf{x}) =
         \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]
         - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$

    Reparameterization trick:
        $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon},
         \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

    KL divergence (closed form for diagonal Gaussian vs standard normal):
        $D_{KL}(q \| p) = -\frac{1}{2} \sum_{j=1}^{J}
         \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$

    Reconstruction loss: Binary Cross-Entropy per pixel.

Trains on MNIST, generates samples, checks loss convergence.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'ae_lvl3_vae',
        'task_type': 'generative',
        'algorithm': 'Variational Autoencoder (VAE)',
        'input_shape': [1, 28, 28],
        'latent_dim': 20,
        'description': 'VAE with reparameterization trick, KL + reconstruction loss on MNIST'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the device for training."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    """Variational Autoencoder with MLP encoder and decoder."""

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: x -> hidden -> (mu, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z -> hidden -> x_recon
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


def vae_loss(x_recon, x, mu, log_var):
    """
    Compute VAE loss = Reconstruction + KL divergence.
    Reconstruction: BCE summed over pixels, averaged over batch.
    KL: -0.5 * sum(1 + log_var - mu^2 - exp(log_var)), averaged over batch.
    """
    bce = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    batch_size = x.size(0)
    return (bce + kl) / batch_size, bce / batch_size, kl / batch_size


def make_dataloaders(batch_size=128, val_ratio=0.15, num_workers=2):
    """Create MNIST data loaders for VAE training."""
    transform = transforms.Compose([
        transforms.ToTensor()  # No normalization -- keep [0,1] for BCE
    ])

    full_train = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def build_model(input_dim=784, hidden_dim=400, latent_dim=20):
    """Build the VAE model."""
    device = get_device()
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim,
                latent_dim=latent_dim).to(device)
    return model


def train(model, train_loader, val_loader, epochs=15, lr=1e-3):
    """Train the VAE."""
    device = get_device()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_kls = []
    train_recons = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0

        for data, _ in train_loader:
            data = data.to(device).view(data.size(0), -1)

            optimizer.zero_grad()
            x_recon, mu, log_var = model(data)
            loss, recon, kl = vae_loss(x_recon, data, mu, log_var)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches

        train_losses.append(avg_loss)
        train_recons.append(avg_recon)
        train_kls.append(avg_kl)

        # Validation
        val_metrics = evaluate(model, val_loader, return_samples=False)
        val_losses.append(val_metrics['loss'])

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Loss: {avg_loss:.2f}, "
              f"Recon: {avg_recon:.2f}, "
              f"KL: {avg_kl:.2f}, "
              f"Val Loss: {val_metrics['loss']:.2f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_recons': train_recons,
        'train_kls': train_kls
    }


def evaluate(model, data_loader, return_samples=True):
    """Evaluate VAE on a data split. Returns loss metrics and optional samples."""
    device = get_device()
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    sample_originals = None
    sample_recons = None

    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            data = data.to(device).view(data.size(0), -1)
            x_recon, mu, log_var = model(data)
            loss, recon, kl = vae_loss(x_recon, data, mu, log_var)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

            # Capture first batch for visualization
            if return_samples and i == 0:
                sample_originals = data[:8].cpu()
                sample_recons = x_recon[:8].cpu()

    n_batches = len(data_loader)
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'kl_loss': total_kl / n_batches
    }

    if return_samples:
        metrics['sample_originals'] = sample_originals
        metrics['sample_recons'] = sample_recons

    return metrics


def predict(model, data_loader):
    """Encode data and return latent representations."""
    device = get_device()
    model.eval()
    all_mu = []
    all_z = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device).view(data.size(0), -1)
            mu, log_var = model.encode(data)
            z = model.reparameterize(mu, log_var)
            all_mu.extend(mu.cpu().numpy())
            all_z.extend(z.cpu().numpy())

    return np.array(all_mu), np.array(all_z)


def _generate_samples(model, n_samples=64):
    """Generate new samples by decoding random latent vectors."""
    device = get_device()
    model.eval()

    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim, device=device)
        samples = model.decode(z).cpu()

    return samples


def save_artifacts(model, metrics, output_dir='./output'):
    """Save model, metrics, reconstruction grid, and generated samples."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    # Save serializable metrics
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (list, float, int, str, bool)):
            serializable[k] = v
        elif isinstance(v, dict):
            serializable[k] = {
                sk: sv if isinstance(sv, (float, int, str, bool, list)) else str(sv)
                for sk, sv in v.items()
            }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    # Save reconstruction comparison
    if 'sample_originals' in metrics and metrics['sample_originals'] is not None:
        originals = metrics['sample_originals']
        recons = metrics['sample_recons']
        n = originals.size(0)

        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        for i in range(n):
            axes[0, i].imshow(originals[i].view(28, 28).numpy(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            axes[1, i].imshow(recons[i].view(28, 28).numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Recon', fontsize=10)

        plt.suptitle('VAE Reconstructions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vae_reconstructions.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Save generated samples grid
    samples = _generate_samples(model, n_samples=64)
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(64):
        ax = axes[i // 8, i % 8]
        ax.imshow(samples[i].view(28, 28).numpy(), cmap='gray')
        ax.axis('off')
    plt.suptitle('VAE Generated Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_generated_samples.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Loss curves
    if 'history' in metrics:
        h = metrics['history']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(h['train_losses'], label='Train ELBO')
        ax1.plot(h['val_losses'], label='Val ELBO')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss (ELBO)')
        ax1.legend()

        ax2.plot(h['train_recons'], label='Recon Loss')
        ax2.plot(h['train_kls'], label='KL Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Components')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vae_loss_curves.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("Variational Autoencoder (VAE) on MNIST")
    print("=" * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Algorithm: {metadata['algorithm']}")
    print(f"Latent dim: {metadata['latent_dim']}")

    # Data
    print("\nLoading MNIST...")
    train_loader, val_loader = make_dataloaders(batch_size=128)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model
    model = build_model(input_dim=784, hidden_dim=400, latent_dim=20)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"Architecture:\n{model}")

    # Train
    print("\n" + "-" * 60)
    history = train(model, train_loader, val_loader, epochs=15, lr=1e-3)

    # Evaluate on both splits
    print("\n" + "-" * 60)
    print("Evaluating on training set...")
    train_metrics = evaluate(model, train_loader)
    print(f"Train Loss: {train_metrics['loss']:.2f}, "
          f"Recon: {train_metrics['recon_loss']:.2f}, "
          f"KL: {train_metrics['kl_loss']:.2f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader)
    print(f"Val Loss: {val_metrics['loss']:.2f}, "
          f"Recon: {val_metrics['recon_loss']:.2f}, "
          f"KL: {val_metrics['kl_loss']:.2f}")

    # Generate samples
    print("\nGenerating samples...")
    samples = _generate_samples(model, n_samples=16)
    has_nan = torch.isnan(samples).any().item()
    print(f"Generated 16 samples, NaN check: {'FAIL - NaNs found' if has_nan else 'OK'}")

    # Check for posterior collapse (KL too small)
    kl_value = val_metrics['kl_loss']
    posterior_collapse = kl_value < 0.1
    if posterior_collapse:
        print(f"WARNING: Possible posterior collapse (KL = {kl_value:.4f})")
    else:
        print(f"KL divergence healthy: {kl_value:.2f}")

    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {
        'train_loss': train_metrics['loss'],
        'train_recon': train_metrics['recon_loss'],
        'train_kl': train_metrics['kl_loss'],
        'val_loss': val_metrics['loss'],
        'val_recon': val_metrics['recon_loss'],
        'val_kl': val_metrics['kl_loss'],
        'sample_originals': val_metrics.get('sample_originals'),
        'sample_recons': val_metrics.get('sample_recons'),
        'history': {
            'train_losses': history['train_losses'],
            'val_losses': history['val_losses'],
            'train_recons': history['train_recons'],
            'train_kls': history['train_kls']
        }
    }
    save_artifacts(model, all_metrics)

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Total Loss:  {train_metrics['loss']:.2f}")
    print(f"Train Recon Loss:  {train_metrics['recon_loss']:.2f}")
    print(f"Train KL Loss:     {train_metrics['kl_loss']:.2f}")
    print(f"Val Total Loss:    {val_metrics['loss']:.2f}")
    print(f"Val Recon Loss:    {val_metrics['recon_loss']:.2f}")
    print(f"Val KL Loss:       {val_metrics['kl_loss']:.2f}")

    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    # Check 1: Loss decreased during training
    loss_decreased = history['train_losses'][-1] < history['train_losses'][0]
    s1 = "PASS" if loss_decreased else "FAIL"
    print(f"[{s1}] Loss decreased: "
          f"{history['train_losses'][0]:.2f} -> {history['train_losses'][-1]:.2f}")
    checks_passed = checks_passed and loss_decreased

    # Check 2: No NaN in generated samples
    no_nan = not has_nan
    s2 = "PASS" if no_nan else "FAIL"
    print(f"[{s2}] No NaN in generated samples")
    checks_passed = checks_passed and no_nan

    # Check 3: No posterior collapse (KL > 0.1)
    no_collapse = not posterior_collapse
    s3 = "PASS" if no_collapse else "FAIL"
    print(f"[{s3}] No posterior collapse (KL = {kl_value:.2f} > 0.1)")
    checks_passed = checks_passed and no_collapse

    # Check 4: Val loss within reasonable range of train loss
    loss_ratio = val_metrics['loss'] / train_metrics['loss'] if train_metrics['loss'] > 0 else 999
    reasonable_gap = loss_ratio < 1.3
    s4 = "PASS" if reasonable_gap else "FAIL"
    print(f"[{s4}] Val/Train loss ratio < 1.3: {loss_ratio:.4f}")
    checks_passed = checks_passed and reasonable_gap

    # Check 5: Reconstruction loss is reasonable (below 100 per sample for MNIST)
    recon_ok = val_metrics['recon_loss'] < 100
    s5 = "PASS" if recon_ok else "FAIL"
    print(f"[{s5}] Val recon loss < 100: {val_metrics['recon_loss']:.2f}")
    checks_passed = checks_passed and recon_ok

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    sys.exit(0 if checks_passed else 1)
