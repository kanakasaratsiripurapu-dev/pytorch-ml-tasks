r"""
Variational Autoencoder (VAE) — reparameterisation trick on MNIST

Evidence Lower Bound (ELBO):
    $\mathcal{L}(\theta,\phi;\mathbf{x})
     = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}
       [\log p_\theta(\mathbf{x}|\mathbf{z})]
     - D_{KL}\bigl(q_\phi(\mathbf{z}|\mathbf{x})
                    \| p(\mathbf{z})\bigr)$

Reparameterisation:
    $\mathbf{z} = \boldsymbol{\mu}
     + \boldsymbol{\sigma}\odot\boldsymbol{\epsilon},
     \quad \boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$

KL divergence (diagonal Gaussian vs standard normal):
    $D_{KL} = -\tfrac{1}{2}\sum_{j=1}^{J}
     \bigl(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\bigr)$

Reconstruction uses pixel-wise BCE (images kept in [0,1]).
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
LATENT = 20
HIDDEN = 400
IMG_DIM = 784  # 28*28


# ── model ─────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, img_dim=IMG_DIM, h=HIDDEN, z=LATENT):
        super().__init__()
        self.z_dim = z
        self.enc = nn.Sequential(nn.Linear(img_dim, h), nn.ReLU(),
                                 nn.Linear(h, h),       nn.ReLU())
        self.to_mu     = nn.Linear(h, z)
        self.to_logvar = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h),       nn.ReLU(),
                                 nn.Linear(h, h),       nn.ReLU(),
                                 nn.Linear(h, img_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterise(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterise(mu, lv)
        return self.decode(z), mu, lv


def _elbo_loss(xr, x, mu, lv):
    """Returns (total, recon, kl), each averaged over batch."""
    bce = nn.functional.binary_cross_entropy(xr, x, reduction='sum')
    kl  = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
    bs  = x.size(0)
    return (bce + kl) / bs, bce / bs, kl / bs


# ── protocol functions ────────────────────────────────────────────

def get_task_metadata():
    return {
        'task_name':   'ae_lvl3_vae',
        'task_type':   'generative',
        'algorithm':   'Variational Autoencoder (VAE)',
        'input_shape': [1, 28, 28],
        'latent_dim':  LATENT,
        'description': 'VAE with reparam trick, KL+BCE loss, trained on MNIST',
    }


def set_seed(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(batch_size=128, val_ratio=0.15, num_workers=2):
    """MNIST loaders; images stay in [0,1] (no normalisation) for BCE."""
    tfm = transforms.ToTensor()
    full = datasets.MNIST('./data', train=True, download=True, transform=tfm)
    nv = int(len(full) * val_ratio)
    tr, va = random_split(full, [len(full) - nv, nv],
                          generator=torch.Generator().manual_seed(SEED))
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return DataLoader(tr, shuffle=True, **kw), DataLoader(va, shuffle=False, **kw)


def build_model(input_dim=IMG_DIM, hidden_dim=HIDDEN, latent_dim=LATENT):
    return VAE(input_dim, hidden_dim, latent_dim).to(get_device())


def train(model, tr_dl, va_dl, epochs=15, lr=1e-3):
    dev = get_device()
    opt = optim.Adam(model.parameters(), lr=lr)
    h_loss, h_val, h_rec, h_kl = [], [], [], []

    for ep in range(epochs):
        model.train()
        s_l = s_r = s_k = 0.0
        for imgs, _ in tr_dl:
            x = imgs.to(dev).view(imgs.size(0), -1)
            opt.zero_grad()
            xr, mu, lv = model(x)
            loss, rec, kl = _elbo_loss(xr, x, mu, lv)
            loss.backward(); opt.step()
            s_l += loss.item(); s_r += rec.item(); s_k += kl.item()

        nb = len(tr_dl)
        h_loss.append(s_l / nb); h_rec.append(s_r / nb); h_kl.append(s_k / nb)
        vm = evaluate(model, va_dl, grab_samples=False)
        h_val.append(vm['loss'])

        print(f"  ep {ep+1:>2d}/{epochs}  "
              f"loss={s_l/nb:.1f}  rec={s_r/nb:.1f}  kl={s_k/nb:.1f}  "
              f"val={vm['loss']:.1f}")

    return dict(train_losses=h_loss, val_losses=h_val,
                train_recons=h_rec, train_kls=h_kl)


def evaluate(model, dl, grab_samples=True):
    dev = get_device(); model.eval()
    s_l = s_r = s_k = 0.0
    orig = recon = None
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dl):
            x = imgs.to(dev).view(imgs.size(0), -1)
            xr, mu, lv = model(x)
            l, r, k = _elbo_loss(xr, x, mu, lv)
            s_l += l.item(); s_r += r.item(); s_k += k.item()
            if grab_samples and i == 0:
                orig, recon = x[:8].cpu(), xr[:8].cpu()
    nb = len(dl)
    out = dict(loss=s_l/nb, recon_loss=s_r/nb, kl_loss=s_k/nb)
    if grab_samples:
        out['sample_originals'] = orig; out['sample_recons'] = recon
    return out


def predict(model, dl):
    dev = get_device(); model.eval()
    mus, zs = [], []
    with torch.no_grad():
        for imgs, _ in dl:
            x = imgs.to(dev).view(imgs.size(0), -1)
            mu, lv = model.encode(x)
            mus.append(mu.cpu()); zs.append(model.reparameterise(mu, lv).cpu())
    return torch.cat(mus).numpy(), torch.cat(zs).numpy()


def _sample(model, n=64):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.z_dim, device=next(model.parameters()).device)
        return model.decode(z).cpu()


def save_artifacts(model, metrics, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    safe = {k: v for k, v in metrics.items()
            if isinstance(v, (float, int, str, bool, list, dict))}
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(safe, f, indent=2)

    # reconstruction grid
    if metrics.get('sample_originals') is not None:
        o, r = metrics['sample_originals'], metrics['sample_recons']
        n = o.size(0)
        fig, ax = plt.subplots(2, n, figsize=(n * 1.8, 3.6))
        for i in range(n):
            ax[0, i].imshow(o[i].view(28, 28), cmap='gray'); ax[0, i].axis('off')
            ax[1, i].imshow(r[i].view(28, 28), cmap='gray'); ax[1, i].axis('off')
        ax[0, 0].set_title('input', fontsize=9)
        ax[1, 0].set_title('recon', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vae_reconstructions.png'), dpi=120)
        plt.close()

    # generated grid
    samp = _sample(model, 64)
    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(64):
        ax[i // 8, i % 8].imshow(samp[i].view(28, 28), cmap='gray')
        ax[i // 8, i % 8].axis('off')
    plt.suptitle('Generated digits'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vae_generated_samples.png'), dpi=120)
    plt.close()

    # loss curves
    if 'history' in metrics:
        h = metrics['history']
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(h['train_losses'], label='train'); a1.plot(h['val_losses'], label='val')
        a1.set(xlabel='epoch', ylabel='ELBO', title='Total loss'); a1.legend()
        a2.plot(h['train_recons'], label='recon'); a2.plot(h['train_kls'], label='KL')
        a2.set(xlabel='epoch', ylabel='loss', title='Components'); a2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vae_loss_curves.png'), dpi=120)
        plt.close()

    print(f"[save_artifacts] wrote to {output_dir}/")


# ── main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print(" VAE  |  MNIST, reparam trick, ELBO objective")
    print("=" * 55)
    set_seed()
    meta = get_task_metadata()
    print(f"task: {meta['task_name']}   z_dim={LATENT}\n")

    tr_dl, va_dl = make_dataloaders(batch_size=128)
    print(f"data : {len(tr_dl.dataset)} train / {len(va_dl.dataset)} val")

    model = build_model()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par:,}\n")

    hist = train(model, tr_dl, va_dl, epochs=15, lr=1e-3)

    tr_m = evaluate(model, tr_dl)
    va_m = evaluate(model, va_dl)
    print(f"\n  train  loss={tr_m['loss']:.1f}  rec={tr_m['recon_loss']:.1f}  kl={tr_m['kl_loss']:.1f}")
    print(f"  val    loss={va_m['loss']:.1f}  rec={va_m['recon_loss']:.1f}  kl={va_m['kl_loss']:.1f}")

    samp = _sample(model, 16)
    has_nan = torch.isnan(samp).any().item()
    kl_val = va_m['kl_loss']
    collapse = kl_val < 0.1
    if collapse:
        print(f"  WARNING: possible posterior collapse (KL={kl_val:.3f})")
    else:
        print(f"  KL healthy: {kl_val:.1f}")

    save_artifacts(model, {
        **{f'{split}_{k}': va_m[k] if split == 'val' else tr_m[k]
           for split in ('train', 'val') for k in ('loss', 'recon_loss', 'kl_loss')},
        'sample_originals': va_m.get('sample_originals'),
        'sample_recons':    va_m.get('sample_recons'),
        'history': dict(train_losses=hist['train_losses'],
                        val_losses=hist['val_losses'],
                        train_recons=hist['train_recons'],
                        train_kls=hist['train_kls']),
    })

    print(f"\n{'='*55}")
    print(" QUALITY CHECKS")
    print(f"{'='*55}")
    ok = True
    for tag, cond in [
        (f"loss decreased  ({hist['train_losses'][0]:.0f} -> {hist['train_losses'][-1]:.0f})",
         hist['train_losses'][-1] < hist['train_losses'][0]),
        ("no NaN in samples",        not has_nan),
        (f"no posterior collapse (KL={kl_val:.1f})",  not collapse),
        (f"val/train ratio < 1.3  ({va_m['loss']/tr_m['loss']:.3f})",
         va_m['loss'] / tr_m['loss'] < 1.3 if tr_m['loss'] > 0 else False),
        (f"val recon < 100        ({va_m['recon_loss']:.1f})",
         va_m['recon_loss'] < 100),
    ]:
        print(f"  [{'PASS' if cond else 'FAIL'}] {tag}")
        ok = ok and cond

    print(f"\n{'PASS' if ok else 'FAIL'}: all quality checks {'passed' if ok else 'did not pass'}")
    sys.exit(0 if ok else 1)
