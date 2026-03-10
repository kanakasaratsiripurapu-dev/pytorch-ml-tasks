"""
Univariate Linear Regression — Raw Tensor Implementation

Model:
    h_theta(x) = theta_0 + theta_1 * x

Cost (Mean Squared Error):
    J(theta) = (1/m) * sum_i (h_theta(x_i) - y_i)^2

Partial derivatives (computed by hand, no autograd):
    dJ/d(theta_0) = (2/m) * sum(h - y)
    dJ/d(theta_1) = (2/m) * sum((h - y) * x)

This file intentionally avoids torch.nn, torch.optim, and autograd
to demonstrate gradient descent at the lowest level of abstraction.
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
TRUE_SLOPE = 2.0
TRUE_INTERCEPT = 3.0


# ── protocol functions ────────────────────────────────────────────

def get_task_metadata():
    return {
        'task_name': 'linreg_lvl1_raw_tensors',
        'task_type': 'regression',
        'algorithm': 'Linear Regression (Raw Tensors)',
        'description': 'Univariate LR with hand-derived gradients; no nn / optim / autograd',
    }


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=500, noise_std=0.5, val_ratio=0.2, batch_size=None):
    """Synthetic data  y = 2x + 3 + N(0, noise_std).  80-20 split."""
    set_seed()
    dev = get_device()

    x = torch.randn(n_samples, 1, device=dev) * 3.0
    y = TRUE_SLOPE * x + TRUE_INTERCEPT + torch.randn_like(x) * noise_std

    perm = torch.randperm(n_samples)
    cut = int(n_samples * val_ratio)
    return (x[perm[cut:]], y[perm[cut:]]), (x[perm[:cut]], y[perm[:cut]])


def build_model():
    """Two scalar parameters stored in a plain dict — no nn.Module."""
    dev = get_device()
    return {'theta_0': torch.zeros(1, device=dev),
            'theta_1': torch.zeros(1, device=dev)}


def train(model, train_data, val_data, epochs=300, lr=0.01):
    """Vanilla batch gradient descent with manually derived gradients."""
    x_tr, y_tr = train_data
    m = float(x_tr.shape[0])
    t0, t1 = model['theta_0'].clone(), model['theta_1'].clone()

    hist_tr, hist_val = [], []
    for ep in range(epochs):
        pred = t0 + t1 * x_tr
        residual = pred - y_tr
        mse = (residual ** 2).mean().item()
        hist_tr.append(mse)

        # gradient update (hand-computed)
        t0 = t0 - lr * (2.0 / m) * residual.sum()
        t1 = t1 - lr * (2.0 / m) * (residual * x_tr).sum()

        # track validation
        xv, yv = val_data
        val_mse = ((t0 + t1 * xv - yv) ** 2).mean().item()
        hist_val.append(val_mse)

        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>4d}/{epochs}  train_mse={mse:.4f}  "
                  f"val_mse={val_mse:.4f}  t0={t0.item():.4f}  t1={t1.item():.4f}")

    model['theta_0'], model['theta_1'] = t0, t1
    return {'train_losses': hist_tr, 'val_losses': hist_val,
            'final_theta_0': t0.item(), 'final_theta_1': t1.item()}


def evaluate(model, data):
    """MSE, R-squared, and parameter-recovery error."""
    x, y = data
    h = model['theta_0'] + model['theta_1'] * x
    ss_res = ((y - h) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    mse = ((h - y) ** 2).mean().item()
    r2 = 1.0 - ss_res / ss_tot
    t0_err = abs(model['theta_0'].item() - TRUE_INTERCEPT)
    t1_err = abs(model['theta_1'].item() - TRUE_SLOPE)
    return dict(mse=mse, r2=r2,
                theta_0=model['theta_0'].item(), theta_1=model['theta_1'].item(),
                theta_0_error=t0_err, theta_1_error=t1_err)


def predict(model, x):
    return model['theta_0'] + model['theta_1'] * x


def save_artifacts(model, metrics, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.items()},
               os.path.join(output_dir, 'model.pth'))

    # dump JSON-safe subset
    safe = {k: (v if isinstance(v, (float, int, str, bool, list, dict)) else str(v))
            for k, v in metrics.items()
            if k not in ('history',)}
    if 'history' in metrics:
        safe['history'] = metrics['history']
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(safe, f, indent=2)

    # loss curve
    if 'history' in metrics:
        plt.figure(figsize=(9, 4))
        plt.plot(metrics['history']['train_losses'], label='Train MSE', linewidth=0.8)
        plt.plot(metrics['history']['val_losses'], label='Val MSE', linewidth=0.8)
        plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
        plt.title('Loss Curve — Linear Regression (raw tensors)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'linreg_lvl1_loss.png'), dpi=120)
        plt.close()
    print(f"[save_artifacts] wrote to {output_dir}/")


# ── main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print(" Linear Regression  |  raw tensors, manual GD")
    print("=" * 55)
    set_seed()
    meta = get_task_metadata()
    print(f"task  : {meta['task_name']}")

    train_data, val_data = make_dataloaders()
    print(f"data  : {train_data[0].shape[0]} train / {val_data[0].shape[0]} val")
    print(f"target: y = {TRUE_SLOPE}x + {TRUE_INTERCEPT} + noise\n")

    model = build_model()
    history = train(model, train_data, val_data, epochs=300, lr=0.01)

    tr = evaluate(model, train_data)
    va = evaluate(model, val_data)

    save_artifacts(model, {**va, 'history': {
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses']}})

    print(f"\n{'='*55}")
    print(f" RESULTS")
    print(f"{'='*55}")
    print(f"  train  MSE={tr['mse']:.4f}   R2={tr['r2']:.4f}")
    print(f"  val    MSE={va['mse']:.4f}   R2={va['r2']:.4f}")
    print(f"  theta_0 = {va['theta_0']:.4f}  (true {TRUE_INTERCEPT})")
    print(f"  theta_1 = {va['theta_1']:.4f}  (true {TRUE_SLOPE})")

    # ── assertions ────────────────────────────────────────────────
    ok = True
    for tag, cond in [
        (f"R2 > 0.9          ({va['r2']:.4f})",              va['r2'] > 0.9),
        (f"theta_0 err < 1.0 ({va['theta_0_error']:.4f})",   va['theta_0_error'] < 1.0),
        (f"theta_1 err < 1.0 ({va['theta_1_error']:.4f})",   va['theta_1_error'] < 1.0),
        (f"MSE < 1.0         ({va['mse']:.4f})",             va['mse'] < 1.0),
    ]:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {tag}")
        ok = ok and cond

    print(f"\n{'PASS' if ok else 'FAIL'}: all quality checks {'passed' if ok else 'did not pass'}")
    sys.exit(0 if ok else 1)
