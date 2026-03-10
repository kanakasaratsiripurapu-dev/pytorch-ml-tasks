"""
Univariate Linear Regression using ONLY PyTorch tensors.

Mathematical formulation:
    h_theta(x) = theta_0 + theta_1 * x
    Cost: J(theta) = (1/m) * sum_i (h_theta(x_i) - y_i)^2   (MSE)

Gradient (manual):
    dJ/d(theta_0) = (2/m) * sum(h - y)
    dJ/d(theta_1) = (2/m) * sum((h - y) * x)

Constraints: No torch.nn, no torch.optim, no autograd.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'linreg_lvl1_raw_tensors',
        'task_type': 'regression',
        'algorithm': 'Linear Regression (Raw Tensors)',
        'description': 'Univariate linear regression with manual gradient descent, no autograd'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the device for computation."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=500, noise_std=0.5, val_ratio=0.2, batch_size=None):
    """
    Generate synthetic data: y = 2x + 3 + noise.
    Split into 80% train / 20% validation.
    Returns raw tensors (not DataLoaders) since we avoid torch.nn.
    """
    set_seed(42)
    device = get_device()

    x = torch.randn(n_samples, 1, device=device) * 3.0
    noise = torch.randn(n_samples, 1, device=device) * noise_std
    y = 2.0 * x + 3.0 + noise

    # Shuffle and split
    indices = torch.randperm(n_samples)
    val_size = int(n_samples * val_ratio)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    return (x_train, y_train), (x_val, y_val)


def build_model():
    """Initialize parameters theta_0 (bias) and theta_1 (weight) as raw tensors."""
    device = get_device()
    theta_0 = torch.zeros(1, device=device)
    theta_1 = torch.zeros(1, device=device)
    return {'theta_0': theta_0, 'theta_1': theta_1}


def train(model, train_data, val_data, epochs=300, lr=0.01):
    """Train using manual gradient descent (no autograd)."""
    x_train, y_train = train_data
    m = x_train.shape[0]

    theta_0 = model['theta_0'].clone()
    theta_1 = model['theta_1'].clone()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Forward pass: h = theta_0 + theta_1 * x
        h = theta_0 + theta_1 * x_train

        # MSE loss
        error = h - y_train
        loss = (error ** 2).mean().item()
        train_losses.append(loss)

        # Manual gradients
        grad_theta_0 = (2.0 / m) * error.sum()
        grad_theta_1 = (2.0 / m) * (error * x_train).sum()

        # Parameter update
        theta_0 = theta_0 - lr * grad_theta_0
        theta_1 = theta_1 - lr * grad_theta_1

        # Validation loss
        x_val, y_val = val_data
        h_val = theta_0 + theta_1 * x_val
        val_loss = ((h_val - y_val) ** 2).mean().item()
        val_losses.append(val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train MSE: {loss:.4f}, Val MSE: {val_loss:.4f}, "
                  f"theta_0: {theta_0.item():.4f}, theta_1: {theta_1.item():.4f}")

    model['theta_0'] = theta_0
    model['theta_1'] = theta_1

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_theta_0': theta_0.item(),
        'final_theta_1': theta_1.item()
    }


def evaluate(model, data):
    """Compute MSE, R2 score, and parameter accuracy on given data split."""
    x, y = data
    theta_0 = model['theta_0']
    theta_1 = model['theta_1']

    h = theta_0 + theta_1 * x

    mse = ((h - y) ** 2).mean().item()
    ss_res = ((y - h) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot

    theta_0_error = abs(theta_0.item() - 3.0)
    theta_1_error = abs(theta_1.item() - 2.0)

    return {
        'mse': mse,
        'r2': r2,
        'theta_0': theta_0.item(),
        'theta_1': theta_1.item(),
        'theta_0_error': theta_0_error,
        'theta_1_error': theta_1_error
    }


def predict(model, x):
    """Predict y values given input x tensor."""
    return model['theta_0'] + model['theta_1'] * x


def save_artifacts(model, metrics, output_dir='./output'):
    """Save model parameters, metrics, and loss plot."""
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters
    torch.save({
        'theta_0': model['theta_0'].cpu(),
        'theta_1': model['theta_1'].cpu()
    }, os.path.join(output_dir, 'model.pth'))

    # Save metrics
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            serializable[k] = {
                sk: sv if isinstance(sv, (float, int, str, bool, list)) else str(sv)
                for sk, sv in v.items()
            }
        elif isinstance(v, list):
            serializable[k] = v
        else:
            serializable[k] = v

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    # Loss curve plot
    if 'history' in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['history']['train_losses'], label='Train MSE')
        plt.plot(metrics['history']['val_losses'], label='Val MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'linreg_lvl1_loss.png'), dpi=150)
        plt.close()

    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("Univariate Linear Regression - Raw Tensors (No Autograd)")
    print("=" * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Algorithm: {metadata['algorithm']}")

    # Create data
    print("\nCreating synthetic data: y = 2x + 3 + noise ...")
    train_data, val_data = make_dataloaders(n_samples=500, noise_std=0.5)
    print(f"Train samples: {train_data[0].shape[0]}")
    print(f"Val samples: {val_data[0].shape[0]}")

    # Build model
    model = build_model()
    print(f"\nInitial params: theta_0={model['theta_0'].item():.4f}, "
          f"theta_1={model['theta_1'].item():.4f}")

    # Train
    print("\n" + "-" * 60)
    history = train(model, train_data, val_data, epochs=300, lr=0.01)

    # Evaluate on both splits
    print("\n" + "-" * 60)
    print("Evaluating on training set...")
    train_metrics = evaluate(model, train_data)
    print(f"Train MSE: {train_metrics['mse']:.4f}, R2: {train_metrics['r2']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_data)
    print(f"Val MSE: {val_metrics['mse']:.4f}, R2: {val_metrics['r2']:.4f}")

    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'history': {
            'train_losses': history['train_losses'],
            'val_losses': history['val_losses']
        }
    }
    save_artifacts(model, all_metrics)

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train MSE:  {train_metrics['mse']:.4f}")
    print(f"Val MSE:    {val_metrics['mse']:.4f}")
    print(f"Train R2:   {train_metrics['r2']:.4f}")
    print(f"Val R2:     {val_metrics['r2']:.4f}")
    print(f"Learned:    theta_0 = {val_metrics['theta_0']:.4f}  (true = 3.0)")
    print(f"            theta_1 = {val_metrics['theta_1']:.4f}  (true = 2.0)")
    print(f"Errors:     theta_0_err = {val_metrics['theta_0_error']:.4f}")
    print(f"            theta_1_err = {val_metrics['theta_1_error']:.4f}")

    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    c1 = val_metrics['r2'] > 0.9
    s1 = "PASS" if c1 else "FAIL"
    print(f"[{s1}] Val R2 > 0.9: {val_metrics['r2']:.4f}")
    checks_passed = checks_passed and c1

    c2 = val_metrics['theta_0_error'] < 1.0
    s2 = "PASS" if c2 else "FAIL"
    print(f"[{s2}] theta_0 error < 1.0: {val_metrics['theta_0_error']:.4f}")
    checks_passed = checks_passed and c2

    c3 = val_metrics['theta_1_error'] < 1.0
    s3 = "PASS" if c3 else "FAIL"
    print(f"[{s3}] theta_1 error < 1.0: {val_metrics['theta_1_error']:.4f}")
    checks_passed = checks_passed and c3

    c4 = val_metrics['mse'] < 1.0
    s4 = "PASS" if c4 else "FAIL"
    print(f"[{s4}] Val MSE < 1.0: {val_metrics['mse']:.4f}")
    checks_passed = checks_passed and c4

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    sys.exit(0 if checks_passed else 1)
