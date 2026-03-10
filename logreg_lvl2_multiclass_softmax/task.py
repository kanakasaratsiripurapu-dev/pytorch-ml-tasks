r"""
Multiclass Softmax Regression using torch.nn.Module + CrossEntropyLoss.

Mathematical formulation:
    Softmax:
        $P(y=k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$

    Cross-Entropy Loss:
        $L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log P(y=k \mid \mathbf{x}_i)$

Uses Adam optimizer on 3-class synthetic blob data.
Produces decision boundary contour plot.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'logreg_lvl2_multiclass_softmax',
        'task_type': 'classification',
        'num_classes': 3,
        'input_features': 2,
        'algorithm': 'Softmax Regression (Multiclass)',
        'description': 'Multiclass softmax regression using nn.Module, Adam, CrossEntropyLoss'
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


class SoftmaxClassifier(nn.Module):
    """Single linear layer mapping inputs to class logits."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def make_dataloaders(n_samples=600, num_classes=3, val_ratio=0.2, batch_size=32):
    """
    Create 3-class blobs dataset with standardized features.
    Returns train_loader, val_loader, and raw (X, y, scaler) for plotting.
    """
    set_seed(42)

    X, y = make_blobs(
        n_samples=n_samples, centers=num_classes,
        n_features=2, cluster_std=1.5, random_state=42
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(n_samples * val_ratio)
    train_size = n_samples - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, (X_scaled, y)


def build_model(input_dim=2, num_classes=3):
    """Build the softmax classifier model."""
    device = get_device()
    model = SoftmaxClassifier(input_dim, num_classes).to(device)
    return model


def train(model, train_loader, val_loader, epochs=100, lr=0.01):
    """Train the model with Adam and CrossEntropyLoss."""
    device = get_device()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_f1_scores = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        val_metrics = evaluate(model, val_loader, return_predictions=False)
        val_losses.append(val_metrics['loss'])
        val_f1_scores.append(val_metrics['macro_f1'])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val F1: {val_metrics['macro_f1']:.4f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1_scores': val_f1_scores
    }


def evaluate(model, data_loader, return_predictions=True):
    """Evaluate model, computing loss, accuracy, and macro-F1."""
    device = get_device()
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }

    if return_predictions:
        metrics['predictions'] = np.array(all_preds)
        metrics['targets'] = np.array(all_targets)

    return metrics


def predict(model, data_loader):
    """Get predictions and probability estimates."""
    device = get_device()
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)


def _save_decision_boundary(model, X, y, output_dir):
    """Generate and save the decision boundary contour plot."""
    device = get_device()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(grid)
        _, Z = torch.max(logits, 1)
        Z = Z.cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                          edgecolors='black', s=30)
    plt.colorbar(scatter)
    plt.title('Softmax Regression - Decision Boundary (3 classes)')
    plt.xlabel('Feature 1 (standardized)')
    plt.ylabel('Feature 2 (standardized)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'logreg_lvl2_boundary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def save_artifacts(model, metrics, output_dir='./output'):
    """Save model, metrics JSON, and decision boundary plot."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    # Save metrics (filter non-serializable values)
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (list, dict, float, int, str, bool)):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("Multiclass Softmax Regression (3-class Blobs)")
    print("=" * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Algorithm: {metadata['algorithm']}")

    # Data
    print("\nCreating 3-class blob dataset...")
    train_loader, val_loader, (X_all, y_all) = make_dataloaders(
        n_samples=600, num_classes=3, batch_size=32
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model
    model = build_model(input_dim=2, num_classes=3)
    print(f"\nModel:\n{model}")

    # Train
    print("\n" + "-" * 60)
    history = train(model, train_loader, val_loader, epochs=100, lr=0.01)

    # Evaluate on both splits
    print("\n" + "-" * 60)
    print("Evaluating on training set...")
    train_metrics = evaluate(model, train_loader)
    print(f"Train Acc: {train_metrics['accuracy']:.4f}, "
          f"Train F1: {train_metrics['macro_f1']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader)
    print(f"Val Acc: {val_metrics['accuracy']:.4f}, "
          f"Val F1: {val_metrics['macro_f1']:.4f}")

    # Decision boundary plot
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    _save_decision_boundary(model, X_all, y_all, output_dir)
    print("\nDecision boundary saved to output/logreg_lvl2_boundary.png")

    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {
        'train_accuracy': train_metrics['accuracy'],
        'train_f1': train_metrics['macro_f1'],
        'val_accuracy': val_metrics['accuracy'],
        'val_f1': val_metrics['macro_f1'],
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'val_f1_scores': history['val_f1_scores']
    }
    save_artifacts(model, all_metrics, output_dir=output_dir)

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"Train Macro-F1:  {train_metrics['macro_f1']:.4f}")
    print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"Val Macro-F1:    {val_metrics['macro_f1']:.4f}")

    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    c1 = val_metrics['macro_f1'] > 0.85
    s1 = "PASS" if c1 else "FAIL"
    print(f"[{s1}] Macro-F1 > 0.85: {val_metrics['macro_f1']:.4f}")
    checks_passed = checks_passed and c1

    c2 = val_metrics['accuracy'] > 0.80
    s2 = "PASS" if c2 else "FAIL"
    print(f"[{s2}] Val Accuracy > 0.80: {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and c2

    c3 = train_metrics['accuracy'] > 0.80
    s3 = "PASS" if c3 else "FAIL"
    print(f"[{s3}] Train Accuracy > 0.80: {train_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and c3

    acc_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
    c4 = acc_gap < 0.10
    s4 = "PASS" if c4 else "FAIL"
    print(f"[{s4}] Accuracy gap < 0.10: {acc_gap:.4f}")
    checks_passed = checks_passed and c4

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    sys.exit(0 if checks_passed else 1)
