"""
K-Means Clustering from scratch using PyTorch tensors.

Mathematical formulation:
    Objective (within-cluster SSE):
        J = sum_{k=1}^{K} sum_{x_i in C_k} || x_i - mu_k ||^2

    Algorithm:
        1. Initialize centroids via k-means++ (greedy distance-based selection)
        2. Assign each point to nearest centroid (vectorized L2 distance)
        3. Recompute centroids as cluster means
        4. Repeat until convergence or max iterations

Validates against sklearn KMeans (inertia within 5%).
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'cluster_lvl1_kmeans',
        'task_type': 'clustering',
        'algorithm': 'K-Means (From Scratch)',
        'k': 4,
        'description': 'K-Means with k-means++ init and vectorized updates, compared to sklearn'
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


def _kmeans_plus_plus_init(X, k):
    """
    K-means++ initialization: pick first centroid randomly,
    then each subsequent centroid with probability proportional to D(x)^2.
    """
    n = X.shape[0]
    device = X.device

    # First centroid: random
    idx = torch.randint(0, n, (1,), device=device).item()
    centroids = [X[idx]]

    for _ in range(1, k):
        # Compute distance from each point to nearest existing centroid
        stacked = torch.stack(centroids, dim=0)  # [c, d]
        dists = torch.cdist(X.unsqueeze(0), stacked.unsqueeze(0)).squeeze(0)  # [n, c]
        min_dists, _ = dists.min(dim=1)  # [n]
        min_dists_sq = min_dists ** 2

        # Sample proportional to D^2
        probs = min_dists_sq / min_dists_sq.sum()
        idx = torch.multinomial(probs, 1).item()
        centroids.append(X[idx])

    return torch.stack(centroids, dim=0)  # [k, d]


def _compute_distances(X, centroids):
    """Compute pairwise L2 distances between X [n, d] and centroids [k, d]."""
    return torch.cdist(X, centroids)  # [n, k]


def _assign_clusters(X, centroids):
    """Assign each point to the nearest centroid. Returns labels [n]."""
    dists = _compute_distances(X, centroids)
    return torch.argmin(dists, dim=1)


def _compute_inertia(X, centroids, labels):
    """Compute within-cluster SSE (inertia)."""
    assigned_centroids = centroids[labels]  # [n, d]
    return ((X - assigned_centroids) ** 2).sum().item()


def make_dataloaders(n_samples=800, n_features=2, k=4, val_ratio=0.2):
    """
    Generate synthetic blob data for clustering.
    Split into 80% train / 20% validation.
    Returns train/val tensors, ground-truth labels, and raw numpy arrays.
    """
    set_seed(42)

    X, y_true = make_blobs(
        n_samples=n_samples, centers=k, n_features=n_features,
        cluster_std=1.0, random_state=42
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/val
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * val_ratio)
    train_idx = indices[val_size:]
    val_idx = indices[:val_size]

    device = get_device()
    X_train = torch.FloatTensor(X_scaled[train_idx]).to(device)
    X_val = torch.FloatTensor(X_scaled[val_idx]).to(device)
    y_train = y_true[train_idx]
    y_val = y_true[val_idx]

    return X_train, X_val, y_train, y_val, X_scaled, y_true


def build_model(X, k=4):
    """Initialize centroids using k-means++."""
    centroids = _kmeans_plus_plus_init(X, k)
    return {'centroids': centroids, 'k': k}


def train(model, X, max_iters=100, tol=1e-6):
    """
    Run K-Means iterations until convergence.
    Returns training history with inertia per iteration.
    """
    centroids = model['centroids'].clone()
    k = model['k']
    inertia_history = []

    for iteration in range(max_iters):
        # Step 1: Assign clusters
        labels = _assign_clusters(X, centroids)

        # Step 2: Compute new centroids
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = (labels == j)
            if mask.sum() > 0:
                new_centroids[j] = X[mask].mean(dim=0)
            else:
                # Reinitialize empty cluster to a random point
                new_centroids[j] = X[torch.randint(0, X.shape[0], (1,))].squeeze()

        # Compute inertia
        inertia = _compute_inertia(X, new_centroids, labels)
        inertia_history.append(inertia)

        # Check convergence
        shift = ((new_centroids - centroids) ** 2).sum().item()
        centroids = new_centroids

        if (iteration + 1) % 10 == 0 or shift < tol:
            print(f"Iter [{iteration+1}/{max_iters}], "
                  f"Inertia: {inertia:.4f}, Shift: {shift:.6f}")

        if shift < tol:
            print(f"Converged at iteration {iteration+1}")
            break

    model['centroids'] = centroids

    return {
        'inertia_history': inertia_history,
        'n_iterations': iteration + 1,
        'final_inertia': inertia_history[-1]
    }


def evaluate(model, X, y_true=None):
    """Evaluate clustering: inertia and optionally ARI against ground truth."""
    centroids = model['centroids']
    labels = _assign_clusters(X, centroids)
    inertia = _compute_inertia(X, centroids, labels)

    metrics = {
        'inertia': inertia,
        'labels': labels.cpu().numpy()
    }

    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels.cpu().numpy())
        metrics['adjusted_rand_index'] = ari

    return metrics


def predict(model, X):
    """Predict cluster assignments for new data."""
    centroids = model['centroids']
    labels = _assign_clusters(X, centroids)
    dists = _compute_distances(X, centroids)
    return labels.cpu().numpy(), dists.cpu().numpy()


def save_artifacts(model, metrics, output_dir='./output'):
    """Save centroids, metrics, and cluster visualization."""
    os.makedirs(output_dir, exist_ok=True)

    # Save centroids
    torch.save(model['centroids'].cpu(),
               os.path.join(output_dir, 'centroids.pth'))

    # Save serializable metrics
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (list, float, int, str, bool)):
            serializable[k] = v
        elif isinstance(v, dict):
            serializable[k] = {
                sk: sv.tolist() if isinstance(sv, np.ndarray) else sv
                for sk, sv in v.items()
            }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(serializable, f, indent=2)

    # Inertia convergence plot
    if 'inertia_history' in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(metrics['inertia_history'], marker='o', markersize=3)
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (SSE)')
        plt.title('K-Means Inertia Convergence')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kmeans_inertia.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # Cluster scatter plot (2D only)
    if 'X_2d' in metrics:
        X_2d = metrics['X_2d']
        centroids = model['centroids'].cpu().numpy()
        # Assign all points to nearest centroid for plotting
        X_full = torch.FloatTensor(X_2d).to(model['centroids'].device)
        labels = _assign_clusters(X_full, model['centroids']).cpu().numpy()

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                              cmap='viridis', s=20, alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    c='red', marker='X', s=200, edgecolors='black',
                    linewidths=2, label='Centroids')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.title('K-Means Clustering Result')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kmeans_clusters.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Artifacts saved to {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("K-Means Clustering (From Scratch, k-means++ Init)")
    print("=" * 60)

    set_seed(42)
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Algorithm: {metadata['algorithm']}")
    k = metadata['k']
    print(f"Number of clusters: {k}")

    # Data
    print("\nGenerating 4-cluster blob dataset...")
    X_train, X_val, y_train, y_val, X_np, y_true = make_dataloaders(
        n_samples=800, n_features=2, k=k
    )
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

    # Build model (k-means++ init on training data)
    print("\nInitializing centroids with k-means++...")
    model = build_model(X_train, k=k)
    print(f"Initial centroids shape: {model['centroids'].shape}")

    # Train (run k-means on training data)
    print("\n" + "-" * 60)
    history = train(model, X_train, max_iters=100, tol=1e-6)
    print(f"Finished in {history['n_iterations']} iterations")

    # Evaluate on training set
    print("\n" + "-" * 60)
    print("Evaluating on training set...")
    train_metrics = evaluate(model, X_train, y_true=y_train)
    print(f"Train Inertia: {train_metrics['inertia']:.4f}")
    print(f"Train ARI:     {train_metrics['adjusted_rand_index']:.4f}")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, X_val, y_true=y_val)
    print(f"Val Inertia:   {val_metrics['inertia']:.4f}")
    print(f"Val ARI:       {val_metrics['adjusted_rand_index']:.4f}")

    # Compare with sklearn (on full dataset for fair comparison)
    print("\nRunning sklearn KMeans for comparison...")
    sklearn_km = SklearnKMeans(n_clusters=k, init='k-means++',
                               n_init=10, random_state=42)
    sklearn_km.fit(X_np)
    sklearn_inertia = sklearn_km.inertia_
    sklearn_ari = adjusted_rand_score(y_true, sklearn_km.labels_)
    print(f"Sklearn Inertia: {sklearn_inertia:.4f}")
    print(f"Sklearn ARI:     {sklearn_ari:.4f}")

    # Inertia comparison (use train inertia for ratio)
    total_inertia = train_metrics['inertia'] + val_metrics['inertia']
    inertia_ratio = total_inertia / sklearn_inertia if sklearn_inertia > 0 else 999
    print(f"\nInertia ratio (ours/sklearn): {inertia_ratio:.4f}")

    # Save artifacts
    print("\nSaving artifacts...")
    all_metrics = {
        'train_inertia': train_metrics['inertia'],
        'train_ari': train_metrics['adjusted_rand_index'],
        'val_inertia': val_metrics['inertia'],
        'val_ari': val_metrics['adjusted_rand_index'],
        'sklearn_inertia': sklearn_inertia,
        'sklearn_ari': sklearn_ari,
        'inertia_ratio': inertia_ratio,
        'n_iterations': history['n_iterations'],
        'inertia_history': history['inertia_history'],
        'X_2d': X_np
    }
    save_artifacts(model, all_metrics)

    # Final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Inertia:    {train_metrics['inertia']:.4f}")
    print(f"Train ARI:        {train_metrics['adjusted_rand_index']:.4f}")
    print(f"Val Inertia:      {val_metrics['inertia']:.4f}")
    print(f"Val ARI:          {val_metrics['adjusted_rand_index']:.4f}")
    print(f"Sklearn Inertia:  {sklearn_inertia:.4f}")
    print(f"Sklearn ARI:      {sklearn_ari:.4f}")
    print(f"Inertia Ratio:    {inertia_ratio:.4f}")
    print(f"Iterations:       {history['n_iterations']}")

    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    # Check 1: Inertia non-increasing (converged or decreased)
    inertia_ok = history['inertia_history'][-1] <= history['inertia_history'][0]
    s1 = "PASS" if inertia_ok else "FAIL"
    print(f"[{s1}] Inertia converged: "
          f"{history['inertia_history'][0]:.2f} -> {history['inertia_history'][-1]:.2f}")
    checks_passed = checks_passed and inertia_ok

    # Check 2: Our inertia within 5% of sklearn
    within_5pct = inertia_ratio < 1.05
    s2 = "PASS" if within_5pct else "FAIL"
    print(f"[{s2}] Inertia within 5% of sklearn: ratio = {inertia_ratio:.4f}")
    checks_passed = checks_passed and within_5pct

    # Check 3: Val ARI > 0.8 (good cluster recovery on validation)
    ari_ok = val_metrics['adjusted_rand_index'] > 0.8
    s3 = "PASS" if ari_ok else "FAIL"
    print(f"[{s3}] Val ARI > 0.8: {val_metrics['adjusted_rand_index']:.4f}")
    checks_passed = checks_passed and ari_ok

    # Check 4: Converged within max iterations
    converged = history['n_iterations'] < 100
    s4 = "PASS" if converged else "FAIL"
    print(f"[{s4}] Converged before max iters: {history['n_iterations']} iterations")
    checks_passed = checks_passed and converged

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    sys.exit(0 if checks_passed else 1)
