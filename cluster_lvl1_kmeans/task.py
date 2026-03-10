"""
K-Means Clustering — from-scratch PyTorch implementation

Objective (within-cluster sum of squared errors):
    J = sum_{k=1}^{K} sum_{x_i in C_k} || x_i - mu_k ||^2

Steps:
    1. Initialise centroids using k-means++ (distance-proportional sampling)
    2. Assign every point to its closest centroid (vectorised L2 via torch.cdist)
    3. Recompute each centroid as the mean of its assigned points
    4. Repeat until centroid shift < tolerance or max iterations

Validates against sklearn.cluster.KMeans; our inertia should be within 5 %.
"""

import os, sys, json
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SkKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
K = 4


# ── helpers (private) ─────────────────────────────────────────────

def _kpp_init(X, k):
    """k-means++ centroid seeding."""
    n = X.shape[0]
    centres = [X[torch.randint(0, n, (1,), device=X.device).item()]]
    for _ in range(1, k):
        stk = torch.stack(centres)
        d2 = torch.cdist(X.unsqueeze(0), stk.unsqueeze(0)).squeeze(0).min(1).values ** 2
        idx = torch.multinomial(d2 / d2.sum(), 1).item()
        centres.append(X[idx])
    return torch.stack(centres)


def _labels(X, C):
    return torch.cdist(X, C).argmin(1)


def _inertia(X, C, lab):
    return ((X - C[lab]) ** 2).sum().item()


# ── protocol functions ────────────────────────────────────────────

def get_task_metadata():
    return {
        'task_name':   'cluster_lvl1_kmeans',
        'task_type':   'clustering',
        'algorithm':   'K-Means (From Scratch)',
        'k': K,
        'description': 'K-Means with k-means++ init and vectorised updates',
    }


def set_seed(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(n_samples=800, n_features=2, k=K, val_ratio=0.2):
    """Synthetic blobs, standardised, split 80/20."""
    set_seed()
    X, y = make_blobs(n_samples=n_samples, centers=k,
                      n_features=n_features, cluster_std=1.0, random_state=SEED)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    perm = np.random.permutation(n_samples)
    cut = int(n_samples * val_ratio)
    tr_i, va_i = perm[cut:], perm[:cut]

    dev = get_device()
    Xtr = torch.FloatTensor(Xs[tr_i]).to(dev)
    Xva = torch.FloatTensor(Xs[va_i]).to(dev)
    return Xtr, Xva, y[tr_i], y[va_i], Xs, y


def build_model(X, k=K):
    return {'centroids': _kpp_init(X, k), 'k': k}


def train(model, X, max_iters=100, tol=1e-6):
    """Lloyd's algorithm until convergence."""
    C = model['centroids'].clone()
    k = model['k']
    inertia_log = []

    for it in range(max_iters):
        lab = _labels(X, C)
        C_new = torch.zeros_like(C)
        for j in range(k):
            mask = lab == j
            C_new[j] = X[mask].mean(0) if mask.any() else X[torch.randint(0, len(X), (1,))]
        sse = _inertia(X, C_new, lab)
        inertia_log.append(sse)
        shift = ((C_new - C) ** 2).sum().item()
        C = C_new

        if (it + 1) % 10 == 0 or shift < tol:
            print(f"  iter {it+1:>3d}  inertia={sse:.2f}  shift={shift:.1e}")
        if shift < tol:
            print(f"  converged at iteration {it+1}")
            break

    model['centroids'] = C
    return dict(inertia_history=inertia_log, n_iterations=it + 1,
                final_inertia=inertia_log[-1])


def evaluate(model, X, y_true=None):
    C = model['centroids']
    lab = _labels(X, C)
    out = dict(inertia=_inertia(X, C, lab), labels=lab.cpu().numpy())
    if y_true is not None:
        out['adjusted_rand_index'] = adjusted_rand_score(y_true, lab.cpu().numpy())
    return out


def predict(model, X):
    C = model['centroids']
    lab = _labels(X, C)
    return lab.cpu().numpy(), torch.cdist(X, C).cpu().numpy()


def save_artifacts(model, metrics, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model['centroids'].cpu(), os.path.join(output_dir, 'centroids.pth'))

    safe = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):    safe[k] = v.tolist()
        elif isinstance(v, (list, float, int, str, bool)): safe[k] = v
        elif isinstance(v, dict):
            safe[k] = {sk: (sv.tolist() if isinstance(sv, np.ndarray) else sv)
                       for sk, sv in v.items()}
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(safe, f, indent=2)

    # convergence plot
    if 'inertia_history' in metrics:
        plt.figure(figsize=(7, 4))
        plt.plot(metrics['inertia_history'], 'o-', markersize=3)
        plt.xlabel('Iteration'); plt.ylabel('Inertia (SSE)')
        plt.title('K-Means convergence'); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kmeans_inertia.png'), dpi=120)
        plt.close()

    # scatter (2-D only)
    if 'X_2d' in metrics:
        Xp = metrics['X_2d']
        Cf = model['centroids'].cpu().numpy()
        dev = model['centroids'].device
        full_lab = _labels(torch.FloatTensor(Xp).to(dev),
                           model['centroids']).cpu().numpy()
        plt.figure(figsize=(8, 6))
        plt.scatter(Xp[:, 0], Xp[:, 1], c=full_lab, cmap='tab10',
                    s=15, alpha=0.6)
        plt.scatter(Cf[:, 0], Cf[:, 1], c='red', marker='X',
                    s=180, edgecolors='k', linewidths=1.5, label='centroids')
        plt.legend(); plt.title('Cluster assignments')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kmeans_clusters.png'), dpi=120)
        plt.close()

    print(f"[save_artifacts] wrote to {output_dir}/")


# ── main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print(" K-Means  |  from scratch, k-means++ init")
    print("=" * 55)
    set_seed()
    meta = get_task_metadata()
    k = meta['k']
    print(f"task: {meta['task_name']}   k={k}\n")

    Xtr, Xva, ytr, yva, X_np, y_all = make_dataloaders(n_samples=800, k=k)
    print(f"data : {Xtr.shape[0]} train / {Xva.shape[0]} val")

    model = build_model(Xtr, k=k)
    hist = train(model, Xtr, max_iters=100, tol=1e-6)
    print(f"finished in {hist['n_iterations']} iterations\n")

    tr_m = evaluate(model, Xtr, y_true=ytr)
    va_m = evaluate(model, Xva, y_true=yva)
    print(f"  train  inertia={tr_m['inertia']:.2f}  ARI={tr_m['adjusted_rand_index']:.4f}")
    print(f"  val    inertia={va_m['inertia']:.2f}  ARI={va_m['adjusted_rand_index']:.4f}")

    # sklearn baseline (training data only — fair comparison)
    Xtr_np = Xtr.cpu().numpy()
    sk = SkKMeans(n_clusters=k, init='k-means++', n_init=10, random_state=SEED).fit(Xtr_np)
    sk_ari = adjusted_rand_score(ytr, sk.labels_)
    ratio = tr_m['inertia'] / sk.inertia_ if sk.inertia_ > 0 else 999
    print(f"  sklearn inertia={sk.inertia_:.2f}  ARI={sk_ari:.4f}  (train only)")
    print(f"  our/sklearn inertia ratio = {ratio:.4f}")

    save_artifacts(model, dict(
        train_inertia=tr_m['inertia'], train_ari=tr_m['adjusted_rand_index'],
        val_inertia=va_m['inertia'],   val_ari=va_m['adjusted_rand_index'],
        sklearn_inertia=sk.inertia_,   sklearn_ari=sk_ari,
        inertia_ratio=ratio, n_iterations=hist['n_iterations'],
        inertia_history=hist['inertia_history'], X_2d=X_np))

    print(f"\n{'='*55}")
    print(" QUALITY CHECKS")
    print(f"{'='*55}")
    ok = True
    for tag, cond in [
        (f"inertia converged         "
         f"({hist['inertia_history'][0]:.1f} -> {hist['inertia_history'][-1]:.1f})",
         hist['inertia_history'][-1] <= hist['inertia_history'][0]),
        (f"within 5 % of sklearn     (ratio={ratio:.4f})",  ratio < 1.05),
        (f"val ARI > 0.8             ({va_m['adjusted_rand_index']:.4f})",
         va_m['adjusted_rand_index'] > 0.8),
        (f"converged < 100 iters     ({hist['n_iterations']})",
         hist['n_iterations'] < 100),
    ]:
        print(f"  [{'PASS' if cond else 'FAIL'}] {tag}")
        ok = ok and cond

    print(f"\n{'PASS' if ok else 'FAIL'}: all quality checks {'passed' if ok else 'did not pass'}")
    sys.exit(0 if ok else 1)
