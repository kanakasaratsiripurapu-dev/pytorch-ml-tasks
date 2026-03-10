r"""
Multiclass Softmax Regression — nn.Module + Adam

Softmax function:
    $P(y=k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$

Cross-entropy loss:
    $L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K}
         y_{ik} \log P(y=k \mid \mathbf{x}_i)$

Trained on 3-class synthetic blobs with standardised features.
Saves a decision-boundary contour plot as logreg_lvl2_boundary.png.
"""

import os, sys, json
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

SEED = 42
NUM_CLASSES = 3


# ── protocol functions ────────────────────────────────────────────

def get_task_metadata():
    return {
        'task_name':   'logreg_lvl2_multiclass_softmax',
        'task_type':   'classification',
        'num_classes': NUM_CLASSES,
        'input_dim':   2,
        'algorithm':   'Softmax Regression (Multiclass)',
        'description': '3-class softmax with nn.Module, Adam, and CrossEntropyLoss',
    }


def set_seed(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SoftmaxNet(nn.Module):
    """A single fully-connected layer — logistic regression in disguise."""
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x)          # raw logits; softmax handled by CE loss


def make_dataloaders(n_samples=600, num_classes=NUM_CLASSES,
                     val_ratio=0.2, batch_size=32):
    """3-class blobs, standardised, wrapped in DataLoaders."""
    set_seed()
    X_raw, y_raw = make_blobs(n_samples=n_samples, centers=num_classes,
                              n_features=2, cluster_std=1.5, random_state=SEED)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_raw)

    ds = TensorDataset(torch.FloatTensor(X_sc), torch.LongTensor(y_raw))
    n_val = int(n_samples * val_ratio)
    tr_ds, va_ds = random_split(ds, [n_samples - n_val, n_val],
                                generator=torch.Generator().manual_seed(SEED))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    return tr_dl, va_dl, (X_sc, y_raw)


def build_model(input_dim=2, num_classes=NUM_CLASSES):
    return SoftmaxNet(input_dim, num_classes).to(get_device())


def train(model, train_loader, val_loader, epochs=100, lr=0.01):
    dev = get_device()
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    log_tr, log_va, log_f1 = [], [], []
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            running += loss.item()
        avg = running / len(train_loader)
        log_tr.append(avg)

        vm = evaluate(model, val_loader, keep_preds=False)
        log_va.append(vm['loss']); log_f1.append(vm['macro_f1'])

        if (ep + 1) % 25 == 0:
            print(f"  ep {ep+1:>3d}/{epochs}  "
                  f"tr_loss={avg:.4f}  val_loss={vm['loss']:.4f}  "
                  f"val_f1={vm['macro_f1']:.3f}")

    return dict(train_losses=log_tr, val_losses=log_va, val_f1_scores=log_f1)


def evaluate(model, loader, keep_preds=True):
    dev = get_device()
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss, preds, tgts = 0.0, [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb)
            tot_loss += ce(out, yb).item()
            preds += out.argmax(1).cpu().tolist()
            tgts  += yb.cpu().tolist()

    acc = accuracy_score(tgts, preds)
    f1  = f1_score(tgts, preds, average='macro')
    m = dict(loss=tot_loss / len(loader), accuracy=acc, macro_f1=f1)
    if keep_preds:
        m['predictions'] = np.array(preds)
        m['targets'] = np.array(tgts)
    return m


def predict(model, loader):
    dev = get_device(); model.eval()
    all_p, all_prob = [], []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(dev))
            all_p  += logits.argmax(1).cpu().tolist()
            all_prob.append(torch.softmax(logits, 1).cpu().numpy())
    return np.array(all_p), np.vstack(all_prob)


def _plot_boundary(model, X, y, out_dir):
    """Contour of class regions overlaid with training points."""
    dev = get_device()
    pad = 1.0
    xr = np.arange(X[:, 0].min() - pad, X[:, 0].max() + pad, 0.02)
    yr = np.arange(X[:, 1].min() - pad, X[:, 1].max() + pad, 0.02)
    xx, yy = np.meshgrid(xr, yr)

    grid_t = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(dev)
    model.eval()
    with torch.no_grad():
        Z = model(grid_t).argmax(1).cpu().numpy().reshape(xx.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap='Set2')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Set2',
                edgecolors='k', s=25, linewidths=0.5)
    plt.title('Decision boundary — 3-class softmax regression')
    plt.xlabel('x1 (standardised)'); plt.ylabel('x2 (standardised)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'logreg_lvl2_boundary.png'), dpi=120)
    plt.close()


def save_artifacts(model, metrics, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    safe = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):   safe[k] = v.tolist()
        elif isinstance(v, (list, dict, float, int, str, bool)): safe[k] = v
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(safe, f, indent=2)
    print(f"[save_artifacts] wrote to {output_dir}/")


# ── main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print(" Softmax Regression  |  3-class blobs, nn.Module")
    print("=" * 55)
    set_seed()
    meta = get_task_metadata()
    print(f"task: {meta['task_name']}\n")

    tr_dl, va_dl, (X_all, y_all) = make_dataloaders()
    print(f"data : {len(tr_dl.dataset)} train / {len(va_dl.dataset)} val")

    model = build_model()
    print(f"model: {model}\n")

    hist = train(model, tr_dl, va_dl, epochs=100, lr=0.01)

    tr_m = evaluate(model, tr_dl)
    va_m = evaluate(model, va_dl)

    out_dir = './output'
    _plot_boundary(model, X_all, y_all, out_dir)
    save_artifacts(model, dict(
        train_acc=tr_m['accuracy'], train_f1=tr_m['macro_f1'],
        val_acc=va_m['accuracy'],   val_f1=va_m['macro_f1'],
        train_losses=hist['train_losses'], val_losses=hist['val_losses'],
        val_f1_scores=hist['val_f1_scores'],
    ), output_dir=out_dir)

    print(f"\n{'='*55}")
    print(" RESULTS")
    print(f"{'='*55}")
    print(f"  train  acc={tr_m['accuracy']:.4f}   macro-F1={tr_m['macro_f1']:.4f}")
    print(f"  val    acc={va_m['accuracy']:.4f}   macro-F1={va_m['macro_f1']:.4f}")

    ok = True
    for tag, cond in [
        (f"val macro-F1 > 0.85    ({va_m['macro_f1']:.4f})",    va_m['macro_f1'] > 0.85),
        (f"val accuracy > 0.80    ({va_m['accuracy']:.4f})",     va_m['accuracy'] > 0.80),
        (f"train accuracy > 0.80  ({tr_m['accuracy']:.4f})",     tr_m['accuracy'] > 0.80),
        (f"acc gap < 0.10         ({abs(tr_m['accuracy']-va_m['accuracy']):.4f})",
                                    abs(tr_m['accuracy'] - va_m['accuracy']) < 0.10),
    ]:
        print(f"  [{'PASS' if cond else 'FAIL'}] {tag}")
        ok = ok and cond

    print(f"\n{'PASS' if ok else 'FAIL'}: all quality checks {'passed' if ok else 'did not pass'}")
    sys.exit(0 if ok else 1)
