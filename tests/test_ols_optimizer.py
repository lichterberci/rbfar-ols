import torch
import importlib.util
from pathlib import Path


def _load_ols_optimizer():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "ols-optimizer.py"
    spec = importlib.util.spec_from_file_location("ols_optimizer", str(mod_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.OlsOptimizer


def _make_synthetic(
    l=400, m=80, k_true=5, noise=1e-3, device=None, dtype=torch.float32
):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    P = torch.randn(l, m, device=device, dtype=dtype)
    idx_true = torch.randperm(m, device=device)[:k_true]
    coef_true = torch.randn(k_true, device=device, dtype=dtype)
    d = (P[:, idx_true] @ coef_true) + noise * torch.randn(
        l, device=device, dtype=dtype
    )
    return P, d, idx_true


def test_ols_converges_and_is_sparse():
    torch.manual_seed(0)
    OlsOptimizer = _load_ols_optimizer()
    P, d, idx_true = _make_synthetic()
    opt = OlsOptimizer(rho=1e-2, epsilon=1e-8)
    sel_idx, theta = opt.optimize(P, d)

    # Sanity
    assert sel_idx.ndim == 1 and theta.ndim == 1
    assert sel_idx.numel() == theta.numel() and sel_idx.numel() > 0

    # Convergence
    y = P[:, sel_idx] @ theta
    rel_err = torch.linalg.norm(d - y) / (torch.linalg.norm(d) + 1e-12)
    assert rel_err < 0.12

    # Sparsity and some overlap
    assert sel_idx.numel() <= min(P.shape[1], int(3 * len(idx_true)))
    overlap = len(set(sel_idx.tolist()) & set(idx_true.tolist()))
    assert overlap >= 1
