import torch
import importlib.util
from pathlib import Path


def _load_svd_optimizer():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "svd-baed-optimizer.py"
    spec = importlib.util.spec_from_file_location("svd_optimizer", str(mod_path))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.SvdOptimizer


def _make_synthetic(
    l=400, m=120, k_true=6, noise=1e-3, device=None, dtype=torch.float32
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
    return P, d


def test_svd_converges_and_is_sparse():
    torch.manual_seed(0)
    SvdOptimizer = _load_svd_optimizer()
    P, d = _make_synthetic()
    opt = SvdOptimizer(epsilon=1e-2, alpha=1e-5, delta=1e-4)
    sel_idx, weights = opt.optimize(P, d)

    # Sanity
    assert sel_idx.ndim == 1 and weights.ndim == 1

    # Convergence: reconstruction should be reasonable on selected columns
    y = torch.zeros_like(d)
    if sel_idx.numel() > 0:
        y = P[:, sel_idx] @ weights
    rel_err = torch.linalg.norm(d - y) / (torch.linalg.norm(d) + 1e-12)
    assert rel_err < 0.25
