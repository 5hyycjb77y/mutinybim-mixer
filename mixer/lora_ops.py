"""Exact dense-delta weight-space operations for Gemma 4 31B LoRA adapters.

Pure-Python spec/key/schedule helpers are always importable without PyTorch;
tensor operations lazily import ``torch`` and ``safetensors`` so local tests
and lightweight tools run without GPU dependencies.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.+?)\.(?P<factor>lora_A|lora_B)(?:\.(?P<adapter>[^.]+))?\.weight$"
)
_LAYER_IDX_RE = re.compile(r"\.layers\.(?P<idx>\d+)\.")


@dataclass(frozen=True)
class ParsedLoraKey:
    prefix: str
    factor: str  # "lora_A" or "lora_B"
    layer_idx: int | None
    module_name: str
    adapter_name: str | None


def load_spec(path: str | Path = "spec.json") -> dict[str, Any]:
    """Load the pinned `spec.json` weight-space contract."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_lora_key(key: str) -> ParsedLoraKey | None:
    """Parse a PEFT safetensors tensor key into prefix, factor, layer index, and module."""
    m = _LORA_KEY_RE.match(key)
    if not m:
        return None
    prefix = m.group("prefix")
    factor = m.group("factor")
    adapter_name = m.group("adapter")
    layer_m = _LAYER_IDX_RE.search(prefix)
    layer_idx = int(layer_m.group("idx")) if layer_m else None
    module_name = prefix.rsplit(".", 1)[-1]
    return ParsedLoraKey(
        prefix=prefix,
        factor=factor,
        layer_idx=layer_idx,
        module_name=module_name,
        adapter_name=adapter_name,
    )


def depth_band_weights(band_weights: list[float], num_layers: int = 60) -> list[float]:
    """Map ``K`` depth-band weights smoothly across ``num_layers`` transformer layers.

    Uses piecewise-linear interpolation between band centers so EvoMerge can
    search a compact ``K``-dimensional schedule (e.g. 6 bands across Gemma 4's
    60 layers) without abrupt step discontinuities.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not band_weights:
        raise ValueError("band_weights must be non-empty")
    k = len(band_weights)
    if k == 1:
        return [float(band_weights[0])] * num_layers

    out: list[float] = []
    for layer_idx in range(num_layers):
        # Map layer_idx in [0, num_layers - 1] onto band coordinate in [0, k - 1].
        pos = (layer_idx / max(1, num_layers - 1)) * (k - 1)
        lo = int(math.floor(pos))
        hi = min(k - 1, lo + 1)
        frac = pos - lo
        val = (1.0 - frac) * float(band_weights[lo]) + frac * float(band_weights[hi])
        out.append(float(val))
    return out


def check_adapter_compatibility(adapter_config: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Return a list of incompatibility reasons against ``spec.json`` (empty if compatible)."""
    issues: list[str] = []
    base_spec = spec.get("base_model", {})
    lora_spec = spec.get("lora", {})

    allowed_bases = {base_spec.get("hf_id"), base_spec.get("unsloth_id")}
    allowed_bases.discard(None)
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base and allowed_bases and adapter_base not in allowed_bases:
        issues.append(
            f"Base model mismatch: adapter trained on '{adapter_base}', expected one of {sorted(allowed_bases)}."
        )

    expected_modules = set(lora_spec.get("target_modules", []))
    raw_modules = adapter_config.get("target_modules")
    if isinstance(raw_modules, list) and expected_modules:
        adapter_modules = set(raw_modules)
        missing = sorted(expected_modules - adapter_modules)
        extra = sorted(adapter_modules - expected_modules)
        if missing:
            issues.append(f"Adapter missing target_modules required by spec: {missing}")
        if extra:
            issues.append(f"Adapter includes non-spec target_modules: {extra}")

    bias = adapter_config.get("bias")
    if bias is not None and lora_spec.get("bias") is not None and bias != lora_spec.get("bias"):
        issues.append(f"Bias mismatch: adapter has '{bias}', spec pins '{lora_spec.get('bias')}'.")

    return issues


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for tensor weight-space operations. Run in Google Colab or install mutinybim-mixer[colab]."
        ) from exc


def _resolve_adapter_dir(adapter_dir_or_hf_id: str | Path, revision: str | None = None) -> Path:
    path = Path(adapter_dir_or_hf_id)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Local path '{adapter_dir_or_hf_id}' not found and huggingface_hub is not installed."
        ) from exc
    downloaded = snapshot_download(
        repo_id=str(adapter_dir_or_hf_id),
        revision=revision,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors", "eval_scorecard.json"],
    )
    return Path(downloaded)


def load_lora_deltas(
    adapter_dir_or_hf_id: str | Path,
    *,
    device: str = "cpu",
    dtype: Any = None,
    revision: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a PEFT LoRA adapter and expand each ``(A, B)`` pair into exact dense deltas.

    Computes ``ΔW = (lora_alpha / r) * (B @ A)`` so adapters can be linearly
    combined, negated, TIES-elected, or DARE-sparsified without introducing
    invalid cross-terms.
    """
    torch = _require_torch()
    from safetensors.torch import load_file  # type: ignore

    target_dtype = dtype or torch.float32
    adapter_dir = _resolve_adapter_dir(adapter_dir_or_hf_id, revision=revision)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected adapter_config.json and adapter_model.safetensors inside {adapter_dir}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_tensors = load_file(str(weights_path), device=device)

    r = float(config.get("r", 16))
    alpha = float(config.get("lora_alpha", r))
    scale = alpha / max(r, 1e-8)

    pairs: dict[str, dict[str, Any]] = {}
    for key, tensor in raw_tensors.items():
        parsed = parse_lora_key(key)
        if parsed is None:
            continue
        bucket = pairs.setdefault(parsed.prefix, {})
        bucket[parsed.factor] = tensor.to(device=device, dtype=target_dtype)

    deltas: dict[str, Any] = {}
    for prefix, pair in pairs.items():
        if "lora_A" not in pair or "lora_B" not in pair:
            continue
        a_mat = pair["lora_A"]  # [r, in_dim]
        b_mat = pair["lora_B"]  # [out_dim, r]
        deltas[prefix] = (b_mat @ a_mat) * scale

    return deltas, config


def _trim_by_density(tensor: Any, density: float) -> Any:
    torch = _require_torch()
    if density >= 1.0:
        return tensor
    if density <= 0.0:
        return torch.zeros_like(tensor)
    flat = tensor.abs().view(-1)
    k = max(1, int(math.ceil(flat.numel() * density)))
    if k >= flat.numel():
        return tensor
    threshold = torch.topk(flat, k, sorted=True).values[-1]
    return torch.where(tensor.abs() >= threshold, tensor, torch.zeros_like(tensor))


def _dare_sparsify(tensor: Any, drop_rate: float, generator: Any) -> Any:
    torch = _require_torch()
    if drop_rate <= 0.0:
        return tensor
    if drop_rate >= 1.0:
        return torch.zeros_like(tensor)
    keep_prob = 1.0 - drop_rate
    mask = torch.bernoulli(torch.full_like(tensor, keep_prob), generator=generator)
    return (tensor * mask) / keep_prob


def merge_deltas(
    adapter_deltas_list: list[dict[str, Any]],
    weights: list[float],
    *,
    method: str = "ties",
    density: float = 0.5,
    drop_rate: float = 0.2,
    layer_schedules: list[list[float]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Combine multiple dense LoRA delta dicts in weight space.

    Supported ``method`` values:
    - ``"linear"``: Task Arithmetic ``sum_i w_i * ΔW_i`` (supports negative weights for clash negation).
    - ``"ties"``: Trim by magnitude -> Elect consensus sign -> Disjoint weighted mean.
    - ``"dare_linear"``: DARE Bernoulli drop+rescale followed by linear Task Arithmetic.
    - ``"dare_ties"``: DARE Bernoulli drop+rescale followed by TIES sign election & disjoint merge.
    """
    torch = _require_torch()
    if len(adapter_deltas_list) != len(weights):
        raise ValueError("adapter_deltas_list and weights must have the same length")
    if not adapter_deltas_list:
        return {}
    valid_methods = {"linear", "ties", "dare_linear", "dare_ties"}
    if method not in valid_methods:
        raise ValueError(f"Unknown merge method '{method}'. Expected one of {sorted(valid_methods)}")

    all_keys = sorted({k for d in adapter_deltas_list for k in d.keys()})
    gen = torch.Generator(device=next(iter(adapter_deltas_list[0].values())).device)
    gen.manual_seed(seed)

    merged: dict[str, Any] = {}
    for key in all_keys:
        layer_match = _LAYER_IDX_RE.search(key)
        layer_idx = int(layer_match.group("idx")) if layer_match else None

        weighted_terms: list[Any] = []
        eff_weights: list[float] = []
        ref_shape = None
        ref_dtype = None
        ref_device = None

        for idx, deltas in enumerate(adapter_deltas_list):
            if key not in deltas:
                continue
            t = deltas[key]
            ref_shape, ref_dtype, ref_device = t.shape, t.dtype, t.device
            w = float(weights[idx])
            if layer_schedules is not None and idx < len(layer_schedules) and layer_idx is not None:
                sched = layer_schedules[idx]
                if layer_idx < len(sched):
                    w *= float(sched[layer_idx])
            if w == 0.0:
                continue

            prepared = t
            if method in {"dare_linear", "dare_ties"}:
                prepared = _dare_sparsify(prepared, drop_rate, gen)
            if method in {"ties", "dare_ties"}:
                prepared = _trim_by_density(prepared, density)

            # Negative weight flips direction cleanly for Clash-Vector Negation (-λ * ΔW_clash).
            weighted_terms.append(prepared * w)
            eff_weights.append(abs(w))

        if not weighted_terms:
            if ref_shape is not None:
                merged[key] = torch.zeros(ref_shape, dtype=ref_dtype, device=ref_device)
            continue

        if method in {"linear", "dare_linear"}:
            acc = weighted_terms[0].clone()
            for term in weighted_terms[1:]:
                acc.add_(term)
            merged[key] = acc
            continue

        # TIES: Elect consensus sign by weighted magnitude sum, then disjoint mean.
        stacked = torch.stack(weighted_terms, dim=0)  # [N, ...]
        elected_sign = torch.sign(stacked.sum(dim=0))
        agree_mask = (torch.sign(stacked) == elected_sign.unsqueeze(0)) & (elected_sign.unsqueeze(0) != 0)
        weight_tensor = torch.tensor(eff_weights, dtype=stacked.dtype, device=stacked.device).view(-1, *([1] * (stacked.ndim - 1)))
        denom = (agree_mask.to(stacked.dtype) * weight_tensor).sum(dim=0).clamp(min=1e-8)
        numer = torch.where(agree_mask, stacked, torch.zeros_like(stacked)).sum(dim=0)
        # Preserve original total scale when weights sum != 1 by multiplying back average active weight.
        avg_scale = (agree_mask.to(stacked.dtype) * weight_tensor).sum(dim=0) / agree_mask.to(stacked.dtype).sum(dim=0).clamp(min=1.0)
        merged[key] = (numer / denom) * avg_scale

    return merged


def refactorize_deltas_to_lora(
    merged_deltas: dict[str, Any],
    *,
    out_rank: int = 16,
    lora_alpha: int = 16,
    use_lowrank: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Factor dense merged ``ΔW`` matrices back into PEFT-compatible ``(lora_A, lora_B)``.

    Guarantees ``(lora_alpha / out_rank) * (lora_B @ lora_A) ≈ ΔW`` via truncated SVD.
    Returns ``(lora_state_dict, diagnostics)``.
    """
    torch = _require_torch()
    lora_tensors: dict[str, Any] = {}
    scale = float(lora_alpha) / max(float(out_rank), 1e-8)
    rel_errors: list[float] = []

    for prefix, delta in merged_deltas.items():
        mat = delta.to(torch.float32)
        rows, cols = mat.shape
        rank = min(out_rank, rows, cols)

        if use_lowrank and rank < min(rows, cols):
            u, s, v = torch.svd_lowrank(mat, q=rank, niter=4)
            vh = v.t()
        else:
            u_full, s_full, vh_full = torch.linalg.svd(mat, full_matrices=False)
            u = u_full[:, :rank]
            s = s_full[:rank]
            vh = vh_full[:rank, :]

        # Pad to out_rank if matrix is smaller than requested rank.
        if rank < out_rank:
            pad_r = out_rank - rank
            u = torch.cat([u, torch.zeros(rows, pad_r, dtype=u.dtype, device=u.device)], dim=1)
            s = torch.cat([s, torch.zeros(pad_r, dtype=s.dtype, device=s.device)], dim=0)
            vh = torch.cat([vh, torch.zeros(pad_r, cols, dtype=vh.dtype, device=vh.device)], dim=0)

        sqrt_s = torch.sqrt(torch.clamp(s, min=0.0) / scale)
        lora_b = u * sqrt_s.unsqueeze(0)  # [out_dim, r]
        lora_a = sqrt_s.unsqueeze(1) * vh  # [r, in_dim]

        recon = (lora_b @ lora_a) * scale
        denom = float(torch.linalg.norm(mat).item())
        if denom > 1e-8:
            rel_err = float(torch.linalg.norm(mat - recon).item()) / denom
            rel_errors.append(rel_err)

        lora_tensors[f"{prefix}.lora_A.weight"] = lora_a.to(delta.dtype).cpu()
        lora_tensors[f"{prefix}.lora_B.weight"] = lora_b.to(delta.dtype).cpu()

    stats = {
        "num_tensors": float(len(merged_deltas)),
        "mean_relative_svd_error": sum(rel_errors) / len(rel_errors) if rel_errors else 0.0,
        "max_relative_svd_error": max(rel_errors) if rel_errors else 0.0,
    }
    return lora_tensors, stats


def save_merged_lora(
    lora_tensors: dict[str, Any],
    template_config: dict[str, Any],
    out_dir: str | Path,
    *,
    out_rank: int = 16,
    lora_alpha: int = 16,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save refactorized LoRA weights and PEFT ``adapter_config.json`` to ``out_dir``."""
    _require_torch()
    from safetensors.torch import save_file  # type: ignore

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = dict(template_config)
    cfg["r"] = int(out_rank)
    cfg["lora_alpha"] = int(lora_alpha)
    cfg.setdefault("peft_type", "LORA")
    cfg.setdefault("task_type", "CAUSAL_LM")
    cfg.setdefault("bias", "none")

    (out_path / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    save_file(lora_tensors, str(out_path / "adapter_model.safetensors"))
    if metadata is not None:
        (out_path / "mixer_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out_path


def svd_diagnostics(
    deltas_a: dict[str, Any],
    deltas_b: dict[str, Any] | None = None,
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    """Compute per-layer Frobenius norms, singular-value energy concentration, and trade cosine similarity."""
    torch = _require_torch()
    per_layer: dict[int, dict[str, float]] = {}
    all_cos: list[float] = []

    for key, mat_a in deltas_a.items():
        layer_m = _LAYER_IDX_RE.search(key)
        layer_idx = int(layer_m.group("idx")) if layer_m else -1
        a32 = mat_a.to(torch.float32)
        frob_a = float(torch.linalg.norm(a32).item())

        # Top-k singular energy ratio reveals how low-rank the learned update truly is.
        s_vals = torch.linalg.svdvals(a32)
        total_energy = float((s_vals**2).sum().item())
        top_energy = float((s_vals[:top_k] ** 2).sum().item())
        top_k_ratio = (top_energy / total_energy) if total_energy > 1e-12 else 0.0

        bucket = per_layer.setdefault(
            layer_idx,
            {"frob_sum_a": 0.0, "top_k_ratio_sum": 0.0, "cos_sum": 0.0, "count": 0.0, "cos_count": 0.0},
        )
        bucket["frob_sum_a"] += frob_a
        bucket["top_k_ratio_sum"] += top_k_ratio
        bucket["count"] += 1.0

        if deltas_b is not None and key in deltas_b:
            b32 = deltas_b[key].to(torch.float32)
            denom = frob_a * float(torch.linalg.norm(b32).item())
            if denom > 1e-12:
                cos = float(torch.sum(a32 * b32).item()) / denom
                bucket["cos_sum"] += cos
                bucket["cos_count"] += 1.0
                all_cos.append(cos)

    layer_summary = {}
    for layer_idx in sorted(per_layer.keys()):
        b = per_layer[layer_idx]
        c = max(b["count"], 1.0)
        cc = b["cos_count"]
        layer_summary[layer_idx] = {
            "mean_frobenius_norm": b["frob_sum_a"] / c,
            "mean_top_k_energy_ratio": b["top_k_ratio_sum"] / c,
            "mean_cosine_similarity": (b["cos_sum"] / cc) if cc > 0 else None,
        }

    return {
        "layers": layer_summary,
        "global_mean_cosine_similarity": (sum(all_cos) / len(all_cos)) if all_cos else None,
    }
