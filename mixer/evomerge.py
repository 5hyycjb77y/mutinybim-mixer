"""Evolutionary weight-space optimization (CMA-ES + pure-Python fallback) against bim-lint."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .lora_ops import depth_band_weights


@dataclass
class EvoMergeConfig:
    mode: str = "global"  # "global" or "depth_bands"
    num_bands: int = 6
    num_layers: int = 60
    method: str = "ties"
    population_size: int = 8
    generations: int = 10
    sigma: float = 0.25
    allow_negative_indices: tuple[int, ...] = ()
    search_ties_density: bool = True
    seed: int = 42


@dataclass
class EvoMergeResult:
    best_fitness: float
    best_genome: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "nes"


class EvoMergeOptimizer:
    """Optimize adapter weights (or 60-layer depth-band schedules) against a MutinyBIM verifier score.

    Uses ``cma.CMAEvolutionStrategy`` when the ``cma`` package is installed, and
    falls back seamlessly to a deterministic zero-dependency Separable NES
    optimizer so tests and lightweight environments work everywhere.
    """

    def __init__(self, num_adapters: int, config: EvoMergeConfig | None = None) -> None:
        if num_adapters <= 0:
            raise ValueError("num_adapters must be positive")
        self.num_adapters = num_adapters
        self.config = config or EvoMergeConfig()

    def dimension(self) -> int:
        base = self.num_adapters if self.config.mode == "global" else self.num_adapters * self.config.num_bands
        return base + (1 if self.config.search_ties_density and "ties" in self.config.method else 0)

    def initial_vector(self, init_weights: list[float] | None = None) -> list[float]:
        if init_weights is not None and len(init_weights) != self.num_adapters:
            raise ValueError("init_weights length must match num_adapters")
        defaults = init_weights or [0.8 if i not in self.config.allow_negative_indices else -0.3 for i in range(self.num_adapters)]
        vec: list[float] = []
        if self.config.mode == "global":
            vec.extend(float(w) for w in defaults)
        else:
            for w in defaults:
                vec.extend([float(w)] * self.config.num_bands)
        if self.config.search_ties_density and "ties" in self.config.method:
            vec.append(0.5)
        return vec

    def decode(self, vector: list[float]) -> dict[str, Any]:
        cfg = self.config
        neg_set = set(cfg.allow_negative_indices)

        def _clamp_weight(idx: int, val: float) -> float:
            if idx in neg_set:
                return max(-1.5, min(0.5, float(val)))
            return max(0.0, min(1.5, float(val)))

        if cfg.mode == "global":
            weights = [_clamp_weight(i, vector[i]) for i in range(self.num_adapters)]
            layer_schedules = None
            offset = self.num_adapters
        else:
            weights = [1.0] * self.num_adapters
            layer_schedules = []
            for i in range(self.num_adapters):
                raw_bands = vector[i * cfg.num_bands : (i + 1) * cfg.num_bands]
                clamped_bands = [_clamp_weight(i, v) for v in raw_bands]
                layer_schedules.append(depth_band_weights(clamped_bands, num_layers=cfg.num_layers))
            offset = self.num_adapters * cfg.num_bands

        density = 0.5
        if cfg.search_ties_density and "ties" in cfg.method and offset < len(vector):
            density = max(0.15, min(1.0, float(vector[offset])))

        return {
            "weights": weights,
            "layer_schedules": layer_schedules,
            "density": density,
            "method": cfg.method,
        }

    def optimize(
        self,
        eval_fn: Callable[[dict[str, Any]], float],
        *,
        init_weights: list[float] | None = None,
    ) -> EvoMergeResult:
        x0 = self.initial_vector(init_weights)
        try:
            import cma  # type: ignore

            return self._run_cma(cma, x0, eval_fn)
        except ImportError:
            return self._run_nes(x0, eval_fn)

    def _run_cma(
        self,
        cma_mod: Any,
        x0: list[float],
        eval_fn: Callable[[dict[str, Any]], float],
    ) -> EvoMergeResult:
        cfg = self.config
        opts = {
            "popsize": cfg.population_size,
            "seed": cfg.seed,
            "verbose": -9,
        }
        es = cma_mod.CMAEvolutionStrategy(x0, cfg.sigma, opts)
        best_fit = float("-inf")
        best_vec = list(x0)
        history: list[dict[str, Any]] = []

        for gen in range(cfg.generations):
            candidates = es.ask()
            scores: list[float] = []
            for cand in candidates:
                cand_list = [float(v) for v in cand]
                score = float(eval_fn(self.decode(cand_list)))
                scores.append(score)
                if score > best_fit:
                    best_fit = score
                    best_vec = cand_list
            # CMA-ES minimizes objective; negate maximization fitness.
            es.tell(candidates, [-s for s in scores])
            history.append(
                {
                    "generation": gen,
                    "mean_fitness": sum(scores) / len(scores),
                    "max_fitness": max(scores),
                    "best_so_far": best_fit,
                }
            )

        return EvoMergeResult(
            best_fitness=best_fit,
            best_genome=self.decode(best_vec),
            history=history,
            backend="cma-es",
        )

    def _run_nes(
        self,
        x0: list[float],
        eval_fn: Callable[[dict[str, Any]], float],
    ) -> EvoMergeResult:
        """Zero-dependency Separable NES / hill-climbing ES fallback."""
        cfg = self.config
        rng = random.Random(cfg.seed)
        dim = len(x0)
        mu = list(x0)
        sigma = float(cfg.sigma)

        best_fit = float(eval_fn(self.decode(mu)))
        best_vec = list(mu)
        history: list[dict[str, Any]] = []

        for gen in range(cfg.generations):
            noises: list[list[float]] = []
            scores: list[float] = []
            for _ in range(cfg.population_size):
                eps = [rng.gauss(0.0, 1.0) for _ in range(dim)]
                cand = [mu[d] + sigma * eps[d] for d in range(dim)]
                score = float(eval_fn(self.decode(cand)))
                noises.append(eps)
                scores.append(score)
                if score > best_fit:
                    best_fit = score
                    best_vec = cand

            mean_s = sum(scores) / len(scores)
            var_s = sum((s - mean_s) ** 2 for s in scores) / len(scores)
            std_s = math.sqrt(var_s)
            if std_s > 1e-9:
                norm_scores = [(s - mean_s) / std_s for s in scores]
                lr = 0.25
                for d in range(dim):
                    grad = sum(norm_scores[k] * noises[k][d] for k in range(len(scores))) / len(scores)
                    mu[d] += lr * sigma * grad
            sigma = max(0.04, sigma * 0.92)
            history.append(
                {
                    "generation": gen,
                    "mean_fitness": mean_s,
                    "max_fitness": max(scores),
                    "best_so_far": best_fit,
                }
            )

        return EvoMergeResult(
            best_fitness=best_fit,
            best_genome=self.decode(best_vec),
            history=history,
            backend="nes",
        )
