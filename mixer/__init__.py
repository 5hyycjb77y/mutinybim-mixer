"""MutinyBIM Weight-Space Mixer: exact LoRA delta arithmetic, TIES/DARE, SVD diagnostics, and EvoMerge."""

from .evomerge import EvoMergeConfig, EvoMergeOptimizer, EvoMergeResult
from .lora_ops import (
    ParsedLoraKey,
    check_adapter_compatibility,
    depth_band_weights,
    load_lora_deltas,
    load_spec,
    merge_deltas,
    parse_lora_key,
    refactorize_deltas_to_lora,
    save_merged_lora,
    svd_diagnostics,
)
from .prompts import MULTI_TRADE_BENCHMARK, SYSTEM_PROMPT_TEMPLATE, TRADE_TASKS, format_task_prompt
from .studio_render import ART_DIRECTION_PRESETS, CAMERA_PRESETS, VIBE_INSPIRATION_CHIPS, build_render_prompt

__all__ = [
    "ART_DIRECTION_PRESETS",
    "CAMERA_PRESETS",
    "EvoMergeConfig",
    "EvoMergeOptimizer",
    "EvoMergeResult",
    "MULTI_TRADE_BENCHMARK",
    "ParsedLoraKey",
    "SYSTEM_PROMPT_TEMPLATE",
    "TRADE_TASKS",
    "VIBE_INSPIRATION_CHIPS",
    "build_render_prompt",
    "check_adapter_compatibility",
    "depth_band_weights",
    "format_task_prompt",
    "load_lora_deltas",
    "load_spec",
    "merge_deltas",
    "parse_lora_key",
    "refactorize_deltas_to_lora",
    "save_merged_lora",
    "svd_diagnostics",
]
