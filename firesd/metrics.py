"""Streaming routing diagnostics for stable self-distillation experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .routing import RoutingDiagnostics


@dataclass
class RoutingMeter:
    examples: int = 0
    selected_counts: dict[int, int] = field(default_factory=dict)
    max_scores: list[float] = field(default_factory=list)
    mean_scores: list[float] = field(default_factory=list)
    score_margins: list[float] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    selected_scores: list[float] = field(default_factory=list)
    extra_values: dict[str, list[float]] = field(default_factory=dict)

    def update(self, diag: RoutingDiagnostics) -> None:
        self.examples += 1
        if int(diag.gradient_selected) >= 0:
            self.selected_counts[diag.gradient_selected] = self.selected_counts.get(diag.gradient_selected, 0) + 1
        self.max_scores.append(float(diag.max_score))
        self.mean_scores.append(float(diag.mean_score))
        self.score_margins.append(float(diag.score_margin))
        self.entropies.append(float(diag.selection_entropy))
        if diag.selected_score is not None:
            self.selected_scores.append(float(diag.selected_score))
        # For self-calibrated FIRE, projection_weight is the total field mass.
        self.add_extra("fire_field_mass", diag.projection_weight)
        self.add_extra("fire_projection_weight", diag.projection_weight)
        self.add_extra("fire_retained_influence", diag.retained_influence)
        self.add_extra("fire_active_fields", diag.active_fields)
        self.add_extra("fire_trust_coefficient", diag.trust_coefficient)
        self.add_extra("fire_logit_grad_norm_before", diag.logit_grad_norm_before)
        self.add_extra("fire_logit_grad_norm_after", diag.logit_grad_norm_after)
        self.add_extra("fire_fisher_radius", diag.fisher_radius)
        self.add_extra("fire_max_field_weight", diag.max_field_weight)
        self.add_extra("fire_step_scale", diag.step_scale)

    def add_extra(self, name: str, value: float | int | None) -> None:
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(number):
            return
        self.extra_values.setdefault(str(name), []).append(number)

    def summary(self, prefix: str = "routing") -> dict[str, float | int | None]:
        selected_total = max(sum(self.selected_counts.values()), 1)
        result: dict[str, float | int | None] = {
            f"{prefix}_examples": self.examples,
            f"{prefix}_mean_max_score": float(np.mean(self.max_scores)) if self.max_scores else None,
            f"{prefix}_mean_score": float(np.mean(self.mean_scores)) if self.mean_scores else None,
            f"{prefix}_mean_margin": float(np.mean(self.score_margins)) if self.score_margins else None,
            f"{prefix}_mean_entropy": float(np.mean(self.entropies)) if self.entropies else None,
            f"{prefix}_mean_selected_score": float(np.mean(self.selected_scores)) if self.selected_scores else None,
        }
        for index in range(8):
            if index in self.selected_counts:
                result[f"{prefix}_select_block_{index}"] = self.selected_counts[index] / selected_total
        for key, values in sorted(self.extra_values.items()):
            if values:
                result[f"{prefix}_{key}"] = float(np.mean(values))
        return result


def format_routing(summary: Mapping[str, float | int | None], prefix: str = "routing") -> str:
    def number(value, places: int = 4):
        return "n/a" if value is None else f"{float(value):.{places}f}"

    block_rates = []
    for key, value in sorted(summary.items()):
        marker = f"{prefix}_select_block_"
        if key.startswith(marker) and value is not None:
            block = key[len(marker) :]
            block_rates.append(f"b{block}={100.0 * float(value):.1f}%")
    blocks = ",".join(block_rates) if block_rates else "n/a"
    extras = []
    field_mass = summary.get(prefix + "_fire_field_mass")
    trust = summary.get(prefix + "_fire_trust_coefficient")
    if field_mass is not None:
        extras.append(f"field-mass={number(field_mass, 3)}")
    if trust is not None:
        extras.append(f"trust={number(trust, 3)}")
    topd_ratio = summary.get(prefix + "_topd_ratio_mean")
    topd_kl = summary.get(prefix + "_topd_approx_kl")
    topd_clip = summary.get(prefix + "_topd_clip_fraction")
    topd_updates = summary.get(prefix + "_topd_internal_updates")
    if topd_ratio is not None:
        extras.append(f"topd-ratio={number(topd_ratio, 3)}")
    if topd_kl is not None:
        extras.append(f"topd-kl={number(topd_kl, 4)}")
    if topd_clip is not None:
        extras.append(f"topd-clip={100.0 * float(topd_clip):.1f}%")
    if topd_updates is not None:
        extras.append(f"topd-updates={float(topd_updates):.0f}")
    suffix = (", " + ", ".join(extras)) if extras else ""
    return (
        f"{prefix}(max-score={number(summary.get(prefix + '_mean_max_score'))}, "
        f"margin={number(summary.get(prefix + '_mean_margin'))}, "
        f"entropy={number(summary.get(prefix + '_mean_entropy'))}, select={blocks}{suffix})"
    )
