"""Context-field influence routing for stable on-policy self-distillation.

FIRE is self-calibrated.  It measures the exact leave-one-field-out change in
the reverse-KL student-logit gradient, converts those changes into nonnegative
Fisher-energy barycenter weights, and then contracts the resulting target just
enough to keep its absolute reverse-KL logit-gradient energy below the local
softmax Fisher scale.  The proposed method has no tuned routing budget, ramp,
projection cap, or running controller.

Unconditional deletion of the maximum-influence field remains available as
``hard_gradient_excision`` for a direct hard-removal comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .contexts import ContextRecord, replace_block
from .data import TaskExample
from .ema import LoRAEMA
from .modeling import SequenceBudgetError, response_logits_batch, scoring_budget_errors
from .prompts import REMOVAL_MARKER, render_chat, student_messages, teacher_messages


@dataclass
class RoutingDiagnostics:
    scores: list[float]
    gradient_selected: int
    method_selected: int | None
    selected_score: float | None
    max_score: float
    mean_score: float
    score_margin: float
    selection_entropy: float
    projection_weight: float | None = None
    retained_influence: float | None = None
    active_fields: float | None = None
    trust_coefficient: float | None = None
    logit_grad_norm_before: float | None = None
    logit_grad_norm_after: float | None = None
    fisher_radius: float | None = None
    max_field_weight: float | None = None
    step_scale: float | None = None


@dataclass
class BatchRouteResult:
    target_log_probs: torch.Tensor
    response_mask: torch.Tensor
    diagnostics: list[RoutingDiagnostics]


def normalize_method(method: str) -> str:
    aliases = {
        "fire_no_projection": "fire_attribution_only",
        "fire_no_attribution": "fisher_trust_full_context",
    }
    return aliases.get(str(method), str(method))


def _target_log_probs_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    response_ids: Sequence[torch.Tensor],
    max_context_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scored = response_logits_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        response_ids=response_ids,
        max_context_tokens=max_context_tokens,
        requires_grad=False,
    )
    return torch.log_softmax(scored.logits.float(), dim=-1).detach(), scored.response_mask


def _igt_token_scores_batch(
    p: torch.Tensor,
    full_logq: torch.Tensor,
    candidate_logq: torch.Tensor,
) -> torch.Tensor:
    """Exact half-L1 reverse-KL gradient change at each response token."""
    residual = candidate_logq - full_logq
    centered = residual - (p * residual).sum(dim=-1, keepdim=True)
    delta_gradient = p * centered
    return 0.5 * delta_gradient.abs().sum(dim=-1)


def _reverse_kl_logit_gradient(
    p: torch.Tensor,
    student_logp: torch.Tensor,
    target_logq: torch.Tensor,
) -> torch.Tensor:
    """Exact reverse-KL gradient with respect to the student logits.

    All inputs are detached routing tensors.  The returned tensor has the same
    shape as the distributions and lies in the softmax tangent space.
    """
    residual = student_logp - target_logq
    centered = residual - (p * residual).sum(dim=-1, keepdim=True)
    return p * centered


def _l2_gradient_contrast_batch(
    p: torch.Tensor,
    full_logq: torch.Tensor,
    candidate_logq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact leave-one-out gradient contrast and its squared L2 energy."""
    residual = candidate_logq - full_logq
    centered = residual - (p * residual).sum(dim=-1, keepdim=True)
    delta_gradient = p * centered
    energy = delta_gradient.square().sum(dim=-1)
    return delta_gradient, energy


def _softmax_fisher_energy(p: torch.Tensor) -> torch.Tensor:
    """Trace of the categorical softmax Fisher matrix, 1 - ||p||_2^2."""
    return (1.0 - p.square().sum(dim=-1)).clamp_min(0.0)


def _field_evidence_from_contrast(
    *,
    weighting: str,
    field_energy: torch.Tensor,
    half_l1: torch.Tensor,
    full_disagreement_energy: torch.Tensor | None,
    effective_fisher_energy: torch.Tensor | None,
    response_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return the nonnegative raw field evidence used by the barycenter.

    ``excess_l2_energy`` is the proposed FIRE rule.  It multiplies exact
    field contrast energy by the fraction of full-target gradient energy that
    lies outside the current Fisher step budget.  Thus a field is attenuated
    only when the full target is locally over-demanding.  The other choices are
    explicit mechanism ablations.
    """
    mask = response_mask.to(dtype=field_energy.dtype)
    if weighting == "excess_l2_energy":
        if full_disagreement_energy is None or effective_fisher_energy is None:
            raise ValueError("excess_l2_energy requires full disagreement and effective Fisher energy")
        excess_fraction = torch.where(
            full_disagreement_energy > eps,
            (full_disagreement_energy - effective_fisher_energy).clamp_min(0.0)
            / full_disagreement_energy.clamp_min(eps),
            torch.zeros_like(full_disagreement_energy),
        )
        evidence = field_energy * excess_fraction
    elif weighting == "l2_energy":
        evidence = field_energy
    elif weighting == "uniform":
        evidence = torch.ones_like(field_energy)
    elif weighting == "half_l1":
        evidence = half_l1
    elif weighting == "normalized_disagreement":
        if full_disagreement_energy is None:
            raise ValueError("normalized_disagreement requires full teacher-student disagreement energy")
        evidence = torch.where(
            full_disagreement_energy > eps,
            field_energy / full_disagreement_energy.clamp_min(eps),
            torch.zeros_like(field_energy),
        )
    else:
        raise ValueError(f"Unknown FIRE field weighting {weighting!r}")
    return evidence.clamp_min(0.0) * mask


def _fisher_attribution_target_from_accumulators(
    *,
    p: torch.Tensor,
    full_logq: torch.Tensor,
    weighted_residual_sum: torch.Tensor,
    field_energy_sum: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float,
    step_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the parameter-free, step-calibrated attribution barycenter.

    Let nu^2 = tr F(p), s >= 1 be the learning-rate step scale, and w_j
    be the configured nonnegative field evidence.  The exponential barycenter
    weights are alpha_0 = (nu^2 / s^2) / Z and alpha_j = w_j / Z.  For main
    FIRE, w_j is excess-gated squared contrast energy.  The returned
    ``field_mass`` is sum_j alpha_j.
    """
    if p.shape != full_logq.shape:
        raise ValueError("FIRE student and teacher tensors must have identical shapes.")
    if weighted_residual_sum.shape != full_logq.shape:
        raise ValueError("FIRE residual accumulator must match full_logq.")
    if field_energy_sum.shape != full_logq.shape[:2]:
        raise ValueError("FIRE field-energy accumulator must match [batch, response].")
    if response_mask.shape != field_energy_sum.shape:
        raise ValueError("FIRE response mask must match field-energy accumulator.")

    if float(step_scale) < 1.0:
        raise ValueError("step_scale must be at least 1")
    mask = response_mask.to(dtype=full_logq.dtype)
    fisher_energy = _softmax_fisher_energy(p) * mask
    effective_fisher_energy = fisher_energy / (float(step_scale) ** 2)
    denominator = (effective_fisher_energy + field_energy_sum).clamp_min(eps)
    field_mass = torch.where(
        response_mask,
        field_energy_sum / denominator,
        torch.zeros_like(field_energy_sum),
    )
    attribution_raw = full_logq + weighted_residual_sum / denominator[:, :, None]
    attribution_logq = torch.log_softmax(attribution_raw, dim=-1)
    return attribution_logq, field_mass, effective_fisher_energy


def _fisher_radial_project_target(
    *,
    p: torch.Tensor,
    student_logp: torch.Tensor,
    source_logq: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float,
    step_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a detached target into a learning-rate-normalized Fisher ball.

    Along the exponential path from the detached student to ``source_logq``,
    reverse-KL logit gradients scale exactly by eta.  Choosing
    eta = min(1, (nu / step_scale) / ||g_source||_2) is the largest
    coefficient satisfying ||g_final||_2 <= nu / step_scale, where
    nu^2 = tr F(p).  With step_scale=max(1, lr/nominal_lr), this also gives
    lr * ||g_final||_2 <= nominal_lr * nu.
    """
    if p.shape != source_logq.shape or student_logp.shape != source_logq.shape:
        raise ValueError("Fisher projection tensors must have identical shapes.")
    if response_mask.shape != source_logq.shape[:2]:
        raise ValueError("Fisher projection response mask must match [batch, response].")

    if float(step_scale) < 1.0:
        raise ValueError("step_scale must be at least 1")
    mask = response_mask.to(dtype=source_logq.dtype)
    fisher_energy = _softmax_fisher_energy(p) * mask
    fisher_radius = torch.sqrt(fisher_energy.clamp_min(0.0)) / float(step_scale)
    g_source = _reverse_kl_logit_gradient(p, student_logp, source_logq)
    source_energy = g_source.square().sum(dim=-1) * mask
    before_norm = torch.sqrt(source_energy.clamp_min(0.0))
    radial_scale = fisher_radius / before_norm.clamp_min(eps)
    trust = torch.where(
        before_norm <= fisher_radius,
        torch.ones_like(before_norm),
        radial_scale,
    )
    trust = torch.where(
        response_mask,
        trust.clamp(min=0.0, max=1.0),
        torch.ones_like(trust),
    )
    final_raw = (1.0 - trust[:, :, None]) * student_logp + trust[:, :, None] * source_logq
    target = torch.log_softmax(final_raw, dim=-1)
    after_norm = trust * before_norm
    return target, trust, before_norm, after_norm, fisher_radius


def _softmax_entropy(scores: list[float]) -> float:
    if not scores:
        return 0.0
    finite = [float(s) for s in scores if math.isfinite(float(s))]
    if not finite:
        return 0.0
    maximum = max(finite)
    exps = [math.exp(float(s) - maximum) for s in scores]
    total = sum(exps)
    if total <= 0:
        return 0.0
    probs = [value / total for value in exps]
    return -sum(prob * math.log(prob + 1.0e-12) for prob in probs)


def routing_budget_errors(
    *,
    tokenizer,
    examples: Sequence[TaskExample],
    response_ids: Sequence[torch.Tensor],
    contexts: Sequence[ContextRecord],
    max_context_tokens: int,
) -> list[SequenceBudgetError | None]:
    if len(examples) != len(response_ids) or len(examples) != len(contexts):
        raise ValueError("examples, response_ids, and contexts must have the same batch length.")
    if not examples:
        return []
    block_count = len(contexts[0].blocks)
    if block_count < 1 or any(len(context.blocks) != block_count for context in contexts):
        raise ValueError("All routing items must contain the same positive context-block count.")

    prompt_groups: list[list[str]] = [
        [render_chat(tokenizer, student_messages(example)) for example in examples],
        [
            render_chat(
                tokenizer,
                teacher_messages(example, context.blocks, context.block_fields, context_type=context.context_type),
            )
            for example, context in zip(examples, contexts)
        ],
    ]
    for index in range(block_count):
        prompt_groups.append(
            [
                render_chat(
                    tokenizer,
                    teacher_messages(
                        example,
                        replace_block(context.blocks, index, REMOVAL_MARKER),
                        context.block_fields,
                        context_type=context.context_type,
                    ),
                )
                for example, context in zip(examples, contexts)
            ]
        )

    errors: list[SequenceBudgetError | None] = [None] * len(examples)
    for prompts in prompt_groups:
        group_errors = scoring_budget_errors(tokenizer, prompts, response_ids, max_context_tokens=max_context_tokens)
        for row, error in enumerate(group_errors):
            if errors[row] is None and error is not None:
                errors[row] = error
    return errors


def route_targets_batch(
    *,
    model,
    tokenizer,
    ema: LoRAEMA,
    examples: Sequence[TaskExample],
    response_ids: Sequence[torch.Tensor],
    contexts: Sequence[ContextRecord],
    method: str,
    max_context_tokens: int,
    fire_cfg: dict[str, Any] | None = None,
    step_scale: float = 1.0,
    candidate_chunk_blocks: int = 1,
    collect_diagnostics: bool = True,
) -> BatchRouteResult:
    if not examples:
        raise ValueError("Cannot route an empty batch.")
    if len(examples) != len(response_ids) or len(examples) != len(contexts):
        raise ValueError("examples, response_ids, and contexts must have the same batch length.")

    method = normalize_method(method)
    valid_methods = {
        "fire",
        "fire_attribution_only",
        "fisher_trust_full_context",
        "hard_gradient_excision",
        "full_context",
    }
    if method not in valid_methods:
        raise ValueError(f"Unknown routing method {method!r}; expected one of {sorted(valid_methods)}")

    block_count = len(contexts[0].blocks)
    if block_count < 1:
        raise ValueError("At least one context block is required for routing.")
    if any(len(context.blocks) != block_count for context in contexts):
        raise ValueError("All examples in a routed microbatch must use the same number of context blocks.")

    fire_cfg = dict(fire_cfg or {})
    configured_step_scale = float(step_scale) if bool(fire_cfg.get("step_normalized", False)) else 1.0
    if configured_step_scale < 1.0:
        raise ValueError("FIRE step_scale must be at least 1")
    batch_size = len(examples)
    student_prompts = [render_chat(tokenizer, student_messages(example)) for example in examples]
    full_teacher_prompts = [
        render_chat(
            tokenizer,
            teacher_messages(example, context.blocks, context.block_fields, context_type=context.context_type),
        )
        for example, context in zip(examples, contexts)
    ]

    need_all_candidates = method in {
        "fire",
        "fire_attribution_only",
        "hard_gradient_excision",
    } or (collect_diagnostics and method != "fisher_trust_full_context")
    need_student_distribution = need_all_candidates or method == "fisher_trust_full_context"

    model_was_training = model.training
    model.eval()
    response_mask: torch.Tensor | None = None
    p: torch.Tensor | None = None
    student_logp: torch.Tensor | None = None
    full_logq: torch.Tensor | None = None
    target: torch.Tensor | None = None
    score_columns: list[torch.Tensor] = []
    field_energy_columns: list[torch.Tensor] = []
    field_evidence_columns: list[torch.Tensor] = []
    best_scores: torch.Tensor | None = None
    gradient_selected: torch.Tensor | None = None
    best_target: torch.Tensor | None = None
    method_selected: list[int | None] = [None] * batch_size
    projection_weights: torch.Tensor | None = None
    retained_influence: torch.Tensor | None = None
    active_fields: torch.Tensor | None = None
    trust_coefficients: torch.Tensor | None = None
    logit_grad_norm_before: torch.Tensor | None = None
    logit_grad_norm_after: torch.Tensor | None = None
    fisher_radius: torch.Tensor | None = None
    max_field_weight: torch.Tensor | None = None

    try:
        with torch.no_grad():
            if need_student_distribution:
                student_scored = response_logits_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=student_prompts,
                    response_ids=response_ids,
                    max_context_tokens=max_context_tokens,
                    requires_grad=False,
                )
                student_logp = torch.log_softmax(student_scored.logits.float(), dim=-1).detach()
                p = student_logp.exp()
                response_mask = student_scored.response_mask
                del student_scored

            with ema.swap_into(model):
                full_logq, full_mask = _target_log_probs_batch(
                    model, tokenizer, full_teacher_prompts, response_ids, max_context_tokens
                )
                response_mask = full_mask if response_mask is None else response_mask
                if not torch.equal(response_mask, full_mask):
                    raise RuntimeError("Full-context target mask disagrees with student response mask.")

                if need_all_candidates:
                    assert p is not None and full_logq is not None and response_mask is not None
                    best_scores = torch.full((batch_size,), float("-inf"), device=full_logq.device)
                    gradient_selected = torch.zeros(batch_size, dtype=torch.long, device=full_logq.device)
                    chunk_blocks = max(1, min(int(candidate_chunk_blocks), block_count))

                    eps = float(fire_cfg.get("eps", 1.0e-8))
                    weighting = str(fire_cfg.get("weighting", "excess_l2_energy"))
                    # Main FIRE: streaming Fisher-energy sufficient statistics.
                    fisher_methods = {"fire", "fire_attribution_only"}
                    fisher_weighted_residual_sum = torch.zeros_like(full_logq) if method in fisher_methods else None
                    fisher_field_energy_sum = (
                        torch.zeros_like(response_mask, dtype=full_logq.dtype) if method in fisher_methods else None
                    )
                    full_disagreement_energy: torch.Tensor | None = None
                    effective_fisher_energy = (
                        _softmax_fisher_energy(p) / (configured_step_scale ** 2)
                    ) * response_mask.to(dtype=p.dtype)
                    if method in fisher_methods and weighting in {"normalized_disagreement", "excess_l2_energy"}:
                        assert student_logp is not None
                        g_full = _reverse_kl_logit_gradient(p, student_logp, full_logq)
                        full_disagreement_energy = g_full.square().sum(dim=-1)

                    for block_start in range(0, block_count, chunk_blocks):
                        indices = list(range(block_start, min(block_start + chunk_blocks, block_count)))
                        flat_prompts: list[str] = []
                        flat_responses: list[torch.Tensor] = []
                        for index in indices:
                            flat_prompts.extend(
                                render_chat(
                                    tokenizer,
                                    teacher_messages(
                                        example,
                                        replace_block(context.blocks, index, REMOVAL_MARKER),
                                        context.block_fields,
                                        context_type=context.context_type,
                                    ),
                                )
                                for example, context in zip(examples, contexts)
                            )
                            flat_responses.extend(response_ids)
                        flat_logq, flat_mask = _target_log_probs_batch(
                            model, tokenizer, flat_prompts, flat_responses, max_context_tokens
                        )

                        for local, index in enumerate(indices):
                            start = local * batch_size
                            stop = start + batch_size
                            candidate_logq = flat_logq[start:stop]
                            candidate_mask = flat_mask[start:stop]
                            if not torch.equal(response_mask, candidate_mask):
                                raise RuntimeError("Leave-one-out target mask disagrees with student response mask.")
                            token_scores = _igt_token_scores_batch(p, full_logq, candidate_logq)
                            mask_f = response_mask.to(dtype=token_scores.dtype)
                            scores = (token_scores * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp_min(1.0)
                            score_columns.append(scores)
                            better = scores > best_scores
                            best_scores = torch.where(better, scores, best_scores)
                            gradient_selected = torch.where(
                                better, torch.full_like(gradient_selected, index), gradient_selected
                            )

                            if method == "hard_gradient_excision":
                                if best_target is None:
                                    best_target = candidate_logq.clone()
                                else:
                                    best_target = torch.where(better[:, None, None], candidate_logq, best_target)

                            if method in fisher_methods:
                                assert (
                                    fisher_weighted_residual_sum is not None
                                    and fisher_field_energy_sum is not None
                                )
                                _, field_energy = _l2_gradient_contrast_batch(p, full_logq, candidate_logq)
                                field_energy = field_energy * response_mask.to(dtype=field_energy.dtype)
                                field_energy_columns.append(field_energy)
                                field_evidence = _field_evidence_from_contrast(
                                    weighting=weighting,
                                    field_energy=field_energy,
                                    half_l1=token_scores,
                                    full_disagreement_energy=full_disagreement_energy,
                                    effective_fisher_energy=effective_fisher_energy,
                                    response_mask=response_mask,
                                    eps=eps,
                                )
                                field_evidence_columns.append(field_evidence)
                                fisher_weighted_residual_sum.add_(
                                    field_evidence[:, :, None] * (candidate_logq - full_logq)
                                )
                                fisher_field_energy_sum.add_(field_evidence)

                        del flat_logq, flat_mask

                    assert best_scores is not None and gradient_selected is not None
                    if method in fisher_methods:
                        assert student_logp is not None
                        assert (
                            fisher_weighted_residual_sum is not None
                            and fisher_field_energy_sum is not None
                        )
                        if len(field_energy_columns) != block_count or len(field_evidence_columns) != block_count:
                            raise RuntimeError("FIRE did not collect exactly one energy/evidence tensor per context field.")
                        attribution_logq, token_field_mass, _ = _fisher_attribution_target_from_accumulators(
                            p=p,
                            full_logq=full_logq,
                            weighted_residual_sum=fisher_weighted_residual_sum,
                            field_energy_sum=fisher_field_energy_sum,
                            response_mask=response_mask,
                            eps=eps,
                            step_scale=configured_step_scale,
                        )
                        if method == "fire":
                            (
                                target,
                                token_trust,
                                token_before_norm,
                                token_after_norm,
                                token_fisher_radius,
                            ) = _fisher_radial_project_target(
                                p=p,
                                student_logp=student_logp,
                                source_logq=attribution_logq,
                                response_mask=response_mask,
                                eps=eps,
                                step_scale=configured_step_scale,
                            )
                        else:
                            target = attribution_logq
                            mask_f_local = response_mask.to(dtype=full_logq.dtype)
                            g_attr = _reverse_kl_logit_gradient(p, student_logp, attribution_logq)
                            token_before_norm = torch.linalg.vector_norm(g_attr, dim=-1) * mask_f_local
                            token_after_norm = token_before_norm
                            token_fisher_radius = torch.sqrt(
                                (_softmax_fisher_energy(p) * mask_f_local).clamp_min(0.0)
                            ) / configured_step_scale
                            token_trust = torch.ones_like(token_before_norm)
                        field_energies = torch.stack(field_energy_columns, dim=0)
                        field_evidence = torch.stack(field_evidence_columns, dim=0)
                        fisher_energy = (
                            _softmax_fisher_energy(p) / (configured_step_scale ** 2)
                        ) * response_mask.to(dtype=p.dtype)
                        normalizer = (fisher_energy + fisher_field_energy_sum).clamp_min(eps)
                        alphas = field_evidence / normalizer[None, :, :]
                        mask_f = response_mask.to(dtype=full_logq.dtype)
                        denom = mask_f.sum(dim=-1).clamp_min(1.0)
                        projection_weights = (token_field_mass * mask_f).sum(dim=-1) / denom
                        trust_coefficients = (token_trust * mask_f).sum(dim=-1) / denom
                        logit_grad_norm_before = (token_before_norm * mask_f).sum(dim=-1) / denom
                        logit_grad_norm_after = (token_after_norm * mask_f).sum(dim=-1) / denom
                        fisher_radius = (token_fisher_radius * mask_f).sum(dim=-1) / denom
                        max_field_weight = (alphas.amax(dim=0) * mask_f).sum(dim=-1) / denom
                        field_energy_sum_actual = field_energies.sum(dim=0)
                        energy_sq_sum = field_energies.square().sum(dim=0)
                        participation = torch.where(
                            field_energy_sum_actual > eps,
                            field_energy_sum_actual.square() / energy_sq_sum.clamp_min(eps),
                            torch.zeros_like(field_energy_sum_actual),
                        )
                        active_fields = (participation * mask_f).sum(dim=-1) / denom
                        retained_influence = logit_grad_norm_after
                        method_selected = [None] * batch_size
                    elif method == "hard_gradient_excision":
                        assert best_target is not None
                        target = best_target
                        projection_weights = torch.ones_like(best_scores)
                        retained_influence = torch.zeros_like(best_scores)
                        active_fields = torch.ones_like(best_scores)
                        method_selected = [int(value) for value in gradient_selected.tolist()]
                    elif method == "full_context":
                        target = full_logq
                    else:  # pragma: no cover
                        raise AssertionError(method)

                else:
                    if method == "full_context":
                        assert full_logq is not None
                        target = full_logq
                    elif method == "fisher_trust_full_context":
                        assert p is not None and student_logp is not None and full_logq is not None and response_mask is not None
                        (
                            target,
                            token_trust,
                            token_before_norm,
                            token_after_norm,
                            token_fisher_radius,
                        ) = _fisher_radial_project_target(
                            p=p,
                            student_logp=student_logp,
                            source_logq=full_logq,
                            response_mask=response_mask,
                            eps=float(fire_cfg.get("eps", 1.0e-8)),
                            step_scale=configured_step_scale,
                        )
                        mask_f = response_mask.to(dtype=full_logq.dtype)
                        denom = mask_f.sum(dim=-1).clamp_min(1.0)
                        trust_coefficients = (token_trust * mask_f).sum(dim=-1) / denom
                        logit_grad_norm_before = (token_before_norm * mask_f).sum(dim=-1) / denom
                        logit_grad_norm_after = (token_after_norm * mask_f).sum(dim=-1) / denom
                        fisher_radius = (token_fisher_radius * mask_f).sum(dim=-1) / denom
                    else:  # pragma: no cover
                        raise AssertionError(method)
    finally:
        if model_was_training:
            model.train()

    if target is None or response_mask is None:
        raise RuntimeError("Routing failed to construct a target and response mask.")

    diagnostics: list[RoutingDiagnostics] = []
    if score_columns:
        score_matrix = torch.stack(score_columns, dim=1).detach().cpu()
        assert gradient_selected is not None
        gradient_selected_values = [int(value) for value in gradient_selected.detach().cpu().tolist()]
        projection_values = (
            [float(value) for value in projection_weights.detach().cpu().tolist()]
            if projection_weights is not None
            else [None] * batch_size
        )
        retained_values = (
            [float(value) for value in retained_influence.detach().cpu().tolist()]
            if retained_influence is not None
            else [None] * batch_size
        )
        active_field_values = (
            [float(value) for value in active_fields.detach().cpu().tolist()]
            if active_fields is not None
            else [None] * batch_size
        )
        trust_values = (
            [float(value) for value in trust_coefficients.detach().cpu().tolist()]
            if trust_coefficients is not None
            else [None] * batch_size
        )
        before_norm_values = (
            [float(value) for value in logit_grad_norm_before.detach().cpu().tolist()]
            if logit_grad_norm_before is not None
            else [None] * batch_size
        )
        after_norm_values = (
            [float(value) for value in logit_grad_norm_after.detach().cpu().tolist()]
            if logit_grad_norm_after is not None
            else [None] * batch_size
        )
        fisher_radius_values = (
            [float(value) for value in fisher_radius.detach().cpu().tolist()]
            if fisher_radius is not None
            else [None] * batch_size
        )
        max_field_weight_values = (
            [float(value) for value in max_field_weight.detach().cpu().tolist()]
            if max_field_weight is not None
            else [None] * batch_size
        )
        for row in range(batch_size):
            scores = [float(value) for value in score_matrix[row].tolist()]
            sorted_scores = sorted(scores, reverse=True)
            maximum = sorted_scores[0]
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            chosen = method_selected[row]
            selected_score = None if chosen is None else scores[int(chosen)]
            diagnostics.append(
                RoutingDiagnostics(
                    scores=scores,
                    gradient_selected=gradient_selected_values[row],
                    method_selected=chosen,
                    selected_score=selected_score,
                    max_score=maximum,
                    mean_score=float(sum(scores) / len(scores)),
                    score_margin=float(margin),
                    selection_entropy=_softmax_entropy(scores),
                    projection_weight=projection_values[row],
                    retained_influence=retained_values[row],
                    active_fields=active_field_values[row],
                    trust_coefficient=trust_values[row],
                    logit_grad_norm_before=before_norm_values[row],
                    logit_grad_norm_after=after_norm_values[row],
                    fisher_radius=fisher_radius_values[row],
                    max_field_weight=max_field_weight_values[row],
                    step_scale=configured_step_scale,
                )
            )
    else:
        trust_values = (
            [float(value) for value in trust_coefficients.detach().cpu().tolist()]
            if trust_coefficients is not None
            else [None] * batch_size
        )
        before_norm_values = (
            [float(value) for value in logit_grad_norm_before.detach().cpu().tolist()]
            if logit_grad_norm_before is not None
            else [None] * batch_size
        )
        after_norm_values = (
            [float(value) for value in logit_grad_norm_after.detach().cpu().tolist()]
            if logit_grad_norm_after is not None
            else [None] * batch_size
        )
        fisher_radius_values = (
            [float(value) for value in fisher_radius.detach().cpu().tolist()]
            if fisher_radius is not None
            else [None] * batch_size
        )
        for row in range(batch_size):
            chosen = method_selected[row]
            diagnostics.append(
                RoutingDiagnostics(
                    scores=[],
                    gradient_selected=-1,
                    method_selected=chosen,
                    selected_score=None,
                    max_score=0.0,
                    mean_score=0.0,
                    score_margin=0.0,
                    selection_entropy=0.0,
                    trust_coefficient=trust_values[row],
                    logit_grad_norm_before=before_norm_values[row],
                    logit_grad_norm_after=after_norm_values[row],
                    fisher_radius=fisher_radius_values[row],
                )
            )

    return BatchRouteResult(
        target_log_probs=target.detach(),
        response_mask=response_mask.detach(),
        diagnostics=diagnostics,
    )
