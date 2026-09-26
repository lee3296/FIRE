"""Batched evaluation for task utility and routing stability diagnostics."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterator, Sequence, TypeVar

import torch

from .contexts import ContextRecord, TeacherContextBuilder
from .data import DatasetAdapter, TaskExample
from .ema import LoRAEMA
from .metrics import RoutingMeter
from .modeling import GeneratedRollout, SequenceBudgetError, generate_rollout_batch, generation_budget_errors
from .prompts import render_chat, student_messages
from .routing import normalize_method, route_targets_batch, routing_budget_errors


@dataclass(frozen=True)
class _RoutingProbeRequest:
    example: TaskExample
    response_ids: torch.Tensor
    context: ContextRecord


_T = TypeVar("_T")


def _batched(items: Sequence[_T], batch_size: int) -> Iterator[Sequence[_T]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    spread = z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _generate_task_rollouts(
    *,
    model,
    tokenizer,
    examples: Sequence[TaskExample],
    generation_cfg: dict,
    sequence_cfg: dict,
    batch_size: int,
) -> tuple[list[tuple[TaskExample, GeneratedRollout]], int]:
    generated_items: list[tuple[TaskExample, GeneratedRollout]] = []
    skipped_budget = 0
    for example_chunk in _batched(list(examples), batch_size):
        prompts = [render_chat(tokenizer, student_messages(example)) for example in example_chunk]
        errors = generation_budget_errors(
            tokenizer,
            prompts,
            max_new_tokens=int(generation_cfg["max_new_tokens"]),
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
        )
        valid = [(example, prompt) for example, prompt, error in zip(example_chunk, prompts, errors) if error is None]
        skipped_budget += sum(error is not None for error in errors)
        if not valid:
            continue
        try:
            rollouts = generate_rollout_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt for _, prompt in valid],
                generation_cfg=generation_cfg,
                sequence_cfg=sequence_cfg,
                do_sample=False,
            )
        except SequenceBudgetError:
            skipped_budget += len(valid)
            continue
        generated_items.extend((example, rollout) for (example, _), rollout in zip(valid, rollouts))
    return generated_items, skipped_budget


def evaluate(
    *,
    cfg: dict[str, Any],
    model,
    tokenizer,
    ema: LoRAEMA,
    adapter: DatasetAdapter,
    examples: list[TaskExample],
    context_builder: TeacherContextBuilder,
    step: int,
) -> dict[str, Any]:
    eval_cfg = cfg["eval"]
    generation_cfg = cfg["generation"]
    sequence_cfg = cfg["sequence"]
    task_limit = min(int(eval_cfg["task_examples"]), len(examples))
    routing_limit = min(int(eval_cfg["routing_examples"]), task_limit)
    eval_batch_size = int(eval_cfg.get("batch_size", 1))

    clean_correct = 0
    evaluated = 0
    category_correct: dict[str, int] = {}
    category_total: dict[str, int] = {}
    routing_probes = 0
    routing = RoutingMeter()
    skipped_budget = 0
    rng = random.Random(int(cfg["train"]["seed"]) + 1_000_003 + step)

    task_items, task_skipped = _generate_task_rollouts(
        model=model,
        tokenizer=tokenizer,
        examples=examples[:task_limit],
        generation_cfg=generation_cfg,
        sequence_cfg=sequence_cfg,
        batch_size=eval_batch_size,
    )
    skipped_budget += task_skipped

    routing_candidates: list[_RoutingProbeRequest] = []
    method = str(cfg["experiment"]["method"])
    route_method = normalize_method(method)
    probe_methods = {
        "fire",
        "fire_attribution_only",
        "fisher_trust_full_context",
        "hard_gradient_excision",
        "full_context",
    }
    can_probe_routing = route_method in probe_methods

    truncated_rollouts = 0
    missing_final_answer = 0
    response_token_total = 0

    for example, rollout in task_items:
        evaluated += 1
        # Budget health of the held-out rollouts.  A high truncation rate means
        # the benchmark score is measuring generation.max_new_tokens as much as
        # it is measuring the model.
        truncated_rollouts += int(not bool(getattr(rollout, "finished", True)))
        missing_final_answer += int(not bool(adapter.has_final_answer(rollout.response_text, example)))
        response_token_total += int(rollout.response_ids.numel())
        is_correct = int(adapter.is_correct(rollout.response_text, example))
        clean_correct += is_correct
        category = str(getattr(example, "category", "")).strip()
        if category:
            category_total[category] = category_total.get(category, 0) + 1
            category_correct[category] = category_correct.get(category, 0) + is_correct
        if not can_probe_routing:
            continue
        context = context_builder.build(example, rollout.response_text, adapter, rng=rng)
        if len(routing_candidates) >= routing_limit:
            continue
        route_error = routing_budget_errors(
            tokenizer=tokenizer,
            examples=[example],
            response_ids=[rollout.response_ids],
            contexts=[context],
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
        )[0]
        if route_error is not None:
            skipped_budget += 1
            continue
        routing_candidates.append(_RoutingProbeRequest(example, rollout.response_ids, context))

    for request_chunk in _batched(routing_candidates[:routing_limit], eval_batch_size):
        try:
            routed = route_targets_batch(
                model=model,
                tokenizer=tokenizer,
                ema=ema,
                examples=[request.example for request in request_chunk],
                response_ids=[request.response_ids for request in request_chunk],
                contexts=[request.context for request in request_chunk],
                method=route_method,
                max_context_tokens=int(sequence_cfg["max_context_tokens"]),
                fire_cfg=cfg.get("fire", {}),
                candidate_chunk_blocks=int(cfg.get("routing", {}).get("candidate_chunk_blocks", 1)),
                collect_diagnostics=True,
            )
        except SequenceBudgetError:
            skipped_budget += len(request_chunk)
            continue
        for diagnostics in routed.diagnostics:
            routing.update(diagnostics)
        routing_probes += len(request_chunk)
        del routed

    task_denom = max(evaluated, 1)
    task_accuracy = clean_correct / task_denom
    ci_low, ci_high = _wilson_interval(clean_correct, evaluated)
    result: dict[str, Any] = {
        "step": step,
        "eval_batch_size": eval_batch_size,
        "context_type": context_builder.context_type,
        "context_order_mode": context_builder.order_mode,
        "native_field_order": list(context_builder.field_names),
        "task_accuracy": task_accuracy,
        "task_error_rate": 1.0 - task_accuracy,
        "task_accuracy_se": math.sqrt(task_accuracy * (1.0 - task_accuracy) / task_denom),
        "task_accuracy_wilson_low": ci_low,
        "task_accuracy_wilson_high": ci_high,
        "task_examples": evaluated,
        "routing_probe_examples": routing_probes,
        "skipped_over_budget": skipped_budget,
        "rollout_truncated_rate": truncated_rollouts / task_denom,
        "rollout_missing_final_answer_rate": missing_final_answer / task_denom,
        "rollout_mean_response_tokens": response_token_total / task_denom,
        "rollout_length_saturation": (
            (response_token_total / task_denom) / max(int(generation_cfg["max_new_tokens"]), 1)
        ),
    }
    if category_total:
        # The benchmark-wide score remains `task_accuracy`; these are diagnostic
        # slices only and do not change training, model selection, or aggregation.
        result["task_accuracy_by_category"] = {
            category: category_correct.get(category, 0) / total
            for category, total in sorted(category_total.items())
        }
        result["task_examples_by_category"] = dict(sorted(category_total.items()))
        result["task_correct_by_category"] = {
            category: category_correct.get(category, 0)
            for category in sorted(category_total)
        }
    result.update(routing.summary("routing"))
    return result
