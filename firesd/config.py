"""YAML composition and validation for stable self-distillation experiments."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


class ConfigError(ValueError):
    """Raised for invalid experiment configuration."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_one(path: Path, seen: set[Path]) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in list(seen) + [path])
        raise ConfigError(f"Cyclic _base_ include: {chain}")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    seen = set(seen)
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML must be a mapping: {path}")

    bases = payload.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigError(f"_base_ must be a string or a list in {path}")

    merged: Dict[str, Any] = {}
    for base in bases:
        if not isinstance(base, str):
            raise ConfigError(f"Non-string _base_ entry in {path}")
        merged = deep_merge(merged, _load_one(path.parent / base, seen))
    return deep_merge(merged, payload)


def load_config(path: str | Path) -> Dict[str, Any]:
    return _load_one(Path(path), set())


def parse_override(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid --set value {value!r}: {exc}") from exc


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    output = copy.deepcopy(config)
    for raw in overrides:
        if "=" not in raw:
            raise ConfigError(f"Override must use PATH=VALUE, received: {raw!r}")
        dotted, raw_value = raw.split("=", 1)
        if not dotted or dotted.startswith(".") or dotted.endswith("."):
            raise ConfigError(f"Invalid dotted path: {dotted!r}")
        cursor: Dict[str, Any] = output
        pieces = dotted.split(".")
        for key in pieces[:-1]:
            existing = cursor.get(key)
            if existing is None:
                cursor[key] = {}
                existing = cursor[key]
            if not isinstance(existing, dict):
                raise ConfigError(f"Cannot set {dotted!r}: {key!r} is not a mapping in this config")
            cursor = existing
        cursor[pieces[-1]] = parse_override(raw_value)
    return output


def _active_context_fields(context_type: str, block_count: int) -> list[str]:
    fields = ["verifier_result", "parser_record", "verifier_provenance", "response_format"]
    if context_type != "sdpo_feedback":
        raise ConfigError("context.type must be 'sdpo_feedback'")
    return fields[:block_count]


def validate_config(cfg: Mapping[str, Any]) -> None:
    required_top = {"experiment", "model", "data", "train", "context", "loss", "eval"}
    missing = sorted(required_top - set(cfg))
    if missing:
        raise ConfigError(f"Missing top-level configuration sections: {missing}")

    train = cfg["train"]
    if int(train["micro_batch_size"]) < 1:
        raise ConfigError("train.micro_batch_size must be at least 1")
    if int(train["gradient_accumulation_steps"]) < 1:
        raise ConfigError("train.gradient_accumulation_steps must be at least 1")
    if int(train["max_steps"]) < 1:
        raise ConfigError("train.max_steps must be at least 1")

    sequence = cfg["sequence"]
    if int(sequence["max_new_tokens"]) < 1:
        raise ConfigError("sequence.max_new_tokens must be at least 1")
    if int(sequence["max_context_tokens"]) < int(sequence["max_new_tokens"]):
        raise ConfigError("sequence.max_context_tokens must exceed sequence.max_new_tokens")

    context = cfg["context"]
    context_type = str(context.get("type", "sdpo_feedback"))
    block_count = int(context.get("num_blocks", 4))
    if block_count < 1 or block_count > 4:
        raise ConfigError("context.num_blocks must lie in [1, 4]")
    order_mode = str(context.get("order_mode", "native"))
    if order_mode not in {"native", "shuffle"}:
        raise ConfigError("context.order_mode must be 'native' or 'shuffle'")
    expected = _active_context_fields(context_type, block_count)
    observed = [str(item) for item in context.get("native_field_order", expected)]
    if len(observed) != block_count or set(observed) != set(expected):
        raise ConfigError(
            "context.native_field_order must be a permutation of the active fields: "
            f"expected {expected}, got {observed}"
        )

    method = str(cfg["experiment"]["method"])
    valid_methods = {
        "fire",
        "fire_no_projection",
        "fire_no_attribution",
        "hard_gradient_excision",
        "full_context",
        "on_policy_sft",
        "veto",
        "topd",
        "tropd",
        "srpo",
        "demopsd",
        "scope",
    }
    if method not in valid_methods:
        raise ConfigError(f"Unknown method {method!r}; expected one of {sorted(valid_methods)}")

    if cfg["loss"]["mode"] not in {"full", "topk_union"}:
        raise ConfigError("loss.mode must be 'full' or 'topk_union'")
    eval_cfg = cfg["eval"]
    if int(eval_cfg.get("batch_size", 1)) < 1:
        raise ConfigError("eval.batch_size must be at least 1")
    for key in ("task_examples", "routing_examples"):
        if int(eval_cfg[key]) < 1:
            raise ConfigError(f"eval.{key} must be at least 1")
    if float(cfg["ema"]["alpha"]) <= 0 or float(cfg["ema"]["alpha"]) > 1:
        raise ConfigError("ema.alpha must lie in (0, 1]")

    routing = cfg.get("routing", {})
    if int(routing.get("candidate_chunk_blocks", 1)) < 1:
        raise ConfigError("routing.candidate_chunk_blocks must be at least 1")

    fire = cfg.get("fire", {})
    if float(fire.get("eps", 1.0e-8)) <= 0.0:
        raise ConfigError("fire.eps must be positive")
    weighting = str(fire.get("weighting", "excess_l2_energy"))
    valid_weightings = {
        "excess_l2_energy", "l2_energy", "uniform", "half_l1", "normalized_disagreement"
    }
    if weighting not in valid_weightings:
        raise ConfigError(
            f"fire.weighting must be one of {sorted(valid_weightings)}, got {weighting!r}"
        )

    for key in ("step_normalized", "project_correct_sft"):
        value = fire.get(key, False)
        if not isinstance(value, bool):
            raise ConfigError(f"fire.{key} must be a boolean")

    generation = cfg.get("generation", {})
    if not isinstance(generation.get("drop_unfinished_rollouts", False), bool):
        raise ConfigError("generation.drop_unfinished_rollouts must be a boolean")

    nominal_lr = float(cfg.get("optimizer", {}).get("nominal_lr", cfg["optimizer"]["lr"]))
    if nominal_lr <= 0.0:
        raise ConfigError("optimizer.nominal_lr must be positive")

    correctness_routing = cfg.get("correctness_routing", {})
    for key in ("enabled", "normalize_over_eligible"):
        value = correctness_routing.get(key, False if key == "enabled" else True)
        if not isinstance(value, bool):
            raise ConfigError(f"correctness_routing.{key} must be a boolean")
    correct_action = str(correctness_routing.get("correct_action", "skip"))
    incorrect_action = str(correctness_routing.get("incorrect_action", "distill"))
    if correct_action not in {"distill", "skip", "sft"}:
        raise ConfigError(
            "correctness_routing.correct_action must be 'distill', 'skip', or 'sft'"
        )
    if incorrect_action not in {"distill", "skip"}:
        raise ConfigError(
            "correctness_routing.incorrect_action must be 'distill' or 'skip'"
        )

    on_policy_sft = cfg.get("on_policy_sft", {})
    if int(on_policy_sft.get("group_size", 8)) < 1:
        raise ConfigError("on_policy_sft.group_size must be at least 1")
    if int(on_policy_sft.get("length_limit", cfg["generation"]["max_new_tokens"])) < 1:
        raise ConfigError("on_policy_sft.length_limit must be positive")

    veto = cfg.get("veto", {})
    if float(veto.get("beta_start", 0.8)) < 0 or float(veto.get("beta_end", 0.0)) < 0:
        raise ConfigError("Veto beta values must be nonnegative")
    if str(veto.get("objective", "forward_kl")) not in {"forward_kl", "reverse_kl"}:
        raise ConfigError("veto.objective must be forward_kl or reverse_kl")

    topd = cfg.get("topd", {})
    if int(topd.get("group_size", 8)) < 2:
        raise ConfigError("topd.group_size must be at least 2")
    if not 0 < float(topd.get("alpha", 0.2)) < 1:
        raise ConfigError("topd.alpha must lie in (0,1)")
    if not 0 < float(topd.get("clip_epsilon", 0.2)) < 1:
        raise ConfigError("topd.clip_epsilon must lie in (0,1)")
    if int(topd.get("score_chunk_size", 2)) < 1:
        raise ConfigError("topd.score_chunk_size must be at least 1")
    if int(topd.get("minibatch_prompts", 1)) < 1:
        raise ConfigError("topd.minibatch_prompts must be at least 1")
    if int(topd.get("off_policy_epochs", 1)) < 1:
        raise ConfigError("topd.off_policy_epochs must be at least 1")
    if not isinstance(topd.get("shuffle_minibatches", True), bool):
        raise ConfigError("topd.shuffle_minibatches must be a boolean")
    effective_batch = int(cfg["train"]["micro_batch_size"]) * int(
        cfg["train"]["gradient_accumulation_steps"]
    )
    if int(topd.get("minibatch_prompts", 1)) > effective_batch:
        raise ConfigError("topd.minibatch_prompts cannot exceed the effective prompt batch")

    demopsd = cfg.get("demopsd", {})
    demopsd_alpha_max = float(demopsd.get("alpha_max", 0.15))
    if not 0.0 <= demopsd_alpha_max <= 1.0:
        raise ConfigError("demopsd.alpha_max must lie in [0, 1]")
    if float(demopsd.get("beta", 50.0)) <= 0.0:
        raise ConfigError("demopsd.beta must be positive")

    scope = cfg.get("scope", {})
    if int(scope.get("group_size", 4)) < 2:
        raise ConfigError("scope.group_size must be at least 2")
    if float(scope.get("weight_temperature", 1.0)) <= 0.0:
        raise ConfigError("scope.weight_temperature must be positive")
    if int(scope.get("score_chunk_size", 2)) < 1:
        raise ConfigError("scope.score_chunk_size must be at least 1")

    tropd = cfg.get("tropd", {})
    if int(tropd.get("group_size", 4)) < 1:
        raise ConfigError("tropd.group_size must be at least 1")
    if int(tropd.get("top_k", 64)) < 1:
        raise ConfigError("tropd.top_k must be at least 1")
    if float(tropd.get("guidance_beta", 0.001)) < 0:
        raise ConfigError("tropd.guidance_beta must be nonnegative")
    if int(tropd.get("score_chunk_size", 1)) < 1:
        raise ConfigError("tropd.score_chunk_size must be at least 1")
    if not isinstance(tropd.get("off_policy_guidance", True), bool):
        raise ConfigError("tropd.off_policy_guidance must be a boolean")


def config_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Composed YAML experiment config.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Dotted override; may be supplied more than once.",
    )
    return parser
