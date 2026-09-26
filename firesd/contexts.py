"""Teacher-context construction for SDPO-style feedback."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .data import DatasetAdapter, TaskExample, stable_example_seed


@dataclass(frozen=True)
class ContextRecord:
    blocks: tuple[str, ...]
    context_type: str
    block_order: tuple[int, ...]
    block_fields: tuple[str, ...]


_SDPO_FIELDS = ["verifier_result", "parser_record", "verifier_provenance", "response_format"]


def _safe_inline(text: str, max_chars: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _numeric_tokens(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _bool_flag(value: bool) -> str:
    return "yes" if value else "no"


def _task_final_marker(example: TaskExample) -> str:
    return "####" if example.task_type == "math" else "Answer:"


def _expected_answer_type(example: TaskExample) -> str:
    if example.task_type == "math":
        return "numeric_final_answer"
    if example.task_type == "multiple_choice":
        return "single_choice_label"
    return "free_form"


def _rollout_stats(text: str) -> dict[str, int]:
    return {
        "chars": len(text),
        "lines": len([line for line in text.splitlines() if line.strip()]),
        "numbers": len(_numeric_tokens(text)),
        "arith_exprs": len(re.findall(r"[-+]?\d+(?:\.\d+)?\s*[+\-*/]\s*[-+]?\d+(?:\.\d+)?", text)),
    }


def _arithmetic_expressions(text: str, limit: int = 3) -> list[str]:
    pattern = r"[-+]?\d+(?:\.\d+)?\s*[+\-*/]\s*[-+]?\d+(?:\.\d+)?(?:\s*=\s*[-+]?\d+(?:\.\d+)?)?"
    return [_safe_inline(match, 60) for match in re.findall(pattern, text)[:limit]]


def _format_issues(example: TaskExample, rollout_text: str) -> list[str]:
    marker = _task_final_marker(example)
    issues: list[str] = []
    if marker not in rollout_text:
        issues.append("missing_final_marker")
    if rollout_text.count(marker) > 1:
        issues.append("multiple_final_markers")
    if not _last_nonempty_line(rollout_text).startswith(marker):
        issues.append("final_line_not_canonical")
    return issues or ["none"]


def replace_block(blocks: Sequence[str], index: int, marker: str) -> tuple[str, ...]:
    copied = list(blocks)
    copied[index] = marker
    return tuple(copied)


class TeacherContextBuilder:
    """Build block-structured SDPO teacher contexts."""

    def __init__(self, context_cfg: dict[str, Any], *, seed: int = 0):
        self.context_cfg = dict(context_cfg)
        self.context_type = str(self.context_cfg.get("type", "sdpo_feedback"))
        if self.context_type != "sdpo_feedback":
            raise ValueError("context.type must be 'sdpo_feedback'")
        self.order_mode = str(self.context_cfg.get("order_mode", "native"))
        if self.order_mode not in {"native", "shuffle"}:
            raise ValueError("context.order_mode must be 'native' or 'shuffle'")
        self.seed = int(seed)
        default_fields = _SDPO_FIELDS
        num_blocks = int(self.context_cfg.get("num_blocks", len(default_fields)))
        self.field_names = list(self.context_cfg.get("native_field_order", default_fields[:num_blocks]))
        if len(self.field_names) != num_blocks or set(self.field_names) != set(default_fields[:num_blocks]):
            raise ValueError(
                "context.native_field_order must be a permutation of active fields: "
                f"expected {default_fields[:num_blocks]}, got {self.field_names}"
            )

    @property
    def num_blocks(self) -> int:
        return len(self.field_names)


    @staticmethod
    def fields_for(context_type: str, num_blocks: int) -> list[str]:
        if context_type != "sdpo_feedback":
            raise ValueError("context_type must be 'sdpo_feedback'")
        return _SDPO_FIELDS[: int(num_blocks)]

    @staticmethod
    def _validate_order(order: Sequence[int], block_count: int) -> tuple[int, ...]:
        normalized = tuple(int(item) for item in order)
        if sorted(normalized) != list(range(block_count)):
            raise ValueError(f"block_order must be a permutation of 0..{block_count - 1}, got {normalized}")
        return normalized

    def _native_order(self, canonical_fields: Sequence[str]) -> tuple[int, ...]:
        return tuple(canonical_fields.index(name) for name in self.field_names)

    @staticmethod
    def _shuffle_order(block_count: int, rng: random.Random | None) -> tuple[int, ...]:
        generator = rng if rng is not None else random
        order = list(range(block_count))
        generator.shuffle(order)
        return tuple(order)

    def _select_order(self, block_count: int, canonical_fields: Sequence[str], rng: random.Random | None) -> tuple[int, ...]:
        return self._native_order(canonical_fields) if self.order_mode == "native" else self._shuffle_order(block_count, rng)

    def _render_verifier_block(self, *, correct: bool, rendered_reference: str, parsed_answer: str) -> str:
        if correct:
            return (
                "Verifier feedback: source=reference_checker; status=correct; decision=final; "
                f"accepted final answer=`{rendered_reference}`; submitted normalized answer=`{parsed_answer}`; "
                "action=retain submitted final answer."
            )
        return (
            "Verifier feedback: source=reference_checker; status=incorrect; decision=final; "
            f"expected final answer=`{rendered_reference}`; submitted normalized answer=`{parsed_answer}`; "
            f"replacement final line=`{rendered_reference}`; "
            "action=revise only the final answer while preserving task-normal format."
        )

    def _render_parser_block(self, example: TaskExample, rollout_text: str, adapter: DatasetAdapter, parsed: str) -> str:
        last_line = _last_nonempty_line(rollout_text)
        marker = _task_final_marker(example)
        marker_count = rollout_text.count(marker)
        numbers = _numeric_tokens(last_line if example.task_type == "math" else rollout_text)
        numeric_tail = ", ".join(numbers[-4:]) if numbers else "none"
        parse_ok = parsed not in {"", "<no parse>"}
        return (
            "Parser diagnostics: source=response_parser; "
            f"task uid=`{example.uid}`; task type={example.task_type}; expected answer type={_expected_answer_type(example)}; "
            f"final marker=`{marker}`; marker count={marker_count}; parse success={_bool_flag(parse_ok)}; "
            f"extracted normalized final answer=`{parsed}`; final line snapshot=`{_safe_inline(last_line)}`; "
            f"numeric tokens near parsed span={numeric_tail}."
        )

    def _render_provenance_block(self, example: TaskExample, rollout_text: str, adapter: DatasetAdapter, parsed: str) -> str:
        stats = _rollout_stats(rollout_text)
        fingerprint = f"{stable_example_seed('context', adapter.name, example.uid) & 0xFFFF:04x}"
        prompt_numbers = _numeric_tokens(example.prompt)
        prompt_number_summary = ", ".join(prompt_numbers[:6]) if prompt_numbers else "none"
        choice_summary = ",".join(str(label) for label in example.choices) if example.choices else "n/a"
        return (
            "Context provenance: source=environment_audit; "
            f"dataset adapter={adapter.name}; task fingerprint={fingerprint}; task type={example.task_type}; "
            f"choice labels={choice_summary}; prompt numeric cues={prompt_number_summary}; "
            f"checker mode=deterministic_exact_match; "
            "normalization=task_adapter_final_answer; "
            f"submitted normalized answer=`{parsed}`; response chars={stats['chars']}; "
            f"response nonempty lines={stats['lines']}; response numeric-token count={stats['numbers']}."
        )

    def _render_response_format_block(self, example: TaskExample, rollout_text: str, adapter: DatasetAdapter, parsed: str) -> str:
        marker = _task_final_marker(example)
        stats = _rollout_stats(rollout_text)
        exprs = _arithmetic_expressions(rollout_text)
        expr_summary = " | ".join(exprs) if exprs else "none detected"
        issues = ",".join(_format_issues(example, rollout_text))
        has_reasoning = len([line for line in rollout_text.splitlines() if line.strip()]) > 1
        return (
            "Response-format diagnostics: source=format_checker; "
            f"required final marker=`{marker}`; final-line parse result=`{parsed}`; format issues={issues}; "
            f"reasoning text before final line={_bool_flag(has_reasoning)}; arithmetic expression count={stats['arith_exprs']}; "
            f"arithmetic expression snippets=`{expr_summary}`; "
            "instruction=end with exactly one task-normal final-answer line and no text after it."
        )

    def _sdpo_blocks(self, example: TaskExample, rollout_text: str, adapter: DatasetAdapter) -> tuple[list[str], list[str]]:
        parsed = adapter.extract_prediction(rollout_text, example) or "<no parse>"
        correct = adapter.is_correct(rollout_text, example)
        reference = adapter.render_answer(adapter.normalize_gold(example.answer), example)
        blocks = [
            self._render_verifier_block(correct=correct, rendered_reference=reference, parsed_answer=parsed),
            self._render_parser_block(example, rollout_text, adapter, parsed),
            self._render_provenance_block(example, rollout_text, adapter, parsed),
            self._render_response_format_block(example, rollout_text, adapter, parsed),
        ]
        return blocks[: self.num_blocks], _SDPO_FIELDS[: self.num_blocks]

    def build(
        self,
        example: TaskExample,
        rollout_text: str,
        adapter: DatasetAdapter,
        *,
        rng: random.Random | None = None,
        block_order: Sequence[int] | None = None,
    ) -> ContextRecord:
        canonical, canonical_fields = self._sdpo_blocks(example, rollout_text, adapter)
        if len(canonical) != self.num_blocks:
            raise ValueError("Constructed context block count does not match configuration.")
        order = self._select_order(len(canonical), canonical_fields, rng) if block_order is None else self._validate_order(block_order, len(canonical))
        blocks = tuple(canonical[index] for index in order)
        fields = tuple(canonical_fields[index] for index in order)
        return ContextRecord(blocks=blocks, context_type=self.context_type, block_order=order, block_fields=fields)
