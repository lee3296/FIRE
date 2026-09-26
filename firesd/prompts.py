"""Prompt and context formatting for stable on-policy self-distillation."""
from __future__ import annotations

from typing import Sequence

from .data import TaskExample


# All leave-one-block-out methods use exactly the same in-place deletion marker.
# Keeping the empty slot avoids a rendering mismatch between gradient excision and
# the random / mean leave-one-out baselines.
REMOVAL_MARKER = ""


def answer_format_instruction(example: TaskExample) -> str:
    if example.task_type == "math":
        return (
            "Work through the problem step by step, showing your arithmetic. "
            "Then, on a new final line, write the final answer as `#### ` followed "
            "by the number only (for example `#### 42`). Write nothing after that line."
        )
    if example.task_type == "multiple_choice":
        return (
            "Reason step by step about the options. Then, on a new final line, write "
            "`Answer: ` followed by the single option letter only (for example "
            "`Answer: B`). Write nothing after that line."
        )
    return "Reason step by step. Then, on a new final line, write your final answer only."


def student_messages(example: TaskExample) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Solve the following task.\n\n"
                f"{answer_format_instruction(example)}\n\n"
                f"{example.prompt}"
            ),
        }
    ]


def _render_blocks(blocks: Sequence[str]) -> str:
    rendered = "\n".join(f"[Context block {index + 1}]\n{block}" for index, block in enumerate(blocks))
    return rendered if rendered else "[No teacher context was provided.]"


def teacher_messages(
    example: TaskExample,
    blocks: Sequence[str],
    block_fields: Sequence[str] | None = None,
    *,
    context_type: str = "sdpo_feedback",
) -> list[dict[str, str]]:
    if block_fields is not None and len(block_fields) != len(blocks):
        raise ValueError("block_fields must be omitted or contain one entry per context block.")
    if context_type != "sdpo_feedback":
        raise ValueError(f"Unknown context_type={context_type!r}")
    rendered = _render_blocks(blocks)
    instruction = (
        "You are the feedback-conditioned teacher in on-policy self-distillation. The earlier "
        "student attempt will appear as the assistant continuation. Use the structured feedback "
        "context to revise the continuation distribution for the same task. Preserve useful "
        "reasoning tokens and end with the task's normal final-answer line. Do not quote or "
        "describe the context.\n\n"
        "Task:\n"
        f"{example.prompt}\n\n"
        "Structured feedback context:\n"
        f"{rendered}"
    )
    return [{"role": "user", "content": instruction}]


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    """Use the checkpoint's chat template when available, with a safe text fallback."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **getattr(tokenizer, "_firesd_chat_template_kwargs", {}),
        )
    chunks: list[str] = []
    for message in messages:
        chunks.append(f"{message['role'].title()}: {message['content']}")
    chunks.append("Assistant:")
    return "\n\n".join(chunks)
