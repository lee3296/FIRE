"""Transformers/PEFT loading plus batched rollout generation and response scoring."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Sequence

import torch


@dataclass
class GeneratedRollout:
    response_ids: torch.Tensor
    response_text: str
    # ``False`` when decoding stopped because it exhausted
    # ``generation.max_new_tokens`` rather than because the model emitted EOS.
    # A budget-truncated rollout has no final-answer line, so the verifier label
    # derived from it reports a formatting accident rather than a reasoning
    # failure.  Defaults to ``True`` so every legacy construction is unchanged.
    finished: bool = True


@dataclass
class BatchedResponseLogits:
    """Response-token logits with an explicit valid-token mask.

    ``logits`` has shape ``[batch, max_response_tokens, vocab]``.  Rows beyond a
    sample's response length are arbitrary model values and must be ignored with
    ``response_mask``.  Keeping a mask rather than truncating allows every model
    forward in a microbatch to remain genuinely vectorized despite variable
    prompt and rollout lengths.
    """

    logits: torch.Tensor
    response_mask: torch.Tensor


class SequenceBudgetError(RuntimeError):
    pass


def resolve_torch_dtype(name: str) -> torch.dtype:
    values = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model.dtype={name!r}; choose one of {sorted(values)}") from exc


def load_tokenizer_and_model(model_cfg: dict, lora_cfg: dict, device: torch.device):
    from .model_compat import configure_transformers_logging

    configure_transformers_logging()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model_id = str(model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=bool(model_cfg.get("trust_remote_code", False)))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_torch_dtype(str(model_cfg.get("dtype", "bfloat16")))
    kwargs = {
        "torch_dtype": dtype if device.type == "cuda" else torch.float32,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    use_4bit = bool(model_cfg.get("load_in_4bit", False)) and device.type == "cuda"
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}
    elif device.type == "cuda":
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not use_4bit:
        model.to(device)
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)))
    elif bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    model.config.use_cache = False
    adapter = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, adapter)
    return tokenizer, model


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the active device for the repository's one-device model setup."""
    return next(parameter for parameter in model.parameters() if parameter.device.type != "meta").device


def _tokenize_one_cpu(tokenizer, prompt: str) -> torch.Tensor:
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Tokenizer must return exactly one prompt row for scalar prompt tokenization.")
    return input_ids[0].detach().to(device="cpu", dtype=torch.long)


def check_generation_budget(prompt_ids: torch.Tensor, max_new_tokens: int, max_context_tokens: int) -> None:
    total = int(prompt_ids.shape[-1]) + int(max_new_tokens)
    if total > int(max_context_tokens):
        raise SequenceBudgetError(
            f"Generation would require {total} tokens but sequence.max_context_tokens="
            f"{max_context_tokens}. The implementation refuses to truncate text."
        )


def check_scoring_budget(prompt_ids: torch.Tensor, response_ids: torch.Tensor, max_context_tokens: int) -> None:
    total = int(prompt_ids.shape[-1]) + int(response_ids.numel())
    if total > int(max_context_tokens):
        raise SequenceBudgetError(
            f"Teacher/student scoring would require {total} tokens but sequence.max_context_tokens="
            f"{max_context_tokens}. The implementation refuses to truncate text."
        )


def generation_budget_errors(
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
    max_context_tokens: int,
) -> list[SequenceBudgetError | None]:
    """Return per-example generation-budget errors without silently trimming any item."""
    errors: list[SequenceBudgetError | None] = []
    for prompt in prompts:
        prompt_ids = _tokenize_one_cpu(tokenizer, prompt).unsqueeze(0)
        try:
            check_generation_budget(prompt_ids, max_new_tokens, max_context_tokens)
        except SequenceBudgetError as exc:
            errors.append(exc)
        else:
            errors.append(None)
    return errors


def scoring_budget_errors(
    tokenizer,
    prompts: Sequence[str],
    response_ids: Sequence[torch.Tensor],
    max_context_tokens: int,
) -> list[SequenceBudgetError | None]:
    """Return per-example scoring-budget errors for a variable-length batch."""
    if len(prompts) != len(response_ids):
        raise ValueError("prompts and response_ids must have the same batch length.")
    errors: list[SequenceBudgetError | None] = []
    for prompt, response in zip(prompts, response_ids):
        prompt_ids = _tokenize_one_cpu(tokenizer, prompt).unsqueeze(0)
        response = response.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        try:
            if response.numel() == 0:
                raise SequenceBudgetError("The sampled rollout is empty; no response tokens can be scored.")
            check_scoring_budget(prompt_ids, response, max_context_tokens)
        except SequenceBudgetError as exc:
            errors.append(exc)
        else:
            errors.append(None)
    return errors


def _left_pad_sequences(
    sequences: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad token sequences and return IDs, attention mask, and true lengths."""
    if not sequences:
        raise ValueError("Cannot create a batch from zero sequences.")
    normalized = [sequence.detach().to(device="cpu", dtype=torch.long).reshape(-1) for sequence in sequences]
    lengths = [int(sequence.numel()) for sequence in normalized]
    if min(lengths) <= 0:
        raise ValueError("Empty prompt/sequence cannot be left-padded for causal-LM scoring.")
    width = max(lengths)
    input_ids = torch.full((len(normalized), width), int(pad_token_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(normalized), width), dtype=torch.long, device=device)
    for row, sequence in enumerate(normalized):
        start = width - sequence.numel()
        input_ids[row, start:] = sequence.to(device)
        attention_mask[row, start:] = 1
    return input_ids, attention_mask, lengths


def _forward_accepts_position_ids(model: torch.nn.Module) -> bool:
    """Whether a model forward can receive explicit RoPE/absolute positions."""
    signature = inspect.signature(model.forward)
    return "position_ids" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def _score_forward(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    *, logits_to_keep: int | None = None,
):
    """Run a padded causal-LM score pass with single-example-equivalent positions.

    Left padding alone is not sufficient for every decoder implementation: some
    architectures otherwise count padding tokens as positions during a direct
    forward call.  Explicit positions make every real token begin at zero, the
    same as its singleton score forward.  The target models accept
    ``position_ids``; minimalist test doubles without that argument keep the
    plain compatible call.
    """
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = logits_to_keep
    if _forward_accepts_position_ids(model):
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        kwargs["position_ids"] = position_ids
    return model(**kwargs)


def _generation_eos_token_id(tokenizer):
    # Existing checkpoints keep their original, tokenizer-based stopping rule.
    return getattr(tokenizer, "_firesd_eos_token_id", tokenizer.eos_token_id)


def _trim_generated_response(
    tokens: torch.Tensor, *, eos_token_id: int | Sequence[int] | None, pad_token_id: int | None
) -> tuple[torch.Tensor, bool]:
    """Match single-example ``generate`` semantics for a padded batch output.

    The first EOS belongs to the generated continuation and is retained, exactly
    as in the old single-example path.  Any trailing pad region after a finished
    row is excluded.  A model normally never emits PAD as a content token.

    Returns ``(response_ids, finished)``.  ``finished`` is ``False`` when neither
    EOS nor a pad region appears, which means the row consumed the whole
    ``max_new_tokens`` budget and was cut off mid-continuation.
    """
    row = tokens.detach().reshape(-1)
    eos_ids = list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else (
        [] if eos_token_id is None else [int(eos_token_id)]
    )
    if eos_ids:
        eos_mask = row.eq(int(eos_ids[0]))
        for token_id in eos_ids[1:]:
            eos_mask = eos_mask | row.eq(int(token_id))
        eos_positions = torch.nonzero(eos_mask, as_tuple=False)
        if eos_positions.numel() > 0:
            return row[: int(eos_positions[0].item()) + 1].clone(), True
    if pad_token_id is not None and pad_token_id not in eos_ids:
        pad_positions = torch.nonzero(row.eq(int(pad_token_id)), as_tuple=False)
        if pad_positions.numel() > 0:
            return row[: int(pad_positions[0].item())].clone(), True
    return row.clone(), False


@torch.no_grad()
def generate_rollout_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    generation_cfg: dict,
    sequence_cfg: dict,
    do_sample: bool,
) -> list[GeneratedRollout]:
    """Generate one continuation per prompt in one padded model ``generate`` call."""
    if not prompts:
        return []
    prompt_tokens = [_tokenize_one_cpu(tokenizer, prompt) for prompt in prompts]
    for prompt_ids in prompt_tokens:
        check_generation_budget(
            prompt_ids.unsqueeze(0),
            int(generation_cfg["max_new_tokens"]),
            int(sequence_cfg["max_context_tokens"]),
        )

    device = model_device(model)
    input_ids, attention_mask, _ = _left_pad_sequences(
        prompt_tokens,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    model_was_training = model.training
    model.eval()
    try:
        generation_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(generation_cfg["max_new_tokens"]),
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": _generation_eos_token_id(tokenizer),
            "use_cache": True,
        }
        if do_sample:
            generation_args.update(
                temperature=float(generation_cfg["temperature"]),
                top_p=float(generation_cfg["top_p"]),
            )
        generated = model.generate(**generation_args)
    finally:
        if model_was_training:
            model.train()

    source_width = input_ids.shape[1]
    rollouts: list[GeneratedRollout] = []
    for row in range(len(prompts)):
        response_ids, finished = _trim_generated_response(
            generated[row, source_width:],
            eos_token_id=_generation_eos_token_id(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        rollouts.append(
            GeneratedRollout(
                response_ids=response_ids,
                response_text=tokenizer.decode(response_ids.detach().cpu(), skip_special_tokens=True),
                finished=finished,
            )
        )
    return rollouts


def response_logits_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    response_ids: Sequence[torch.Tensor],
    max_context_tokens: int,
    requires_grad: bool,
) -> BatchedResponseLogits:
    """Score all response prefixes in one padded causal-LM forward.

    Prompts and sampled responses may differ in length.  Each pair is rebuilt as
    ``prompt || response``, left padded at the batch level, and gathered at the
    causal positions that predict response token ``t``.  No prompt, feedback, or
    continuation is shortened; any over-budget pair raises ``SequenceBudgetError``.
    """
    if len(prompts) != len(response_ids):
        raise ValueError("prompts and response_ids must have the same batch length.")
    if not prompts:
        raise ValueError("Cannot score an empty response batch.")

    prompt_tokens = [_tokenize_one_cpu(tokenizer, prompt) for prompt in prompts]
    response_tokens = [response.detach().to(device="cpu", dtype=torch.long).reshape(-1) for response in response_ids]
    for prompt_ids, response in zip(prompt_tokens, response_tokens):
        if response.numel() == 0:
            raise SequenceBudgetError("The sampled rollout is empty; no response tokens can be scored.")
        check_scoring_budget(prompt_ids.unsqueeze(0), response, max_context_tokens)

    joined = [torch.cat([prompt, response], dim=0) for prompt, response in zip(prompt_tokens, response_tokens)]
    device = model_device(model)
    full_ids, attention_mask, full_lengths = _left_pad_sequences(
        joined,
        pad_token_id=int(tokenizer.pad_token_id),
        device=device,
    )
    batch_size, full_width = full_ids.shape
    response_lengths = torch.tensor([int(response.numel()) for response in response_tokens], device=device)
    prompt_lengths = torch.tensor([int(prompt.numel()) for prompt in prompt_tokens], device=device)
    max_response = int(response_lengths.max().item())
    response_mask = torch.arange(max_response, device=device).unsqueeze(0) < response_lengths.unsqueeze(1)

    left_padding = torch.tensor([full_width - length for length in full_lengths], device=device)
    starts = left_padding + prompt_lengths - 1
    time_offsets = torch.arange(max_response, device=device).unsqueeze(0)
    score_kwargs = {}
    logit_start = 0
    if getattr(model, "_firesd_response_logits_only", False):
        # All rows are left-padded to the same end. Include the token preceding
        # the longest response, which predicts that response's first token.
        logit_start = max(0, int(full_width) - max_response - 1)
        score_kwargs["logits_to_keep"] = int(full_width) - logit_start
    gather_positions = starts.unsqueeze(1) + time_offsets - logit_start
    gather_positions = torch.where(response_mask, gather_positions, torch.zeros_like(gather_positions))

    if requires_grad:
        output = _score_forward(model, full_ids, attention_mask, **score_kwargs)
    else:
        with torch.no_grad():
            output = _score_forward(model, full_ids, attention_mask, **score_kwargs)
    vocab = output.logits.shape[-1]
    logits = torch.gather(
        output.logits,
        dim=1,
        index=gather_positions.unsqueeze(-1).expand(batch_size, max_response, vocab),
    )
    return BatchedResponseLogits(logits=logits, response_mask=response_mask)


@torch.no_grad()
def generate_from_forced_prefix(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    prefix_ids: torch.Tensor,
    generation_cfg: dict,
    sequence_cfg: dict,
    do_sample: bool = True,
) -> GeneratedRollout:
    """Generate a continuation after a fixed response prefix without retokenizing it.

    This singleton helper is used by TrOPD's off-policy guidance.  It keeps peak
    memory low and preserves the exact teacher prefix tokenization.  The returned
    rollout contains ``prefix || student_continuation`` and never exceeds the
    ordinary ``generation.max_new_tokens`` response budget.
    """
    prompt_ids = _tokenize_one_cpu(tokenizer, prompt)
    prefix = prefix_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    maximum = int(generation_cfg["max_new_tokens"])
    if prefix.numel() >= maximum:
        trimmed = prefix[:maximum].clone()
        return GeneratedRollout(
            response_ids=trimmed,
            response_text=tokenizer.decode(trimmed, skip_special_tokens=True),
            finished=False,
        )
    remaining = maximum - int(prefix.numel())
    combined = torch.cat([prompt_ids, prefix], dim=0)
    check_generation_budget(combined.unsqueeze(0), remaining, int(sequence_cfg["max_context_tokens"]))
    device = model_device(model)
    input_ids = combined.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    model_was_training = model.training
    model.eval()
    try:
        args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": remaining,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": _generation_eos_token_id(tokenizer),
            "use_cache": True,
        }
        if do_sample:
            args.update(
                temperature=float(generation_cfg["temperature"]),
                top_p=float(generation_cfg["top_p"]),
            )
        generated = model.generate(**args)
    finally:
        if model_was_training:
            model.train()
    continuation, finished = _trim_generated_response(
        generated[0, input_ids.shape[1]:],
        eos_token_id=_generation_eos_token_id(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    continuation = continuation.detach().cpu().long()
    response = torch.cat([prefix, continuation], dim=0)
    # Concatenating the forced prefix can itself reach the response budget.
    finished = bool(finished) and int(response.numel()) <= maximum
    response = response[:maximum]
    return GeneratedRollout(
        response_ids=response,
        response_text=tokenizer.decode(response, skip_special_tokens=True),
        finished=finished,
    )
