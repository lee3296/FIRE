"""Vectorized stable on-policy self-distillation training with an EMA teacher."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from .contexts import ContextRecord, TeacherContextBuilder
from .data import DatasetAdapter, TaskExample
from .ema import LoRAEMA
from .evaluate import evaluate
from .losses import (
    demopsd_barycenter_target_log_probs,
    demopsd_leakage_attenuation,
    full_forward_kl,
    full_reverse_kl,
    full_reverse_kl_tokens,
    fisher_step_projected_sft_loss,
    gathered_token_log_probs,
    jensen_shannon_divergence_tokens,
    k1_policy_gradient_tokens,
    on_policy_sft_group_losses,
    scope_group_weights,
    supervised_nll_loss,
    topd_length_normalized_returns,
    topd_proximal_rewards,
    topk_union_reverse_kl,
    topk_union_reverse_kl_tokens,
    veto_target_log_probs,
)
from .metrics import RoutingMeter, format_routing
from .modeling import (
    GeneratedRollout,
    SequenceBudgetError,
    generate_from_forced_prefix,
    generate_rollout_batch,
    generation_budget_errors,
    response_logits_batch,
    scoring_budget_errors,
)
from .prompts import render_chat, student_messages, teacher_messages
from .routing import normalize_method, route_targets_batch
from .utils import get_logger, write_json


@dataclass
class TrainState:
    optimizer_step: int = 0
    optimizer_updates: int = 0
    seen_examples: int = 0
    skipped_over_budget: int = 0
    dropped_unfinished: int = 0
    loss_sum: float = 0.0
    loss_count: int = 0
    training_seconds: float = 0.0
    generated_tokens: int = 0


@dataclass
class _PreparedExample:
    example: TaskExample
    student_prompt: str
    rollout: GeneratedRollout
    context: ContextRecord | None
    is_correct: bool
    action: str


@dataclass
class _TopDRolloutBuffer:
    """Frozen behavior-policy statistics for one TOP-D global iteration."""

    entries: list[tuple[int, TaskExample, str, GeneratedRollout, ContextRecord]]
    prompts: list[str]
    response_ids: list[torch.Tensor]
    old_token_logp: list[torch.Tensor]
    advantages: list[torch.Tensor]
    active_prompt_indices: list[int]


@dataclass
class _OnPolicySFTNormalization:
    """One retained-length denominator across an optimizer step's microbatches."""

    maximum_length: int = 1
    losses: list[float] = field(default_factory=list)


class StableSDTrainer:
    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        model,
        tokenizer,
        ema: LoRAEMA,
        adapter: DatasetAdapter,
        train_examples: list[TaskExample],
        eval_examples: list[TaskExample],
        run_dir: Path,
    ):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.ema = ema
        self.adapter = adapter
        self.train_examples = train_examples
        self.eval_examples = eval_examples
        self.run_dir = run_dir
        self.logger = get_logger("stablesd.train")
        self.context_builder = TeacherContextBuilder(cfg["context"], seed=int(cfg["train"]["seed"]))
        self.rng = random.Random(int(cfg["train"]["seed"]))
        self.state = TrainState()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(cfg["optimizer"]["lr"]),
            betas=tuple(float(v) for v in cfg["optimizer"]["betas"]),
            weight_decay=float(cfg["optimizer"]["weight_decay"]),
        )
        self.optimizer.zero_grad(set_to_none=True)

    def _lr(self, step: int) -> float:
        schedule = str(self.cfg["optimizer"].get("schedule", "cosine"))
        maximum = float(self.cfg["optimizer"]["lr"])
        warmup = int(self.cfg["optimizer"].get("warmup_steps", 0))
        total = max(int(self.cfg["train"]["max_steps"]), 1)
        if warmup and step <= warmup:
            return maximum * step / warmup
        if schedule == "constant":
            return maximum
        progress = (step - warmup) / max(total - warmup, 1)
        return maximum * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


    def _fire_step_scale(self) -> float:
        """Return max(1, scheduled_lr / nominal_lr) for target construction."""
        fire_cfg = dict(self.cfg.get("fire", {}))
        if not bool(fire_cfg.get("step_normalized", False)):
            return 1.0
        nominal = float(self.cfg["optimizer"].get("nominal_lr", self.cfg["optimizer"]["lr"]))
        if nominal <= 0.0:
            raise ValueError("optimizer.nominal_lr must be positive")
        current = self._lr(self.state.optimizer_step + 1)
        return max(1.0, current / nominal)

    def _loss(self, student_logits: torch.Tensor, target_log_probs: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        loss_cfg = self.cfg["loss"]
        if loss_cfg["mode"] == "full":
            return full_reverse_kl(student_logits, target_log_probs, response_mask=response_mask)
        return topk_union_reverse_kl(
            student_logits,
            target_log_probs,
            student_top_k=int(loss_cfg["student_top_k"]),
            target_top_k=int(loss_cfg["target_top_k"]),
            response_mask=response_mask,
        )

    def _log_skip(self, example: TaskExample, exc: SequenceBudgetError) -> None:
        self.state.skipped_over_budget += 1
        self.logger.warning("Skipping example uid=%s without truncation: %s", example.uid, exc)

    def _keep_baseline_rollout(self, example: TaskExample, rollout: GeneratedRollout) -> bool:
        """Apply the existing completion/answer policy to custom baseline paths."""
        if not bool(self.cfg.get("generation", {}).get("drop_unfinished_rollouts", False)):
            return True
        if bool(getattr(rollout, "finished", True)) and self.adapter.has_final_answer(rollout.response_text, example):
            return True
        self.state.dropped_unfinished += 1
        self.logger.info("Dropping rollout uid=%s: unfinished or missing a final answer.", example.uid)
        return False

    def _correctness_routing_enabled(self) -> bool:
        routing_cfg = dict(self.cfg.get("correctness_routing", {}))
        return bool(routing_cfg.get("enabled", False))

    def _correctness_actions(self) -> tuple[str, str]:
        """Return actions for (correct, incorrect) rollouts.

        With routing disabled, both outcomes receive ordinary distillation.
        Routing-enabled methods may instead send correct rollouts to on-policy
        SFT or skip either outcome entirely.
        """
        if not self._correctness_routing_enabled():
            return "distill", "distill"
        routing_cfg = dict(self.cfg.get("correctness_routing", {}))
        return (
            str(routing_cfg.get("correct_action", "skip")),
            str(routing_cfg.get("incorrect_action", "distill")),
        )

    def _prepare_microbatch(
        self,
        examples: Sequence[TaskExample],
        meter: RoutingMeter | None = None,
    ) -> list[_PreparedExample]:
        sequence_cfg = self.cfg["sequence"]
        student_prompts = [render_chat(self.tokenizer, student_messages(example)) for example in examples]
        generation_errors = generation_budget_errors(
            self.tokenizer,
            student_prompts,
            max_new_tokens=int(self.cfg["generation"]["max_new_tokens"]),
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
        )
        valid_indices = [index for index, error in enumerate(generation_errors) if error is None]
        for index, error in enumerate(generation_errors):
            if error is not None:
                self._log_skip(examples[index], error)
        if not valid_indices:
            return []

        valid_prompts = [student_prompts[index] for index in valid_indices]
        valid_examples = [examples[index] for index in valid_indices]
        try:
            rollouts = generate_rollout_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=valid_prompts,
                generation_cfg=self.cfg["generation"],
                sequence_cfg=sequence_cfg,
                do_sample=True,
            )
        except SequenceBudgetError as exc:
            for example in valid_examples:
                self._log_skip(example, exc)
            return []

        self.state.generated_tokens += sum(int(rollout.response_ids.numel()) for rollout in rollouts)

        # Generation-budget health.  A rollout that exhausts
        # ``generation.max_new_tokens`` without emitting EOS has no final-answer
        # line, so the verifier label built from it reports where the budget ran
        # out rather than whether the reasoning was right.  These counters are
        # pure diagnostics and never change the objective.
        finished_flags = [bool(getattr(rollout, "finished", True)) for rollout in rollouts]
        answered_flags = [
            bool(self.adapter.has_final_answer(rollout.response_text, example))
            for example, rollout in zip(valid_examples, rollouts)
        ]
        if meter is not None:
            total_rollouts = max(len(rollouts), 1)
            budget = max(int(self.cfg["generation"]["max_new_tokens"]), 1)
            lengths = [int(rollout.response_ids.numel()) for rollout in rollouts]
            mean_length = sum(lengths) / total_rollouts
            meter.add_extra(
                "rollout_truncated_rate", sum(not flag for flag in finished_flags) / total_rollouts
            )
            meter.add_extra(
                "rollout_missing_final_answer_rate",
                sum(not flag for flag in answered_flags) / total_rollouts,
            )
            meter.add_extra("rollout_mean_response_tokens", mean_length)
            meter.add_extra("rollout_length_saturation", mean_length / budget)

        if bool(self.cfg.get("generation", {}).get("drop_unfinished_rollouts", False)):
            keep = [
                index
                for index, (finished, answered) in enumerate(zip(finished_flags, answered_flags))
                if finished and answered
            ]
            dropped = len(rollouts) - len(keep)
            if dropped:
                self.state.dropped_unfinished += dropped
                self.logger.info(
                    "Dropping %d/%d rollouts that hit the generation budget without a final "
                    "answer line; their verifier labels would not be real correctness signals.",
                    dropped,
                    len(rollouts),
                )
            valid_examples = [valid_examples[index] for index in keep]
            valid_prompts = [valid_prompts[index] for index in keep]
            rollouts = [rollouts[index] for index in keep]
            if not rollouts:
                return []

        correct_action, incorrect_action = self._correctness_actions()
        correct_flags = [
            self.adapter.is_correct(rollout.response_text, example)
            for example, rollout in zip(valid_examples, rollouts)
        ]
        if meter is not None:
            total = max(len(correct_flags), 1)
            correct_count = sum(bool(flag) for flag in correct_flags)
            incorrect_count = len(correct_flags) - correct_count
            meter.add_extra("correctness_route_correct_rate", correct_count / total)
            meter.add_extra("correctness_route_correct_sft_fraction", (correct_count / total) if correct_action == "sft" else 0.0)
            meter.add_extra("correctness_route_distill_fraction", (
                (correct_count if correct_action == "distill" else 0)
                + (incorrect_count if incorrect_action == "distill" else 0)
            ) / total)
            meter.add_extra("correctness_route_skipped_correct", correct_count if correct_action == "skip" else 0)
            meter.add_extra("correctness_route_skipped_incorrect", incorrect_count if incorrect_action == "skip" else 0)

        prepared: list[_PreparedExample] = []
        for example, prompt, rollout, is_correct in zip(valid_examples, valid_prompts, rollouts, correct_flags):
            action = correct_action if is_correct else incorrect_action
            if action == "skip":
                continue
            context = None
            if action == "distill":
                context = self.context_builder.build(example, rollout.response_text, self.adapter, rng=self.rng)
            prepared.append(
                _PreparedExample(
                    example=example,
                    student_prompt=prompt,
                    rollout=rollout,
                    context=context,
                    is_correct=bool(is_correct),
                    action=action,
                )
            )
        return prepared

    def _filter_scoring_budget(self, prepared: Sequence[_PreparedExample]) -> list[_PreparedExample]:
        """Check the exact scoring prompt used by each routed branch."""
        sequence_cfg = self.cfg["sequence"]
        scoring_prompts: list[str] = []
        for item in prepared:
            if item.action == "sft":
                scoring_prompts.append(item.student_prompt)
                continue
            if item.context is None:
                raise RuntimeError("Distillation item is missing its privileged context.")
            scoring_prompts.append(
                render_chat(
                    self.tokenizer,
                    teacher_messages(
                        item.example,
                        item.context.blocks,
                        item.context.block_fields,
                        context_type=item.context.context_type,
                    ),
                )
            )
        errors = scoring_budget_errors(
            self.tokenizer,
            scoring_prompts,
            [item.rollout.response_ids for item in prepared],
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
        )
        valid: list[_PreparedExample] = []
        for item, error in zip(prepared, errors):
            if error is None:
                valid.append(item)
            else:
                self._log_skip(item.example, error)
        return valid


    def _distill_token_losses(self, student_logits: torch.Tensor, target_log_probs: torch.Tensor) -> torch.Tensor:
        loss_cfg = self.cfg["loss"]
        if loss_cfg["mode"] == "full":
            return full_reverse_kl_tokens(student_logits, target_log_probs)
        return topk_union_reverse_kl_tokens(
            student_logits,
            target_log_probs,
            student_top_k=int(loss_cfg["student_top_k"]),
            target_top_k=int(loss_cfg["target_top_k"]),
        )

    @staticmethod
    def _target_entropy(target_log_probs: torch.Tensor) -> torch.Tensor:
        q = target_log_probs.float().exp()
        return -(q * target_log_probs.float()).sum(dim=-1)


    def _backward_srpo_microbatch(self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float) -> list[float]:
        """Sample-Routed Policy Optimization baseline for SDPO-style feedback.

        Correct rollouts receive a clipped policy-gradient reward update; incorrect rollouts
        with feedback receive entropy-weighted SDPO distillation.
        """
        sequence_cfg = self.cfg["sequence"]
        srpo_cfg = dict(self.cfg.get("srpo", {}))
        group_size = int(srpo_cfg.get("group_size", 4))
        group_size = max(group_size, 2)
        prompts_one = [render_chat(self.tokenizer, student_messages(example)) for example in examples]
        repeated_prompts = [prompt for prompt in prompts_one for _ in range(group_size)]
        repeated_examples = [example for example in examples for _ in range(group_size)]
        gen_errors = generation_budget_errors(
            self.tokenizer,
            repeated_prompts,
            max_new_tokens=int(self.cfg["generation"]["max_new_tokens"]),
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
        )
        valid_indices = [i for i, error in enumerate(gen_errors) if error is None]
        for i, error in enumerate(gen_errors):
            if error is not None:
                self._log_skip(repeated_examples[i], error)
        if not valid_indices:
            return []
        try:
            rollouts = generate_rollout_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[repeated_prompts[i] for i in valid_indices],
                generation_cfg=self.cfg["generation"],
                sequence_cfg=sequence_cfg,
                do_sample=True,
            )
        except SequenceBudgetError as exc:
            for i in valid_indices:
                self._log_skip(repeated_examples[i], exc)
            return []
        self.state.generated_tokens += sum(int(rollout.response_ids.numel()) for rollout in rollouts)
        expanded: list[tuple[int, TaskExample, str, GeneratedRollout, bool, ContextRecord]] = []
        for source_index, rollout in zip(valid_indices, rollouts):
            prompt_index = source_index // group_size
            example = repeated_examples[source_index]
            if not self._keep_baseline_rollout(example, rollout):
                continue
            correct = self.adapter.is_correct(rollout.response_text, example)
            context = self.context_builder.build(example, rollout.response_text, self.adapter, rng=self.rng)
            expanded.append((prompt_index, example, repeated_prompts[source_index], rollout, correct, context))
        if not expanded:
            return []

        scoring_errors = scoring_budget_errors(
            self.tokenizer,
            [item[2] for item in expanded],
            [item[3].response_ids for item in expanded],
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
        )
        retained = [item for item, error in zip(expanded, scoring_errors) if error is None]
        for item, error in zip(expanded, scoring_errors):
            if error is not None:
                self._log_skip(item[1], error)
        if not retained:
            return []

        prompts = [item[2] for item in retained]
        response_ids = [item[3].response_ids for item in retained]
        prompt_indices = [item[0] for item in retained]
        correct_flags = [bool(item[4]) for item in retained]
        rewards_by_prompt: dict[int, list[float]] = {i: [] for i in range(len(examples))}
        for prompt_index, correct in zip(prompt_indices, correct_flags):
            rewards_by_prompt.setdefault(prompt_index, []).append(1.0 if correct else 0.0)
        advantages: list[float] = []
        eps = float(srpo_cfg.get("advantage_eps", 1e-6))
        for prompt_index, correct in zip(prompt_indices, correct_flags):
            values = torch.tensor(rewards_by_prompt[prompt_index], dtype=torch.float32)
            mean = values.mean()
            std = values.std(unbiased=False)
            reward = torch.tensor(1.0 if correct else 0.0)
            advantage = (reward - mean) / (std + eps) if float(std) > 0.0 else torch.tensor(0.0)
            advantages.append(float(advantage.item()))

        with torch.no_grad():
            old_scored = response_logits_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=prompts,
                response_ids=response_ids,
                max_context_tokens=int(sequence_cfg["max_context_tokens"]),
                requires_grad=False,
            )
            old_token_logp = gathered_token_log_probs(old_scored.logits, response_ids).detach()
        self.model.train()
        current_scored = response_logits_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            response_ids=response_ids,
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
            requires_grad=True,
        )
        if not torch.equal(old_scored.response_mask, current_scored.response_mask):
            raise RuntimeError("SRPO old-policy mask disagrees with current-policy mask.")

        response_mask = current_scored.response_mask
        device = current_scored.logits.device
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        correct_tensor = torch.tensor(correct_flags, dtype=torch.bool, device=device)
        valid_mask = response_mask.to(dtype=torch.float32)

        current_token_logp = gathered_token_log_probs(current_scored.logits, response_ids)
        ratio_clip = float(srpo_cfg.get("ratio_clip", 2.0))
        ppo_eps = float(srpo_cfg.get("clip_epsilon", 0.2))
        ratio = (current_token_logp - old_token_logp.to(device)).exp().clamp(max=ratio_clip)
        clipped = ratio.clamp(1.0 - ppo_eps, 1.0 + ppo_eps)
        adv = advantages_tensor[:, None]
        policy_tokens = -torch.minimum(ratio * adv, clipped * adv)
        policy_mask = valid_mask * correct_tensor[:, None].float()
        policy_sum = (policy_tokens * policy_mask).sum()

        sdpo_indices = [index for index, correct in enumerate(correct_flags) if not correct]
        sdpo_sum = torch.zeros((), dtype=torch.float32, device=device)
        sdpo_token_count = torch.zeros((), dtype=torch.float32, device=device)
        if sdpo_indices:
            routed = route_targets_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                ema=self.ema,
                examples=[retained[index][1] for index in sdpo_indices],
                response_ids=[response_ids[index] for index in sdpo_indices],
                contexts=[retained[index][5] for index in sdpo_indices],
                method="full_context",
                max_context_tokens=int(sequence_cfg["max_context_tokens"]),
                candidate_chunk_blocks=int(self.cfg.get("routing", {}).get("candidate_chunk_blocks", 1)),
                collect_diagnostics=False,
            )
            target_width = int(routed.target_log_probs.shape[1])
            sdpo_logits = current_scored.logits[sdpo_indices, :target_width, :]
            sdpo_mask = response_mask[sdpo_indices, :target_width]
            if not torch.equal(sdpo_mask, routed.response_mask):
                raise RuntimeError("SRPO SDPO-branch mask disagrees with routed EMA target mask.")
            token_losses = self._distill_token_losses(sdpo_logits, routed.target_log_probs)
            entropy = self._target_entropy(routed.target_log_probs)
            beta = float(srpo_cfg.get("entropy_beta", 1.0))
            raw_weights = torch.exp(-beta * entropy).clamp_min(1e-8)
            sdpo_mask_f = sdpo_mask.to(dtype=torch.float32)
            normalizer = (raw_weights * sdpo_mask_f).sum() / sdpo_mask_f.sum().clamp_min(1.0)
            weights = raw_weights / normalizer.clamp_min(1e-8)
            sdpo_sum = (token_losses * weights * sdpo_mask_f).sum()
            sdpo_token_count = sdpo_mask_f.sum()
            for diagnostics in routed.diagnostics:
                meter.update(diagnostics)
            del routed, token_losses, entropy, weights

        policy_token_count = policy_mask.sum()
        total_tokens = (policy_token_count + sdpo_token_count).clamp_min(1.0)
        total_loss = (policy_sum + sdpo_sum) / total_tokens
        # Match existing prompt-level accumulation: one prompt contributes one unit,
        # even though SRPO internally samples a small group of rollouts.
        (total_loss * len(examples) * scale).backward()
        correct_rate = sum(correct_flags) / max(len(correct_flags), 1)
        meter.add_extra("srpo_correct_rate", correct_rate)
        meter.add_extra("srpo_sdpo_fraction", len(sdpo_indices) / max(len(correct_flags), 1))
        meter.add_extra("srpo_group_size", group_size)
        raw_loss = float(total_loss.detach().cpu().item())
        del old_scored, current_scored, total_loss
        return [raw_loss] * len(examples)

    @staticmethod
    def _pad_token_lists(
        values: Sequence[torch.Tensor],
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        fill: float = 0.0,
    ) -> torch.Tensor:
        output = torch.full((len(values), width), float(fill), dtype=dtype, device=device)
        for row, value in enumerate(values):
            flat = value.detach().to(device=device, dtype=dtype).reshape(-1)
            length = min(int(flat.numel()), width)
            if length:
                output[row, :length] = flat[:length]
        return output

    def _teacher_prompt(self, example: TaskExample, context: ContextRecord) -> str:
        return render_chat(
            self.tokenizer,
            teacher_messages(
                example,
                context.blocks,
                context.block_fields,
                context_type=context.context_type,
            ),
        )

    def _score_gathered_logp_streamed(
        self,
        *,
        prompts: Sequence[str],
        response_ids: Sequence[torch.Tensor],
        use_ema: bool,
        chunk_size: int,
    ) -> list[torch.Tensor]:
        """Score sampled tokens while discarding each full-vocabulary chunk immediately."""
        output: list[torch.Tensor] = []
        model_was_training = self.model.training
        self.model.eval()
        context = self.ema.swap_into(self.model) if use_ema else None
        if context is not None:
            context.__enter__()
        try:
            for start in range(0, len(prompts), max(1, int(chunk_size))):
                stop = min(start + max(1, int(chunk_size)), len(prompts))
                scored = response_logits_batch(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=list(prompts[start:stop]),
                    response_ids=list(response_ids[start:stop]),
                    max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
                    requires_grad=False,
                )
                gathered = gathered_token_log_probs(scored.logits, list(response_ids[start:stop]))
                for row, ids in enumerate(response_ids[start:stop]):
                    output.append(gathered[row, : int(ids.numel())].detach().cpu())
                del scored, gathered
        finally:
            if context is not None:
                context.__exit__(None, None, None)
            if model_was_training:
                self.model.train()
        return output

    def _score_teacher_topk_streamed(
        self,
        *,
        prompts: Sequence[str],
        response_ids: Sequence[torch.Tensor],
        top_k: int,
        chunk_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Return teacher sampled-token logp and top-k support without retaining full logits."""
        sampled: list[torch.Tensor] = []
        top_values: list[torch.Tensor] = []
        top_indices: list[torch.Tensor] = []
        model_was_training = self.model.training
        self.model.eval()
        with self.ema.swap_into(self.model):
            for start in range(0, len(prompts), max(1, int(chunk_size))):
                stop = min(start + max(1, int(chunk_size)), len(prompts))
                ids_chunk = list(response_ids[start:stop])
                scored = response_logits_batch(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=list(prompts[start:stop]),
                    response_ids=ids_chunk,
                    max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
                    requires_grad=False,
                )
                logq = torch.log_softmax(scored.logits.float(), dim=-1)
                gathered = gathered_token_log_probs(scored.logits, ids_chunk)
                k = min(max(int(top_k), 1), int(logq.shape[-1]))
                values, indices = torch.topk(logq, k=k, dim=-1)
                for row, ids in enumerate(ids_chunk):
                    length = int(ids.numel())
                    sampled.append(gathered[row, :length].detach().cpu())
                    top_values.append(values[row, :length].detach().cpu())
                    top_indices.append(indices[row, :length].detach().cpu())
                del scored, logq, gathered, values, indices
        if model_was_training:
            self.model.train()
        return sampled, top_values, top_indices

    def _backward_on_policy_sft_microbatch(
        self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float,
        *, normalization: _OnPolicySFTNormalization | None = None,
    ) -> list[float]:
        """Faithful Eq. 9 On-Policy SFT with correctness/length filtering."""
        cfg = dict(self.cfg.get("on_policy_sft", {}))
        group_size = max(1, int(cfg.get("group_size", 8)))
        length_limit = int(cfg.get("length_limit", self.cfg["generation"]["max_new_tokens"]))
        prompts_one = [render_chat(self.tokenizer, student_messages(example)) for example in examples]
        prompts = [prompt for prompt in prompts_one for _ in range(group_size)]
        repeated_examples = [example for example in examples for _ in range(group_size)]
        errors = generation_budget_errors(
            self.tokenizer,
            prompts,
            max_new_tokens=int(self.cfg["generation"]["max_new_tokens"]),
            max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
        )
        valid = [index for index, error in enumerate(errors) if error is None]
        for index, error in enumerate(errors):
            if error is not None:
                self._log_skip(repeated_examples[index], error)
        if not valid:
            return []
        rollouts = generate_rollout_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[prompts[index] for index in valid],
            generation_cfg=self.cfg["generation"],
            sequence_cfg=self.cfg["sequence"],
            do_sample=True,
        )
        self.state.generated_tokens += sum(int(rollout.response_ids.numel()) for rollout in rollouts)
        retained: list[tuple[int, str, GeneratedRollout]] = []
        for source_index, rollout in zip(valid, rollouts):
            prompt_index = source_index // group_size
            example = repeated_examples[source_index]
            if not self._keep_baseline_rollout(example, rollout):
                continue
            accepted = self.adapter.is_correct(rollout.response_text, example) and int(rollout.response_ids.numel()) <= length_limit
            if accepted:
                retained.append((prompt_index, prompts[source_index], rollout))
        meter.add_extra("on_policy_sft_rollouts", len(rollouts))
        meter.add_extra("on_policy_sft_accepted", len(retained))
        meter.add_extra("on_policy_sft_accept_rate", len(retained) / max(len(rollouts), 1))
        meter.add_extra("on_policy_sft_group_size", group_size)
        if not retained:
            return []
        score_errors = scoring_budget_errors(
            self.tokenizer,
            [item[1] for item in retained],
            [item[2].response_ids for item in retained],
            max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
        )
        filtered = [item for item, error in zip(retained, score_errors) if error is None]
        for item, error in zip(retained, score_errors):
            if error is not None:
                self._log_skip(examples[item[0]], error)
        if not filtered:
            return []
        self.model.train()
        scored = response_logits_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[item[1] for item in filtered],
            response_ids=[item[2].response_ids for item in filtered],
            max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
            requires_grad=True,
        )
        prompt_indices = torch.tensor([item[0] for item in filtered], device=scored.logits.device)
        prompt_losses = on_policy_sft_group_losses(
            scored.logits,
            [item[2].response_ids for item in filtered],
            scored.response_mask,
            torch.ones(len(filtered), dtype=torch.bool, device=scored.logits.device),
            prompt_indices,
            len(examples),
            group_size,
        )
        if normalization is not None:
            local_maximum = max(int(scored.response_mask.sum(dim=-1).max().item()), 1)
            if local_maximum > normalization.maximum_length:
                # Keep all earlier gradients/losses divided by the largest
                # retained length seen so far.  Updating this running maximum
                # preserves generation/backward order and retains no graphs.
                correction = normalization.maximum_length / local_maximum
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                normalization.losses[:] = [value * correction for value in normalization.losses]
                normalization.maximum_length = local_maximum
            if local_maximum != normalization.maximum_length:
                prompt_losses = prompt_losses * (local_maximum / normalization.maximum_length)
        (prompt_losses.sum() * scale).backward()
        raw = [float(value) for value in prompt_losses.detach().cpu().tolist()]
        if normalization is not None:
            normalization.losses.extend(raw)
        del scored, prompt_losses
        return raw

    def _backward_veto_microbatch(
        self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float
    ) -> list[float]:
        """Veto product-of-experts target with the paper's linear beta decay."""
        prepared = self._filter_scoring_budget(self._prepare_microbatch(examples, meter=meter))
        if not prepared:
            return []
        if any(item.context is None for item in prepared):
            raise RuntimeError("Veto requires a full teacher context for every rollout")
        routed = route_targets_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            ema=self.ema,
            examples=[item.example for item in prepared],
            response_ids=[item.rollout.response_ids for item in prepared],
            contexts=[item.context for item in prepared if item.context is not None],
            method="full_context",
            max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
            collect_diagnostics=False,
        )
        self.model.train()
        scored = response_logits_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[item.student_prompt for item in prepared],
            response_ids=[item.rollout.response_ids for item in prepared],
            max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
            requires_grad=True,
        )
        veto_cfg = dict(self.cfg.get("veto", {}))
        maximum_step = max(int(self.cfg["train"]["max_steps"]) - 1, 1)
        progress = min(max(self.state.optimizer_step / maximum_step, 0.0), 1.0)
        beta_start = float(veto_cfg.get("beta_start", 0.8))
        beta_end = float(veto_cfg.get("beta_end", 0.0))
        beta = beta_start + (beta_end - beta_start) * progress
        target = veto_target_log_probs(routed.target_log_probs, scored.logits, beta)
        objective = str(veto_cfg.get("objective", "forward_kl"))
        if objective == "forward_kl":
            losses = full_forward_kl(scored.logits, target, scored.response_mask)
        elif objective == "reverse_kl":
            losses = full_reverse_kl(scored.logits, target, scored.response_mask)
        else:
            raise ValueError("veto.objective must be forward_kl or reverse_kl")
        (losses.sum() * scale).backward()
        meter.add_extra("veto_beta", beta)
        meter.add_extra("veto_examples", len(prepared))
        raw = [float(value) for value in losses.detach().cpu().tolist()]
        del routed, scored, target, losses
        return raw

    def _backward_demopsd_microbatch(
        self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float
    ) -> list[float]:
        """DemoPSD disagreement-modulated reverse-KL barycenter target.

        The privileged teacher and the unprivileged student reference are both the
        EMA copy of the current student, matching the paper's stability choice
        (Sec. 4.1 / Algorithm 1): the teacher conditions on this repository's
        privileged SDPO feedback context while the reference conditions only on
        the plain question prompt.  Per token, the disagreement is
        ``d_t = JSD(pi_S || pi_T)`` (Eq. 7), the leakage attenuation coefficient
        is ``alpha_t = (sigmoid(beta * d_t) - 0.5) * 2 * alpha_max`` (Eq. 8,
        remap mode), and the student minimizes the configured reverse-KL loss
        toward the stop-gradded geometric barycenter target
        ``pi_target ∝ pi_T^{1-alpha_t} pi_S^{alpha_t}`` (Eqs. 9 and 12).

        The paper's reprompting filter (skip prompts whose rollout group has no
        correct rollout) exists only because its privileged information y* is a
        correct rollout.  In this repository the privileged information is the
        verifier feedback context, which is available for every rollout, so all
        valid rollouts are distilled — the same adaptation applied to Veto,
        TOP-D, and TrOPD, which share this privileged EMA teacher.
        """
        prepared = self._filter_scoring_budget(self._prepare_microbatch(examples, meter=meter))
        if not prepared:
            return []
        if any(item.context is None for item in prepared):
            raise RuntimeError("DemoPSD requires a full privileged teacher context for every rollout")
        demopsd_cfg = dict(self.cfg.get("demopsd", {}))
        alpha_max = float(demopsd_cfg.get("alpha_max", 0.15))
        beta = float(demopsd_cfg.get("beta", 50.0))
        sequence_cfg = self.cfg["sequence"]
        response_ids = [item.rollout.response_ids for item in prepared]
        student_prompts = [item.student_prompt for item in prepared]

        # Privileged teacher distribution pi_T(. | x, y*, yhat_<t) under the EMA copy.
        routed = route_targets_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            ema=self.ema,
            examples=[item.example for item in prepared],
            response_ids=response_ids,
            contexts=[item.context for item in prepared if item.context is not None],
            method="full_context",
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
            collect_diagnostics=False,
        )

        # Unprivileged student reference pi_S(. | x, yhat_<t) under the same EMA copy.
        model_was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self.ema.swap_into(self.model):
                reference_scored = response_logits_batch(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompts=student_prompts,
                    response_ids=response_ids,
                    max_context_tokens=int(sequence_cfg["max_context_tokens"]),
                    requires_grad=False,
                )
                reference_logq = torch.log_softmax(reference_scored.logits.float(), dim=-1).detach()
                reference_mask = reference_scored.response_mask
                del reference_scored
        finally:
            if model_was_training:
                self.model.train()
        if not torch.equal(reference_mask, routed.response_mask):
            raise RuntimeError("DemoPSD reference mask disagrees with the privileged teacher mask.")

        response_mask = routed.response_mask
        mask_f = response_mask.to(dtype=torch.float32)
        disagreement = jensen_shannon_divergence_tokens(reference_logq, routed.target_log_probs)
        alpha = demopsd_leakage_attenuation(disagreement, beta, alpha_max)
        alpha = torch.where(response_mask, alpha, torch.zeros_like(alpha))
        target_log_probs = demopsd_barycenter_target_log_probs(
            routed.target_log_probs, reference_logq, alpha
        )
        denominator = mask_f.sum().clamp_min(1.0)
        mean_alpha = float(((alpha * mask_f).sum() / denominator).detach().cpu().item())
        mean_disagreement = float(
            ((disagreement * mask_f).sum() / denominator).detach().cpu().item()
        )
        del routed, reference_logq, disagreement, alpha

        self.model.train()
        student_scored = response_logits_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=student_prompts,
            response_ids=response_ids,
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
            requires_grad=True,
        )
        if not torch.equal(student_scored.response_mask, response_mask):
            raise RuntimeError("DemoPSD student autograd mask disagrees with the barycenter target mask.")
        per_example_loss = self._loss(
            student_scored.logits, target_log_probs, student_scored.response_mask
        )
        if per_example_loss.ndim != 1 or per_example_loss.shape[0] != len(prepared):
            raise RuntimeError("Batched DemoPSD loss must return one scalar per rollout.")
        (per_example_loss.sum() * scale).backward()
        meter.add_extra("demopsd_alpha_mean", mean_alpha)
        meter.add_extra("demopsd_disagreement_mean", mean_disagreement)
        meter.add_extra("demopsd_alpha_max", alpha_max)
        meter.add_extra("demopsd_beta", beta)
        meter.add_extra("demopsd_examples", len(prepared))
        raw = [float(value) for value in per_example_loss.detach().cpu().tolist()]
        del student_scored, target_log_probs, per_example_loss
        return raw

    def _backward_scope_microbatch(
        self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float
    ) -> list[float]:
        """SCOPE dual-path adaptive weighting.

        For each prompt, a group of on-policy rollouts is routed by verifier
        correctness (Sec. 3.1).  Correct trajectories receive student-perplexity
        weighted importance-sampled MLE, ``L_MLE = -sum_t rho_t`` with weights
        ``w_stu ∝ PPL_S^{1/tau}`` (Eqs. 2 and 4).  Incorrect trajectories receive
        teacher-perplexity weighted on-policy distillation with the detached
        token log-ratio as a negative advantage,
        ``L_OPD = sum_t rho_t (log pi_thetabar - log pi_T)`` with weights
        ``w_tea ∝ PPL_T^{-1/tau}`` (Eqs. 3 and 5).  Both weight distributions
        are normalized with a group-level softmax strictly within each prompt's
        candidate set (Eqs. 6 and 18-19), and the two branches are summed into
        one unified per-prompt objective.

        The teacher is this repository's privileged feedback-conditioned EMA
        teacher, matching the adaptation used for Veto, TOP-D, and TrOPD.
        Perplexities cover response tokens only, exactly as in the paper.
        """
        sequence_cfg = self.cfg["sequence"]
        scope_cfg = dict(self.cfg.get("scope", {}))
        group_size = max(2, int(scope_cfg.get("group_size", 4)))
        temperature = float(scope_cfg.get("weight_temperature", 1.0))
        chunk_size = max(1, int(scope_cfg.get("score_chunk_size", 2)))

        prompts_one = [render_chat(self.tokenizer, student_messages(example)) for example in examples]
        repeated_prompts = [prompt for prompt in prompts_one for _ in range(group_size)]
        repeated_examples = [example for example in examples for _ in range(group_size)]
        gen_errors = generation_budget_errors(
            self.tokenizer,
            repeated_prompts,
            max_new_tokens=int(self.cfg["generation"]["max_new_tokens"]),
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
        )
        valid_indices = [i for i, error in enumerate(gen_errors) if error is None]
        for i, error in enumerate(gen_errors):
            if error is not None:
                self._log_skip(repeated_examples[i], error)
        if not valid_indices:
            return []
        try:
            rollouts = generate_rollout_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[repeated_prompts[i] for i in valid_indices],
                generation_cfg=self.cfg["generation"],
                sequence_cfg=sequence_cfg,
                do_sample=True,
            )
        except SequenceBudgetError as exc:
            for i in valid_indices:
                self._log_skip(repeated_examples[i], exc)
            return []
        self.state.generated_tokens += sum(int(rollout.response_ids.numel()) for rollout in rollouts)

        # Outcome-driven group branching: only incorrect rollouts require the
        # privileged teacher context; correct rollouts are scored on the plain
        # student prompt alone.
        entries: list[tuple[int, TaskExample, str, GeneratedRollout, bool, ContextRecord | None]] = []
        for source_index, rollout in zip(valid_indices, rollouts):
            prompt_index = source_index // group_size
            example = repeated_examples[source_index]
            if not self._keep_baseline_rollout(example, rollout):
                continue
            correct = bool(self.adapter.is_correct(rollout.response_text, example))
            context: ContextRecord | None = None
            if not correct:
                context = self.context_builder.build(example, rollout.response_text, self.adapter, rng=self.rng)
            scoring_prompts = [repeated_prompts[source_index]]
            scoring_ids = [rollout.response_ids]
            if context is not None:
                scoring_prompts.append(self._teacher_prompt(example, context))
                scoring_ids.append(rollout.response_ids)
            errors = scoring_budget_errors(
                self.tokenizer,
                scoring_prompts,
                scoring_ids,
                max_context_tokens=int(sequence_cfg["max_context_tokens"]),
            )
            if any(error is not None for error in errors):
                self._log_skip(example, next(error for error in errors if error is not None))
                continue
            entries.append((prompt_index, example, repeated_prompts[source_index], rollout, correct, context))
        if not entries:
            return []

        prompts = [item[2] for item in entries]
        response_ids = [item[3].response_ids for item in entries]
        correct_flags = [bool(item[4]) for item in entries]
        lengths = torch.tensor(
            [float(int(ids.numel())) for ids in response_ids], dtype=torch.float32
        )

        # Behavior-policy sampled-token log probabilities pi_old = pi_S at
        # sampling time; frozen for the ratio and for the student PPL weights.
        old_logp = self._score_gathered_logp_streamed(
            prompts=prompts,
            response_ids=response_ids,
            use_ema=False,
            chunk_size=chunk_size,
        )
        student_sequence_logp = torch.tensor(
            [float(value.sum().item()) for value in old_logp], dtype=torch.float32
        )

        # Privileged EMA teacher sampled-token log probabilities, incorrect only.
        incorrect_rows = [index for index, correct in enumerate(correct_flags) if not correct]
        teacher_sequence_logp = torch.zeros(len(entries), dtype=torch.float32)
        teacher_logp_by_row: dict[int, torch.Tensor] = {}
        if incorrect_rows:
            teacher_prompts = [
                self._teacher_prompt(entries[index][1], entries[index][5]) for index in incorrect_rows
            ]
            teacher_logp = self._score_gathered_logp_streamed(
                prompts=teacher_prompts,
                response_ids=[response_ids[index] for index in incorrect_rows],
                use_ema=True,
                chunk_size=chunk_size,
            )
            for row, value in zip(incorrect_rows, teacher_logp):
                teacher_logp_by_row[row] = value
                teacher_sequence_logp[row] = float(value.sum().item())

        # Group-level softmax weights, strictly within each prompt's candidate set.
        weights = torch.zeros(len(entries), dtype=torch.float32)
        for prompt_index in range(len(examples)):
            correct_rows = [
                index
                for index, item in enumerate(entries)
                if item[0] == prompt_index and correct_flags[index]
            ]
            wrong_rows = [
                index
                for index, item in enumerate(entries)
                if item[0] == prompt_index and not correct_flags[index]
            ]
            if correct_rows:
                group_weights = scope_group_weights(
                    student_sequence_logp[correct_rows],
                    lengths[correct_rows],
                    temperature,
                    mode="student",
                )
                for row, weight in zip(correct_rows, group_weights.tolist()):
                    weights[row] = float(weight)
            if wrong_rows:
                group_weights = scope_group_weights(
                    teacher_sequence_logp[wrong_rows],
                    lengths[wrong_rows],
                    temperature,
                    mode="teacher",
                )
                for row, weight in zip(wrong_rows, group_weights.tolist()):
                    weights[row] = float(weight)

        self.model.train()
        current_scored = response_logits_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            response_ids=response_ids,
            max_context_tokens=int(sequence_cfg["max_context_tokens"]),
            requires_grad=True,
        )
        current_token_logp = gathered_token_log_probs(current_scored.logits, response_ids)
        device = current_token_logp.device
        width = int(current_token_logp.shape[1])
        old_pad = self._pad_token_lists(old_logp, width, device=device)
        teacher_pad = self._pad_token_lists(
            [
                teacher_logp_by_row.get(index, torch.zeros(int(ids.numel())))
                for index, ids in enumerate(response_ids)
            ],
            width,
            device=device,
        )
        valid_mask = current_scored.response_mask.to(dtype=torch.float32)
        ratio = torch.exp(current_token_logp - old_pad.detach())
        correct_tensor = torch.tensor(correct_flags, dtype=torch.bool, device=device)
        weights_tensor = weights.to(device=device)

        # Eq. 2: L_MLE = -sum_t rho_t over the correct set.
        mle_sequence = -(ratio * valid_mask).sum(dim=-1)
        # Eq. 3: L_OPD = sum_t rho_t (log pi_thetabar - log pi_T) over the
        # incorrect set, with the detached log-ratio as a negative advantage.
        opd_advantage = (old_pad - teacher_pad).detach()
        opd_sequence = (ratio * opd_advantage * valid_mask).sum(dim=-1)
        sequence_loss = torch.where(correct_tensor, mle_sequence, opd_sequence) * weights_tensor

        prompt_loss = torch.zeros(len(examples), dtype=sequence_loss.dtype, device=device)
        prompt_indices = torch.tensor([item[0] for item in entries], dtype=torch.long, device=device)
        prompt_loss.scatter_add_(0, prompt_indices, sequence_loss)
        active_prompts = sorted({int(item[0]) for item in entries})
        (prompt_loss.sum() * scale).backward()

        correct_count = sum(correct_flags)
        incorrect_count = len(correct_flags) - correct_count
        student_ppl = torch.exp(-student_sequence_logp / lengths.clamp_min(1.0))
        meter.add_extra("scope_group_size", group_size)
        meter.add_extra("scope_weight_temperature", temperature)
        meter.add_extra("scope_correct_rate", correct_count / max(len(correct_flags), 1))
        meter.add_extra("scope_correct_rollouts", correct_count)
        meter.add_extra("scope_incorrect_rollouts", incorrect_count)
        meter.add_extra("scope_student_ppl_mean", float(student_ppl.mean().item()))
        if incorrect_rows:
            teacher_ppl = torch.exp(
                -teacher_sequence_logp[incorrect_rows] / lengths[incorrect_rows].clamp_min(1.0)
            )
            meter.add_extra("scope_teacher_ppl_mean", float(teacher_ppl.mean().item()))
        ratio_valid = ratio.detach()[current_scored.response_mask]
        if ratio_valid.numel():
            meter.add_extra("scope_ratio_mean", float(ratio_valid.mean().cpu().item()))
        raw = [
            float(prompt_loss[prompt_index].detach().cpu().item())
            for prompt_index in active_prompts
        ]
        del current_scored, current_token_logp, old_pad, teacher_pad, ratio
        del mle_sequence, opd_sequence, sequence_loss, prompt_loss
        return raw

    def _group_rollout_entries(
        self,
        examples: Sequence[TaskExample],
        *,
        group_size: int,
    ) -> list[tuple[int, TaskExample, str, GeneratedRollout, ContextRecord]]:
        prompts_one = [render_chat(self.tokenizer, student_messages(example)) for example in examples]
        prompts = [prompt for prompt in prompts_one for _ in range(group_size)]
        repeated_examples = [example for example in examples for _ in range(group_size)]
        errors = generation_budget_errors(
            self.tokenizer,
            prompts,
            max_new_tokens=int(self.cfg["generation"]["max_new_tokens"]),
            max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
        )
        valid = [index for index, error in enumerate(errors) if error is None]
        for index, error in enumerate(errors):
            if error is not None:
                self._log_skip(repeated_examples[index], error)
        if not valid:
            return []
        rollouts = generate_rollout_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=[prompts[index] for index in valid],
            generation_cfg=self.cfg["generation"],
            sequence_cfg=self.cfg["sequence"],
            do_sample=True,
        )
        self.state.generated_tokens += sum(int(rollout.response_ids.numel()) for rollout in rollouts)
        entries: list[tuple[int, TaskExample, str, GeneratedRollout, ContextRecord]] = []
        for source_index, rollout in zip(valid, rollouts):
            prompt_index = source_index // group_size
            example = repeated_examples[source_index]
            if not self._keep_baseline_rollout(example, rollout):
                continue
            context = self.context_builder.build(example, rollout.response_text, self.adapter, rng=self.rng)
            teacher_prompt = self._teacher_prompt(example, context)
            errors_pair = scoring_budget_errors(
                self.tokenizer,
                [prompts[source_index], teacher_prompt],
                [rollout.response_ids, rollout.response_ids],
                max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
            )
            if any(error is not None for error in errors_pair):
                self._log_skip(example, next(error for error in errors_pair if error is not None))
                continue
            entries.append((prompt_index, example, prompts[source_index], rollout, context))
        return entries

    def _build_topd_rollout_buffer(
        self,
        examples: Sequence[TaskExample],
        meter: RoutingMeter,
    ) -> _TopDRolloutBuffer | None:
        """Generate and score one frozen TOP-D behavior-policy batch.

        The behavior-policy sampled-token log probabilities, proximal rewards, and
        normalized advantages are computed once and kept on CPU.  No
        full-vocabulary tensor survives a scoring chunk.
        """
        cfg = dict(self.cfg.get("topd", {}))
        group_size = max(2, int(cfg.get("group_size", 8)))
        chunk_size = max(1, int(cfg.get("score_chunk_size", 2)))
        # Collect the global behavior batch in the existing prompt-microbatch
        # size.  This preserves TOP-D's frozen behavior policy while avoiding a
        # single generation call over effective_batch * group_size responses.
        collection_prompts = max(1, int(self.cfg["train"]["micro_batch_size"]))
        entries: list[tuple[int, TaskExample, str, GeneratedRollout, ContextRecord]] = []
        old_logp: list[torch.Tensor] = []
        returns: list[torch.Tensor] = []
        alpha = float(cfg.get("alpha", 0.2))
        for prompt_start in range(0, len(examples), collection_prompts):
            prompt_stop = min(prompt_start + collection_prompts, len(examples))
            local_entries = self._group_rollout_entries(
                examples[prompt_start:prompt_stop],
                group_size=group_size,
            )
            if not local_entries:
                continue
            remapped_entries = [
                (
                    int(item[0]) + prompt_start,
                    item[1],
                    item[2],
                    item[3],
                    item[4],
                )
                for item in local_entries
            ]
            local_prompts = [item[2] for item in remapped_entries]
            local_ids = [item[3].response_ids for item in remapped_entries]
            local_teacher_prompts = [
                self._teacher_prompt(item[1], item[4]) for item in remapped_entries
            ]

            # Both statistics are frozen for every later internal minibatch.
            local_old = self._score_gathered_logp_streamed(
                prompts=local_prompts,
                response_ids=local_ids,
                use_ema=False,
                chunk_size=chunk_size,
            )
            local_teacher = self._score_gathered_logp_streamed(
                prompts=local_teacher_prompts,
                response_ids=local_ids,
                use_ema=True,
                chunk_size=chunk_size,
            )
            local_returns = [
                topd_length_normalized_returns(
                    topd_proximal_rewards(q.unsqueeze(0), p.unsqueeze(0), alpha),
                    torch.ones((1, p.numel()), dtype=torch.bool),
                )[0]
                for q, p in zip(local_teacher, local_old)
            ]
            entries.extend(remapped_entries)
            old_logp.extend(local_old)
            returns.extend(local_returns)

        if not entries:
            return None
        prompts = [item[2] for item in entries]
        response_ids = [item[3].response_ids for item in entries]
        advantages: list[torch.Tensor] = [torch.zeros_like(value) for value in returns]
        eps = float(cfg.get("advantage_eps", 1.0e-6))
        active_prompt_indices: list[int] = []
        for prompt_index in range(len(examples)):
            rows = [index for index, item in enumerate(entries) if item[0] == prompt_index]
            if not rows:
                continue
            active_prompt_indices.append(prompt_index)
            flat = torch.cat([returns[index] for index in rows])
            mean = flat.mean()
            std = flat.std(unbiased=False)
            for index in rows:
                advantages[index] = (returns[index] - mean) / (std + eps)

        valid_advantages = torch.cat(
            [advantages[index] for index in range(len(advantages)) if advantages[index].numel()]
        )
        meter.add_extra("topd_group_size", group_size)
        meter.add_extra("topd_alpha", alpha)
        meter.add_extra("topd_rollouts", len(entries))
        meter.add_extra("topd_behavior_prompts", len(active_prompt_indices))
        meter.add_extra("topd_adv_mean", float(valid_advantages.mean().item()))
        meter.add_extra("topd_adv_std", float(valid_advantages.std(unbiased=False).item()))

        return _TopDRolloutBuffer(
            entries=entries,
            prompts=prompts,
            response_ids=response_ids,
            old_token_logp=old_logp,
            advantages=advantages,
            active_prompt_indices=active_prompt_indices,
        )

    def _backward_topd_internal_minibatch(
        self,
        buffer: _TopDRolloutBuffer,
        prompt_indices: Sequence[int],
    ) -> dict[str, float]:
        """Backpropagate one PPO-clipped TOP-D minibatch against frozen old logp."""
        cfg = dict(self.cfg.get("topd", {}))
        chunk_size = max(1, int(cfg.get("score_chunk_size", 2)))
        clip = float(cfg.get("clip_epsilon", 0.2))
        selected_prompts = set(int(value) for value in prompt_indices)
        rows = [
            index
            for index, entry in enumerate(buffer.entries)
            if int(entry[0]) in selected_prompts
        ]
        if not rows:
            return {
                "loss": 0.0,
                "ratio_mean": 1.0,
                "ratio_std": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "abs_surrogate": 0.0,
                "tokens": 0.0,
            }

        token_counts = {
            prompt_index: sum(
                int(buffer.response_ids[index].numel())
                for index in rows
                if int(buffer.entries[index][0]) == prompt_index
            )
            for prompt_index in selected_prompts
        }
        prompt_denominator = float(max(len(selected_prompts), 1))
        total_loss_value = 0.0
        ratio_values: list[torch.Tensor] = []
        approx_kl_values: list[torch.Tensor] = []
        clip_values: list[torch.Tensor] = []
        abs_surrogate_values: list[torch.Tensor] = []

        self.model.train()
        for start in range(0, len(rows), chunk_size):
            chunk_rows = rows[start : start + chunk_size]
            chunk_prompts = [buffer.prompts[index] for index in chunk_rows]
            chunk_ids = [buffer.response_ids[index] for index in chunk_rows]
            scored = response_logits_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=chunk_prompts,
                response_ids=chunk_ids,
                max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
                requires_grad=True,
            )
            current = gathered_token_log_probs(scored.logits, chunk_ids)
            old_pad = self._pad_token_lists(
                [buffer.old_token_logp[index] for index in chunk_rows],
                current.shape[1],
                device=current.device,
            )
            adv_pad = self._pad_token_lists(
                [buffer.advantages[index] for index in chunk_rows],
                current.shape[1],
                device=current.device,
            )
            ratio = torch.exp(current - old_pad)
            clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
            surrogate = torch.minimum(
                ratio * adv_pad.detach(),
                clipped_ratio * adv_pad.detach(),
            )
            mask = scored.response_mask.to(dtype=surrogate.dtype)
            chunk_loss = torch.zeros((), dtype=surrogate.dtype, device=surrogate.device)
            for row, entry_index in enumerate(chunk_rows):
                prompt_index = int(buffer.entries[entry_index][0])
                denominator = float(max(token_counts[prompt_index], 1))
                chunk_loss = chunk_loss - (
                    (surrogate[row] * mask[row]).sum()
                    / denominator
                    / prompt_denominator
                )
            chunk_loss.backward()
            total_loss_value += float(chunk_loss.detach().cpu().item())

            valid = scored.response_mask
            ratio_values.append(ratio.detach()[valid].cpu())
            approx_kl_values.append((old_pad - current).detach()[valid].cpu())
            clip_values.append(
                ((ratio < 1.0 - clip) | (ratio > 1.0 + clip))
                .to(dtype=torch.float32)
                .detach()[valid]
                .cpu()
            )
            abs_surrogate_values.append(surrogate.detach().abs()[valid].cpu())
            del scored, current, old_pad, adv_pad, ratio, clipped_ratio, surrogate, chunk_loss

        ratio_flat = torch.cat(ratio_values) if ratio_values else torch.ones(1)
        kl_flat = torch.cat(approx_kl_values) if approx_kl_values else torch.zeros(1)
        clip_flat = torch.cat(clip_values) if clip_values else torch.zeros(1)
        abs_flat = torch.cat(abs_surrogate_values) if abs_surrogate_values else torch.zeros(1)
        return {
            "loss": total_loss_value,
            "ratio_mean": float(ratio_flat.mean().item()),
            "ratio_std": float(ratio_flat.std(unbiased=False).item()),
            "approx_kl": float(kl_flat.mean().item()),
            "clip_fraction": float(clip_flat.mean().item()),
            "abs_surrogate": float(abs_flat.mean().item()),
            "tokens": float(ratio_flat.numel()),
        }

    def _run_topd_global_step(
        self,
        examples: Sequence[TaskExample],
        meter: RoutingMeter,
        outer_step: int,
    ) -> dict[str, Any]:
        """Run TOP-D's frozen-policy rollout batch and internal minibatch updates.

        One outer training step is one TOP-D global iteration.  The same frozen
        behavior-policy batch is partitioned into prompt-level minibatches and
        reused for one or more off-policy epochs.  This makes the PPO ratio and
        clipping operator operative without retaining full-vocabulary logits.
        """
        cfg = dict(self.cfg.get("topd", {}))
        buffer = self._build_topd_rollout_buffer(examples, meter)
        lr = self._lr(outer_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if buffer is None or not buffer.active_prompt_indices:
            self.optimizer.zero_grad(set_to_none=True)
            return {
                "losses": [],
                "grad_norm_preclip": 0.0,
                "grad_norm_postclip": 0.0,
                "grad_clip_scale": 1.0,
                "all_gradients_finite": True,
                "nonfinite_updates": 0,
                "completed_updates": 0,
                "lr": lr,
            }

        minibatch_prompts = max(1, int(cfg.get("minibatch_prompts", 1)))
        off_policy_epochs = max(1, int(cfg.get("off_policy_epochs", 1)))
        shuffle = bool(cfg.get("shuffle_minibatches", True))
        max_grad_norm = float(self.cfg["train"]["max_grad_norm"])
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]

        losses: list[float] = []
        grad_pre_values: list[float] = []
        grad_post_values: list[float] = []
        clip_scale_values: list[float] = []
        ratio_means: list[float] = []
        ratio_stds: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        abs_surrogates: list[float] = []
        completed_updates = 0
        nonfinite_updates = 0

        for _ in range(off_policy_epochs):
            prompt_order = list(buffer.active_prompt_indices)
            if shuffle:
                self.rng.shuffle(prompt_order)
            for start in range(0, len(prompt_order), minibatch_prompts):
                prompt_batch = prompt_order[start : start + minibatch_prompts]
                self.optimizer.zero_grad(set_to_none=True)
                diagnostics = self._backward_topd_internal_minibatch(buffer, prompt_batch)
                if diagnostics["tokens"] <= 0:
                    continue

                grad_pre_tensor = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                grad_pre = float(grad_pre_tensor)
                finite = math.isfinite(grad_pre)
                grad_post = min(grad_pre, max_grad_norm) if finite else float("nan")
                clip_scale = (
                    min(1.0, max_grad_norm / max(grad_pre, 1.0e-12))
                    if finite
                    else 0.0
                )
                grad_pre_values.append(grad_pre)
                grad_post_values.append(grad_post)
                clip_scale_values.append(clip_scale)
                losses.append(float(diagnostics["loss"]))
                ratio_means.append(float(diagnostics["ratio_mean"]))
                ratio_stds.append(float(diagnostics["ratio_std"]))
                approx_kls.append(float(diagnostics["approx_kl"]))
                clip_fractions.append(float(diagnostics["clip_fraction"]))
                abs_surrogates.append(float(diagnostics["abs_surrogate"]))

                if finite:
                    self.optimizer.step()
                    self.ema.update(self.model)
                    self.state.optimizer_updates += 1
                    completed_updates += 1
                else:
                    nonfinite_updates += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    self.logger.warning(
                        "Skipping TOP-D internal update at outer step %d because gradients are non-finite.",
                        outer_step,
                    )

        mean = lambda values, default=0.0: sum(values) / len(values) if values else default
        meter.add_extra("topd_minibatch_prompts", minibatch_prompts)
        meter.add_extra("topd_off_policy_epochs", off_policy_epochs)
        meter.add_extra("topd_internal_updates", completed_updates)
        meter.add_extra("topd_nonfinite_internal_updates", nonfinite_updates)
        meter.add_extra("topd_ratio_mean", mean(ratio_means, 1.0))
        meter.add_extra("topd_ratio_std", mean(ratio_stds))
        meter.add_extra("topd_approx_kl", mean(approx_kls))
        meter.add_extra("topd_clip_fraction", mean(clip_fractions))
        meter.add_extra("topd_abs_surrogate", mean(abs_surrogates))
        meter.add_extra("topd_grad_pre_max", max(grad_pre_values) if grad_pre_values else 0.0)

        return {
            "losses": losses,
            "grad_norm_preclip": mean(grad_pre_values),
            "grad_norm_postclip": mean(grad_post_values),
            "grad_clip_scale": mean(clip_scale_values, 1.0),
            "all_gradients_finite": nonfinite_updates == 0,
            "nonfinite_updates": nonfinite_updates,
            "completed_updates": completed_updates,
            "lr": lr,
        }

    def _backward_topd_microbatch(
        self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float
    ) -> list[float]:
        raise RuntimeError(
            "TOP-D must be executed through _run_topd_global_step so that one frozen "
            "behavior-policy batch is reused across sequential internal minibatch updates."
        )

    def _tr_opd_guided_entries(
        self, examples: Sequence[TaskExample]
    ) -> list[tuple[int, TaskExample, str, GeneratedRollout, ContextRecord, int]]:
        cfg = dict(self.cfg.get("tropd", {}))
        maximum = int(self.cfg["generation"]["max_new_tokens"])
        maximum_step = max(int(self.cfg["train"]["max_steps"]) - 1, 1)
        progress = min(max(self.state.optimizer_step / maximum_step, 0.0), 1.0)
        prefix_cap = int(cfg.get("max_prefix_tokens", maximum))
        scheduled_cap = int(round(prefix_cap * 0.5 * (1.0 + math.cos(math.pi * progress))))
        student_prompts = [render_chat(self.tokenizer, student_messages(example)) for example in examples]
        provisional = [self.context_builder.build(example, "", self.adapter, rng=self.rng) for example in examples]
        teacher_prompts = [self._teacher_prompt(example, context) for example, context in zip(examples, provisional)]
        errors = generation_budget_errors(
            self.tokenizer,
            teacher_prompts,
            max_new_tokens=maximum,
            max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
        )
        valid = [index for index, error in enumerate(errors) if error is None]
        for index, error in enumerate(errors):
            if error is not None:
                self._log_skip(examples[index], error)
        if not valid:
            return []
        with self.ema.swap_into(self.model):
            drafts = generate_rollout_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[teacher_prompts[index] for index in valid],
                generation_cfg=self.cfg["generation"],
                sequence_cfg=self.cfg["sequence"],
                do_sample=True,
            )
        self.state.generated_tokens += sum(int(draft.response_ids.numel()) for draft in drafts)
        output: list[tuple[int, TaskExample, str, GeneratedRollout, ContextRecord, int]] = []
        for index, draft in zip(valid, drafts):
            prefix_length = min(scheduled_cap, max(int(draft.response_ids.numel()) - 1, 0), maximum - 1)
            prefix = draft.response_ids[:prefix_length]
            rollout = generate_from_forced_prefix(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=student_prompts[index],
                prefix_ids=prefix,
                generation_cfg=self.cfg["generation"],
                sequence_cfg=self.cfg["sequence"],
                do_sample=True,
            )
            self.state.generated_tokens += max(int(rollout.response_ids.numel()) - prefix_length, 0)
            if not self._keep_baseline_rollout(examples[index], rollout):
                continue
            context = self.context_builder.build(examples[index], rollout.response_text, self.adapter, rng=self.rng)
            teacher_prompt = self._teacher_prompt(examples[index], context)
            pair_errors = scoring_budget_errors(
                self.tokenizer,
                [student_prompts[index], teacher_prompt],
                [rollout.response_ids, rollout.response_ids],
                max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
            )
            if any(error is not None for error in pair_errors):
                self._log_skip(examples[index], next(error for error in pair_errors if error is not None))
                continue
            output.append((index, examples[index], student_prompts[index], rollout, context, prefix_length))
        return output

    def _backward_tropd_microbatch(
        self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float
    ) -> list[float]:
        """TrOPD trust/outlier objectives plus cosine-annealed teacher-prefix guidance."""
        cfg = dict(self.cfg.get("tropd", {}))
        group_size = max(1, int(cfg.get("group_size", 4)))
        chunk_size = max(1, int(cfg.get("score_chunk_size", 1)))
        top_k = max(1, int(cfg.get("top_k", 64)))
        standard = self._group_rollout_entries(examples, group_size=group_size)
        entries: list[tuple[int, TaskExample, str, GeneratedRollout, ContextRecord, int]] = [
            (item[0], item[1], item[2], item[3], item[4], 0) for item in standard
        ]
        if bool(cfg.get("off_policy_guidance", True)):
            entries.extend(self._tr_opd_guided_entries(examples))
        if not entries:
            return []
        prompts = [item[2] for item in entries]
        ids = [item[3].response_ids for item in entries]
        teacher_prompts = [self._teacher_prompt(item[1], item[4]) for item in entries]
        old_logp = self._score_gathered_logp_streamed(
            prompts=prompts, response_ids=ids, use_ema=False, chunk_size=chunk_size
        )
        teacher_sampled, teacher_top_values, teacher_top_indices = self._score_teacher_topk_streamed(
            prompts=teacher_prompts,
            response_ids=ids,
            top_k=top_k,
            chunk_size=chunk_size,
        )
        trust_masks: list[torch.Tensor] = []
        prefix_masks: list[torch.Tensor] = []
        for index, entry in enumerate(entries):
            length = int(ids[index].numel())
            prefix_length = min(int(entry[5]), length)
            acceptance = torch.exp(teacher_sampled[index] - old_logp[index]).clamp(max=1.0)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.rng.randrange(0, 2**31 - 1))
            trust = torch.rand(length, generator=generator).lt(acceptance.cpu())
            prefix_mask = torch.zeros(length, dtype=torch.bool)
            if prefix_length:
                prefix_mask[:prefix_length] = True
                trust[:prefix_length] = False
            trust_masks.append(trust)
            prefix_masks.append(prefix_mask)
        token_counts = [sum(int(ids[index].numel()) for index, item in enumerate(entries) if item[0] == p) for p in range(len(examples))]
        raw_prompt = [0.0 for _ in examples]
        beta = float(cfg.get("guidance_beta", 0.001))
        trust_total = 0
        on_policy_total = 0
        self.model.train()
        for start in range(0, len(entries), chunk_size):
            stop = min(start + chunk_size, len(entries))
            chunk_ids = ids[start:stop]
            scored = response_logits_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=prompts[start:stop],
                response_ids=chunk_ids,
                max_context_tokens=int(self.cfg["sequence"]["max_context_tokens"]),
                requires_grad=True,
            )
            logp = torch.log_softmax(scored.logits.float(), dim=-1)
            current_sampled = gathered_token_log_probs(scored.logits, chunk_ids)
            width = int(current_sampled.shape[1])
            old_pad = self._pad_token_lists(old_logp[start:stop], width, device=current_sampled.device)
            teacher_sampled_pad = self._pad_token_lists(teacher_sampled[start:stop], width, device=current_sampled.device)
            trust_pad = self._pad_token_lists(
                [mask.float() for mask in trust_masks[start:stop]], width, device=current_sampled.device
            )
            prefix_pad = self._pad_token_lists(
                [mask.float() for mask in prefix_masks[start:stop]], width, device=current_sampled.device
            )
            valid_mask = scored.response_mask.to(dtype=torch.float32)
            continuation_mask = valid_mask * (1.0 - prefix_pad)
            outlier_pad = continuation_mask * (1.0 - trust_pad)
            k1 = k1_policy_gradient_tokens(current_sampled, old_pad, teacher_sampled_pad)
            max_k = max(int(value.shape[-1]) for value in teacher_top_values[start:stop])
            index_pad = torch.zeros((stop - start, width, max_k), dtype=torch.long, device=logp.device)
            value_pad = torch.zeros((stop - start, width, max_k), dtype=torch.float32, device=logp.device)
            support_mask = torch.zeros((stop - start, width, max_k), dtype=torch.float32, device=logp.device)
            for row, (values, indices) in enumerate(zip(teacher_top_values[start:stop], teacher_top_indices[start:stop])):
                time = min(int(values.shape[0]), width)
                k = int(values.shape[1])
                value_pad[row, :time, :k] = values[:time].to(logp.device)
                index_pad[row, :time, :k] = indices[:time].to(logp.device)
                support_mask[row, :time, :k] = 1.0
            student_selected = torch.gather(logp, dim=-1, index=index_pad)
            teacher_prob = value_pad.exp() * support_mask
            fkl = (teacher_prob * (value_pad - student_selected) * support_mask).sum(dim=-1)
            prefix_loss = -beta * current_sampled * prefix_pad
            token_loss = k1 * trust_pad + fkl * outlier_pad + prefix_loss
            chunk_loss = torch.zeros((), dtype=torch.float32, device=logp.device)
            for row, entry in enumerate(entries[start:stop]):
                denominator = max(token_counts[entry[0]], 1)
                value = (token_loss[row] * valid_mask[row]).sum() / float(denominator)
                chunk_loss = chunk_loss + value
                raw_prompt[entry[0]] += float(value.detach().cpu().item())
                trust_total += int(trust_pad[row].sum().detach().cpu().item())
                on_policy_total += int(continuation_mask[row].sum().detach().cpu().item())
            (chunk_loss * scale).backward()
            del scored, logp, current_sampled, old_pad, teacher_sampled_pad, trust_pad, prefix_pad
            del valid_mask, continuation_mask, outlier_pad, k1, index_pad, value_pad, support_mask
            del student_selected, teacher_prob, fkl, prefix_loss, token_loss, chunk_loss
        guided = sum(1 for entry in entries if entry[5] > 0)
        average_prefix = sum(entry[5] for entry in entries) / max(len(entries), 1)
        meter.add_extra("tropd_group_size", group_size)
        meter.add_extra("tropd_top_k", top_k)
        meter.add_extra("tropd_trust_fraction", trust_total / max(on_policy_total, 1))
        meter.add_extra("tropd_guided_rollouts", guided)
        meter.add_extra("tropd_average_prefix_tokens", average_prefix)
        meter.add_extra("tropd_guidance_beta", beta)
        return raw_prompt

    def _backward_microbatch(self, examples: Sequence[TaskExample], meter: RoutingMeter, scale: float) -> list[float]:
        method = str(self.cfg["experiment"]["method"])
        if method == "srpo":
            return self._backward_srpo_microbatch(examples, meter, scale)
        if method == "on_policy_sft":
            return self._backward_on_policy_sft_microbatch(examples, meter, scale)
        if method == "veto":
            return self._backward_veto_microbatch(examples, meter, scale)
        if method == "demopsd":
            return self._backward_demopsd_microbatch(examples, meter, scale)
        if method == "scope":
            return self._backward_scope_microbatch(examples, meter, scale)
        if method == "topd":
            return self._backward_topd_microbatch(examples, meter, scale)
        if method == "tropd":
            return self._backward_tropd_microbatch(examples, meter, scale)

        prepared = self._filter_scoring_budget(self._prepare_microbatch(examples, meter=meter))
        if not prepared:
            return []

        # Failure-only and success-only controls preserve the historical mean
        # scale over the retained branch. Main FIRE routes every valid rollout,
        # so no eligibility correction is needed.
        loss_scale = scale
        routing_cfg = dict(self.cfg.get("correctness_routing", {}))
        if self._correctness_routing_enabled() and bool(routing_cfg.get("normalize_over_eligible", True)):
            correct_action, incorrect_action = self._correctness_actions()
            if "skip" in {correct_action, incorrect_action}:
                loss_scale *= len(examples) / max(len(prepared), 1)

        sequence_cfg = self.cfg["sequence"]
        sft_items = [item for item in prepared if item.action == "sft"]
        distill_items = [item for item in prepared if item.action == "distill"]
        raw_losses: list[float] = []

        # Correct branch: verified self-imitation. Main FIRE projects the
        # realized hard-label logit gradient; controls may retain ordinary SFT.
        # No privileged context or EMA teacher is evaluated for these samples.
        if sft_items:
            self.model.train()
            sft_scored = response_logits_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[item.student_prompt for item in sft_items],
                response_ids=[item.rollout.response_ids for item in sft_items],
                max_context_tokens=int(sequence_cfg["max_context_tokens"]),
                requires_grad=True,
            )
            fire_cfg = dict(self.cfg.get("fire", {}))
            use_projected_success = (
                normalize_method(method) in {"fire", "fire_attribution_only", "fisher_trust_full_context", "hard_gradient_excision"}
                and bool(fire_cfg.get("project_correct_sft", False))
            )
            if use_projected_success:
                step_scale = self._fire_step_scale()
                sft_losses, token_trust, token_before, token_after, token_radius = (
                    fisher_step_projected_sft_loss(
                        sft_scored.logits,
                        [item.rollout.response_ids for item in sft_items],
                        step_scale=step_scale,
                        response_mask=sft_scored.response_mask,
                        eps=float(fire_cfg.get("eps", 1.0e-8)),
                    )
                )
                mask_f = sft_scored.response_mask.to(dtype=token_trust.dtype)
                denom = mask_f.sum().clamp_min(1.0)
                denom_cpu = denom.detach().cpu()
                meter.add_extra(
                    "correct_sft_trust",
                    float((token_trust * mask_f).sum().detach().cpu() / denom_cpu),
                )
                meter.add_extra(
                    "correct_sft_logit_grad_norm_before",
                    float((token_before * mask_f).sum().detach().cpu() / denom_cpu),
                )
                meter.add_extra(
                    "correct_sft_logit_grad_norm_after",
                    float((token_after * mask_f).sum().detach().cpu() / denom_cpu),
                )
                meter.add_extra(
                    "correct_sft_effective_radius",
                    float((token_radius * mask_f).sum().detach().cpu() / denom_cpu),
                )
                meter.add_extra("fire_step_scale", step_scale)
                del token_trust, token_before, token_after, token_radius
            else:
                sft_losses = supervised_nll_loss(
                    sft_scored.logits,
                    [item.rollout.response_ids for item in sft_items],
                    sft_scored.response_mask,
                )
            if sft_losses.ndim != 1 or sft_losses.shape[0] != len(sft_items):
                raise RuntimeError("On-policy SFT must return one scalar per correct rollout.")
            (sft_losses.sum() * loss_scale).backward()
            raw_losses.extend(float(value) for value in sft_losses.detach().cpu().tolist())
            meter.add_extra("correct_rollout_sft_examples", len(sft_items))
            del sft_scored, sft_losses

        # Incorrect branch: revised FIRE or the configured distillation control.
        if distill_items:
            if any(item.context is None for item in distill_items):
                raise RuntimeError("Distillation branch contains an item without context.")
            try:
                routed = route_targets_batch(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    ema=self.ema,
                    examples=[item.example for item in distill_items],
                    response_ids=[item.rollout.response_ids for item in distill_items],
                    contexts=[item.context for item in distill_items if item.context is not None],
                    method=method,
                    max_context_tokens=int(sequence_cfg["max_context_tokens"]),
                    fire_cfg=self.cfg.get("fire", {}),
                    step_scale=self._fire_step_scale(),
                    candidate_chunk_blocks=int(self.cfg.get("routing", {}).get("candidate_chunk_blocks", 1)),
                    collect_diagnostics=normalize_method(method) in {
                        "fire",
                        "fire_attribution_only",
                        "fisher_trust_full_context",
                        "hard_gradient_excision",
                    },
                )
            except SequenceBudgetError as exc:
                for item in distill_items:
                    self._log_skip(item.example, exc)
                return raw_losses

            self.model.train()
            student_scored = response_logits_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[item.student_prompt for item in distill_items],
                response_ids=[item.rollout.response_ids for item in distill_items],
                max_context_tokens=int(sequence_cfg["max_context_tokens"]),
                requires_grad=True,
            )
            if not torch.equal(student_scored.response_mask, routed.response_mask):
                raise RuntimeError("Student autograd mask disagrees with routed EMA target mask.")
            per_example_loss = self._loss(
                student_scored.logits, routed.target_log_probs, student_scored.response_mask
            )
            if per_example_loss.ndim != 1 or per_example_loss.shape[0] != len(distill_items):
                raise RuntimeError("Batched loss must return one scalar per distillation example.")
            (per_example_loss.sum() * loss_scale).backward()
            raw_losses.extend(float(value) for value in per_example_loss.detach().cpu().tolist())
            for diagnostics in routed.diagnostics:
                meter.update(diagnostics)
            meter.add_extra("incorrect_distillation_examples", len(distill_items))
            del student_scored, routed, per_example_loss

        return raw_losses

    def _save_checkpoint(self, name: str) -> None:
        destination = self.run_dir / "checkpoints" / name
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)
        write_json(destination / "ema_metadata.json", {"alpha": self.ema.alpha, "optimizer_step": self.state.optimizer_step})

    def _run_evaluation(self, step: int, history: list[dict[str, Any]]) -> None:
        self.model.eval()
        metrics = evaluate(
            cfg=self.cfg,
            model=self.model,
            tokenizer=self.tokenizer,
            ema=self.ema,
            adapter=self.adapter,
            examples=self.eval_examples,
            context_builder=self.context_builder,
            step=step,
        )
        metrics["cumulative_train_seconds"] = float(self.state.training_seconds)
        metrics["cumulative_generated_tokens"] = int(self.state.generated_tokens)
        self.logger.info(
            "EVAL step %d batch=%d task-acc=%.3f±%.3f n=%d train-time=%.1fs generated-tokens=%d "
            "budget(truncated=%.1f%%, no-final-answer=%.1f%%, mean-tokens=%.0f, saturation=%.0f%%) %s",
            step,
            metrics.get("eval_batch_size", 1),
            metrics["task_accuracy"],
            metrics["task_accuracy_se"],
            metrics["task_examples"],
            metrics["cumulative_train_seconds"],
            metrics["cumulative_generated_tokens"],
            100.0 * float(metrics.get("rollout_truncated_rate", 0.0)),
            100.0 * float(metrics.get("rollout_missing_final_answer_rate", 0.0)),
            float(metrics.get("rollout_mean_response_tokens", 0.0)),
            100.0 * float(metrics.get("rollout_length_saturation", 0.0)),
            format_routing(metrics, "routing"),
        )
        category_scores = metrics.get("task_accuracy_by_category", {})
        category_counts = metrics.get("task_examples_by_category", {})
        if category_scores:
            rendered_categories = ", ".join(
                f"{category}={float(score):.3f}(n={int(category_counts.get(category, 0))})"
                for category, score in category_scores.items()
            )
            self.logger.info("EVAL-CATEGORIES step %d %s", step, rendered_categories)
        history.append({"kind": "eval", **metrics})
        write_json(self.run_dir / "latest_eval.json", metrics)
        if bool(self.cfg["output"].get("save_checkpoints", False)):
            self._save_checkpoint(f"step_{step}")

    def train(self) -> dict[str, Any]:
        train_cfg = self.cfg["train"]
        max_steps = int(train_cfg["max_steps"])
        micro_batch_size = int(train_cfg["micro_batch_size"])
        accumulation = int(train_cfg["gradient_accumulation_steps"])
        effective_batch = micro_batch_size * accumulation
        method = normalize_method(str(self.cfg["experiment"]["method"]))
        self.logger.info(
            "=== experiment=%s seed=%s method=%s context=%s model=%s data=%s effective-batch=%d "
            "(vectorized-micro=%d x accumulation=%d) context-order=%s stable-on-policy-sd ===",
            self.cfg["experiment"]["name"],
            train_cfg["seed"],
            self.cfg["experiment"]["method"],
            self.context_builder.context_type,
            self.cfg["model"]["name_or_path"],
            self.cfg["data"]["adapter"],
            effective_batch,
            micro_batch_size,
            accumulation,
            self.context_builder.order_mode,
        )
        if method == "topd":
            topd_cfg = dict(self.cfg.get("topd", {}))
            self.logger.info(
                "TOP-D global iteration: behavior-prompts=%d group-size=%d minibatch-prompts=%d "
                "off-policy-epochs=%d (sequential optimizer updates per rollout batch)",
                effective_batch,
                int(topd_cfg.get("group_size", 8)),
                int(topd_cfg.get("minibatch_prompts", 1)),
                int(topd_cfg.get("off_policy_epochs", 1)),
            )
        if not self.train_examples:
            raise RuntimeError("Training split is empty after dataset loading.")
        order = list(range(len(self.train_examples)))
        self.rng.shuffle(order)
        cursor = 0
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        if bool(self.cfg["eval"].get("at_start", True)):
            self._run_evaluation(0, history)

        for step in range(1, max_steps + 1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_started = time.perf_counter()
            generated_before = self.state.generated_tokens
            meter = RoutingMeter()
            step_losses: list[float] = []

            if method == "topd":
                global_examples: list[TaskExample] = []
                for _ in range(effective_batch):
                    if cursor >= len(order):
                        self.rng.shuffle(order)
                        cursor = 0
                    global_examples.append(self.train_examples[order[cursor]])
                    cursor += 1
                    self.state.seen_examples += 1
                topd_result = self._run_topd_global_step(global_examples, meter, step)
                step_losses = list(topd_result["losses"])
                grad_norm_preclip = float(topd_result["grad_norm_preclip"])
                grad_norm_postclip = float(topd_result["grad_norm_postclip"])
                grad_clip_scale = float(topd_result["grad_clip_scale"])
                gradients_finite = bool(topd_result["all_gradients_finite"])
                has_training_signal = bool(step_losses)
                lr = float(topd_result["lr"])
                nonfinite_update_count = int(topd_result["nonfinite_updates"])
            else:
                self.optimizer.zero_grad(set_to_none=True)
                scale = 1.0 / effective_batch
                sft_normalization = (
                    _OnPolicySFTNormalization()
                    if method == "on_policy_sft" and accumulation > 1 else None
                )
                for _ in range(accumulation):
                    micro_examples: list[TaskExample] = []
                    for _ in range(micro_batch_size):
                        if cursor >= len(order):
                            self.rng.shuffle(order)
                            cursor = 0
                        micro_examples.append(self.train_examples[order[cursor]])
                        cursor += 1
                        self.state.seen_examples += 1
                    if sft_normalization is not None:
                        self._backward_on_policy_sft_microbatch(
                            micro_examples, meter, scale, normalization=sft_normalization
                        )
                    else:
                        step_losses.extend(self._backward_microbatch(micro_examples, meter, scale))
                if sft_normalization is not None:
                    step_losses = sft_normalization.losses

                max_grad_norm = float(train_cfg["max_grad_norm"])
                grad_norm_preclip_tensor = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                grad_norm_preclip = float(grad_norm_preclip_tensor)
                gradients_finite = math.isfinite(grad_norm_preclip)
                has_training_signal = bool(step_losses)
                grad_norm_postclip = (
                    min(grad_norm_preclip, max_grad_norm) if gradients_finite else float("nan")
                )
                grad_clip_scale = (
                    min(1.0, max_grad_norm / max(grad_norm_preclip, 1.0e-12))
                    if gradients_finite
                    else 0.0
                )
                lr = self._lr(step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                nonfinite_update_count = 0
                if has_training_signal and gradients_finite:
                    self.optimizer.step()
                    self.ema.update(self.model)
                    self.state.optimizer_updates += 1
                elif has_training_signal:
                    nonfinite_update_count = 1
                    self.optimizer.zero_grad(set_to_none=True)
                    self.logger.warning(
                        "Skipping optimizer/EMA update at step %d because gradients are non-finite.",
                        step,
                    )
                else:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.logger.info(
                        "Skipping optimizer/EMA update at step %d because routing produced no trainable rollout.",
                        step,
                    )

            meter.add_extra(
                "grad_norm_preclip", grad_norm_preclip if math.isfinite(grad_norm_preclip) else None
            )
            meter.add_extra(
                "grad_norm_postclip", grad_norm_postclip if math.isfinite(grad_norm_postclip) else None
            )
            meter.add_extra("grad_clip_scale", grad_clip_scale)
            meter.add_extra("nonfinite_update_skipped", nonfinite_update_count)
            meter.add_extra("no_eligible_update", 0.0 if has_training_signal else 1.0)
            meter.add_extra("optimizer_updates_cumulative", self.state.optimizer_updates)
            summary = meter.summary("routing")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_train_seconds = time.perf_counter() - step_started
            self.state.training_seconds += step_train_seconds
            step_generated_tokens = self.state.generated_tokens - generated_before
            self.state.optimizer_step = step
            if step_losses:
                self.state.loss_sum += sum(step_losses)
                self.state.loss_count += len(step_losses)

            if step % int(train_cfg["log_every"]) == 0 or step == 1:
                avg_loss = sum(step_losses) / max(len(step_losses), 1)
                self.logger.info(
                    "step %d/%d loss=%.5f grad-pre=%.3f grad-post=%.3f clip-scale=%.3f lr=%.3e "
                    "optimizer-updates=%d train-time=%.1fs (+%.1fs) generated-tokens=%d (+%d) %s "
                    "skipped-over-budget=%d dropped-unfinished=%d",
                    step,
                    max_steps,
                    avg_loss,
                    grad_norm_preclip,
                    grad_norm_postclip,
                    grad_clip_scale,
                    lr,
                    self.state.optimizer_updates,
                    self.state.training_seconds,
                    step_train_seconds,
                    self.state.generated_tokens,
                    step_generated_tokens,
                    format_routing(summary, "routing"),
                    self.state.skipped_over_budget,
                    self.state.dropped_unfinished,
                )
                history.append(
                    {
                        "kind": "train",
                        "step": step,
                        "loss": avg_loss,
                        "grad_norm": grad_norm_preclip,
                        "grad_norm_preclip": grad_norm_preclip,
                        "grad_norm_postclip": grad_norm_postclip,
                        "grad_clip_scale": grad_clip_scale,
                        "update_skipped_nonfinite": nonfinite_update_count > 0,
                        "nonfinite_update_count": nonfinite_update_count,
                        "update_skipped_no_eligible": not has_training_signal,
                        "lr": lr,
                        "optimizer_updates": self.state.optimizer_updates,
                        "step_train_seconds": step_train_seconds,
                        "cumulative_train_seconds": self.state.training_seconds,
                        "step_generated_tokens": step_generated_tokens,
                        "cumulative_generated_tokens": self.state.generated_tokens,
                        "generated_tokens_per_second": (
                            step_generated_tokens / step_train_seconds
                            if step_train_seconds > 0
                            else None
                        ),
                        **summary,
                    }
                )

            if step % int(self.cfg["eval"]["every_steps"]) == 0 or step == max_steps:
                self._run_evaluation(step, history)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_eval = next((entry for entry in reversed(history) if entry["kind"] == "eval"), None)
        result = {
            "status": "completed",
            "elapsed_seconds": time.perf_counter() - started,
            "training_seconds": self.state.training_seconds,
            "generated_tokens": self.state.generated_tokens,
            "state": self.state.__dict__,
            "effective_batch_size": effective_batch,
            "microbatch_execution": (
                "topd_frozen_behavior_internal_minibatches" if method == "topd" else "vectorized"
            ),
            "final_eval": final_eval,
            "history": history,
        }
        write_json(self.run_dir / "metrics.json", result)
        return result
