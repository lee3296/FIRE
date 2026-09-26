"""Full reverse-KL, top-K approximation, and supervised fine-tuning losses."""
from __future__ import annotations

import math

import torch


def _masked_token_reduction(token_values: torch.Tensor, response_mask: torch.Tensor | None) -> torch.Tensor:
    """Average valid tokens per example, preserving old per-example batch weighting.

    With no mask, this is the original global token mean for a singleton path.
    With a ``[batch, time]`` mask, it returns ``[batch]`` so callers can sum the
    per-example losses and divide by the configured effective batch size.  That
    exactly matches the prior sequential accumulation semantics even when rollout
    lengths vary within a true vectorized microbatch.
    """
    if response_mask is None:
        return token_values.mean()
    if token_values.shape != response_mask.shape:
        raise ValueError(
            "response_mask must match token loss shape; "
            f"got token_values={tuple(token_values.shape)} mask={tuple(response_mask.shape)}"
        )
    mask = response_mask.to(dtype=token_values.dtype)
    if token_values.ndim == 1:
        return (token_values * mask).sum() / mask.sum().clamp_min(1.0)
    if token_values.ndim != 2:
        raise ValueError("Masked loss expects token values shaped [batch, time].")
    return (token_values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def full_reverse_kl_tokens(student_logits: torch.Tensor, target_log_probs: torch.Tensor) -> torch.Tensor:
    """Tokenwise KL(student || detached target), without reducing over time."""
    log_p = torch.log_softmax(student_logits.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - target_log_probs.float())).sum(dim=-1)


def full_reverse_kl(
    student_logits: torch.Tensor,
    target_log_probs: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tokenwise KL(student || detached target), in float32.

    A masked batched call returns one mean-token loss per example; the singleton
    unmasked call retains the historical scalar result.
    """
    return _masked_token_reduction(full_reverse_kl_tokens(student_logits, target_log_probs), response_mask)


def _first_occurrence_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Boolean mask retaining the first appearance of each token per row."""
    # tokens: [..., K]. K<=192 by default; the O(K^2) dedupe is intentionally tiny.
    equals = tokens.unsqueeze(-1).eq(tokens.unsqueeze(-2))
    previous = torch.tril(
        torch.ones(tokens.shape[-1], tokens.shape[-1], dtype=torch.bool, device=tokens.device), diagonal=-1
    )
    seen_before = (equals & previous).any(dim=-1)
    return ~seen_before


def topk_union_reverse_kl_tokens(
    student_logits: torch.Tensor,
    target_log_probs: torch.Tensor,
    student_top_k: int,
    target_top_k: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Tokenwise reverse KL on TopK(student) union TopK(target), plus one exact mass tail bucket."""
    log_p = torch.log_softmax(student_logits.float(), dim=-1)
    log_q = target_log_probs.float()
    vocab = log_p.shape[-1]
    k_p = min(int(student_top_k), vocab)
    k_q = min(int(target_top_k), vocab)
    student_indices = torch.topk(log_p, k=k_p, dim=-1).indices
    target_indices = torch.topk(log_q, k=k_q, dim=-1).indices
    union = torch.cat([student_indices, target_indices], dim=-1)
    keep = _first_occurrence_mask(union)

    selected_log_p = torch.gather(log_p, dim=-1, index=union)
    selected_log_q = torch.gather(log_q, dim=-1, index=union)
    selected_p = selected_log_p.exp() * keep
    selected_q = selected_log_q.exp() * keep
    explicit = (selected_p * (selected_log_p - selected_log_q) * keep).sum(dim=-1)
    tail_p = (1.0 - selected_p.sum(dim=-1)).clamp_min(eps)
    tail_q = (1.0 - selected_q.sum(dim=-1)).clamp_min(eps)
    tail = tail_p * (tail_p.log() - tail_q.log())
    return explicit + tail


def topk_union_reverse_kl(
    student_logits: torch.Tensor,
    target_log_probs: torch.Tensor,
    student_top_k: int,
    target_top_k: int,
    eps: float = 1e-8,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reverse KL on TopK(student) union TopK(target), plus one exact mass tail bucket."""
    return _masked_token_reduction(
        topk_union_reverse_kl_tokens(student_logits, target_log_probs, student_top_k, target_top_k, eps),
        response_mask,
    )


def _padded_target_ids(target_ids: list[torch.Tensor], width: int, *, device: torch.device) -> torch.Tensor:
    padded = torch.zeros((len(target_ids), width), dtype=torch.long, device=device)
    for row, ids in enumerate(target_ids):
        ids = ids.detach().to(device=device, dtype=torch.long).reshape(-1)
        length = min(int(ids.numel()), width)
        if length > 0:
            padded[row, :length] = ids[:length]
    return padded


def gathered_token_log_probs(logits: torch.Tensor, target_ids: list[torch.Tensor]) -> torch.Tensor:
    """Return log p(target token_t | prefix) for padded target sequences."""
    if logits.ndim != 3:
        raise ValueError("logits must be shaped [batch, time, vocab]")
    if len(target_ids) != logits.shape[0]:
        raise ValueError("target_ids length must match logits batch size")
    target = _padded_target_ids(target_ids, int(logits.shape[1]), device=logits.device)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


def supervised_nll_tokens(logits: torch.Tensor, target_ids: list[torch.Tensor]) -> torch.Tensor:
    """Tokenwise negative log-likelihood for supervised fine-tuning."""
    return -gathered_token_log_probs(logits, target_ids)


def supervised_nll_loss(
    logits: torch.Tensor,
    target_ids: list[torch.Tensor],
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _masked_token_reduction(supervised_nll_tokens(logits, target_ids), response_mask)


def fisher_step_projected_sft_tokens(
    logits: torch.Tensor,
    target_ids: list[torch.Tensor],
    *,
    step_scale: float = 1.0,
    response_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenwise verified self-imitation with an exact Fisher step bound.

    For a fixed sampled token y, hard-label cross entropy has logit gradient
    ``p - one_hot(y)``.  Under on-policy sampling, the expected squared norm of
    that gradient is ``tr F(p) = 1 - ||p||_2^2``.  We therefore detach the
    student distribution and scale only unusually large realized gradients so
    that

        ||grad_z L_token||_2 <= sqrt(tr F(p)) / step_scale.

    ``step_scale`` is at least one and is derived from the current learning rate
    relative to the experiment's nominal tuned rate.  No extra model forward is
    required.  The returned tensors are (weighted_nll, trust, before_norm,
    after_norm, effective_radius), all shaped [batch, time].
    """
    if logits.ndim != 3:
        raise ValueError("logits must be shaped [batch, time, vocab]")
    if len(target_ids) != logits.shape[0]:
        raise ValueError("target_ids length must match logits batch size")
    if float(step_scale) < 1.0:
        raise ValueError("step_scale must be at least 1")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")

    width = int(logits.shape[1])
    target = _padded_target_ids(target_ids, width, device=logits.device)
    scores = logits.float()
    log_normalizer = torch.logsumexp(scores, dim=-1)
    target_scores = torch.gather(scores, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    nll = log_normalizer - target_scores

    # The two required softmax statistics can be obtained with vocabulary
    # reductions only.  This avoids materializing an additional full [B,T,V]
    # probability tensor on the inexpensive success branch.
    with torch.no_grad():
        detached_scores = scores.detach()
        detached_log_normalizer = log_normalizer.detach()
        p_target = torch.exp(target_scores.detach() - detached_log_normalizer)
        log_p_sq_sum = (
            torch.logsumexp(2.0 * detached_scores, dim=-1)
            - 2.0 * detached_log_normalizer
        )
        p_sq = torch.exp(log_p_sq_sum).clamp(max=1.0)
        before_sq = (1.0 + p_sq - 2.0 * p_target).clamp_min(0.0)
        fisher_sq = (1.0 - p_sq).clamp_min(0.0)
        before = torch.sqrt(before_sq)
        effective_radius = torch.sqrt(fisher_sq) / float(step_scale)
        trust = torch.where(
            before <= effective_radius,
            torch.ones_like(before),
            effective_radius / before.clamp_min(float(eps)),
        ).clamp(min=0.0, max=1.0)
        if response_mask is not None:
            if response_mask.shape != before.shape:
                raise ValueError("response_mask must match [batch, time]")
            trust = torch.where(response_mask, trust, torch.ones_like(trust))
            before = before * response_mask.to(dtype=before.dtype)
            effective_radius = effective_radius * response_mask.to(dtype=effective_radius.dtype)
        after = trust * before

    return nll * trust.detach(), trust, before, after, effective_radius


def fisher_step_projected_sft_loss(
    logits: torch.Tensor,
    target_ids: list[torch.Tensor],
    *,
    step_scale: float = 1.0,
    response_mask: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example reduction of :func:`fisher_step_projected_sft_tokens`."""
    token_loss, trust, before, after, radius = fisher_step_projected_sft_tokens(
        logits,
        target_ids,
        step_scale=step_scale,
        response_mask=response_mask,
        eps=eps,
    )
    return (
        _masked_token_reduction(token_loss, response_mask),
        trust,
        before,
        after,
        radius,
    )

def full_forward_kl_tokens(student_logits: torch.Tensor, target_log_probs: torch.Tensor) -> torch.Tensor:
    """Tokenwise KL(detached target || student), without reducing over time."""
    log_p = torch.log_softmax(student_logits.float(), dim=-1)
    log_q = target_log_probs.float()
    q = log_q.exp()
    return (q * (log_q - log_p)).sum(dim=-1)


def full_forward_kl(
    student_logits: torch.Tensor,
    target_log_probs: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-vocabulary forward KL with the repository's per-example reduction."""
    return _masked_token_reduction(full_forward_kl_tokens(student_logits, target_log_probs), response_mask)


def veto_target_log_probs(
    teacher_log_probs: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Construct Veto's detached product-of-experts target.

    The paper defines ``Q ∝ P_T P_S^beta``.  The student term is detached so Q
    is a target rather than a second gradient path.  ``beta=0`` recovers the
    ordinary teacher target.
    """
    if float(beta) < 0.0:
        raise ValueError("Veto beta must be nonnegative")
    student_logp = torch.log_softmax(student_logits.float(), dim=-1).detach()
    return torch.log_softmax(teacher_log_probs.float() + float(beta) * student_logp, dim=-1).detach()


def on_policy_sft_group_losses(
    logits: torch.Tensor,
    target_ids: list[torch.Tensor],
    response_mask: torch.Tensor,
    accepted: torch.Tensor,
    prompt_indices: torch.Tensor,
    num_prompts: int,
    group_size: int,
) -> torch.Tensor:
    """On-Policy SFT objective.

    Correct responses within the configured length limit are retained.  Every
    retained sequence contributes its summed token NLL divided by the maximum
    retained response length in the complete generated minibatch, and the
    objective is divided by ``B*G`` rather than by the number retained.  The
    result contains one prompt-level scalar so it plugs into the repository's
    prompt-level gradient accumulation without changing the effective scale.
    """
    if logits.ndim != 3 or response_mask.ndim != 2:
        raise ValueError("logits and response_mask must be [batch,time,vocab] and [batch,time]")
    batch = logits.shape[0]
    if len(target_ids) != batch or accepted.numel() != batch or prompt_indices.numel() != batch:
        raise ValueError("On-Policy SFT metadata must match the rollout batch")
    if int(num_prompts) < 1 or int(group_size) < 1:
        raise ValueError("num_prompts and group_size must be positive")
    accepted = accepted.to(device=logits.device, dtype=torch.bool).reshape(-1)
    prompt_indices = prompt_indices.to(device=logits.device, dtype=torch.long).reshape(-1)
    token_nll = supervised_nll_tokens(logits, target_ids)
    mask = response_mask.to(dtype=token_nll.dtype)
    lengths = response_mask.sum(dim=-1)
    if bool(accepted.any()):
        maximum = lengths[accepted].max().to(dtype=token_nll.dtype).clamp_min(1.0)
    else:
        maximum = torch.ones((), dtype=token_nll.dtype, device=logits.device)
    sequence_loss = (token_nll * mask).sum(dim=-1) / maximum
    sequence_loss = sequence_loss * accepted.to(dtype=sequence_loss.dtype)
    prompt_loss = torch.zeros(int(num_prompts), dtype=sequence_loss.dtype, device=logits.device)
    prompt_loss.scatter_add_(0, prompt_indices, sequence_loss)
    return prompt_loss / float(group_size)


def topd_proximal_rewards(
    teacher_token_logp: torch.Tensor,
    old_student_token_logp: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """TOP-D's bounded proximal-teacher immediate reward.

    ``r = log(alpha * exp(log q - log p_old) + 1 - alpha)`` is evaluated with
    ``logaddexp`` for numerical stability.
    """
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("TOP-D alpha must lie in (0,1)")
    ratio_log = teacher_token_logp.float() - old_student_token_logp.float()
    log_alpha = torch.tensor(alpha, dtype=ratio_log.dtype, device=ratio_log.device).log()
    log_one_minus = torch.tensor(1.0 - alpha, dtype=ratio_log.dtype, device=ratio_log.device).log()
    return torch.logaddexp(log_alpha + ratio_log, log_one_minus)


def topd_length_normalized_returns(rewards: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """TOP-D token return: immediate reward plus mean of future rewards."""
    if rewards.shape != response_mask.shape or rewards.ndim != 2:
        raise ValueError("TOP-D rewards and mask must have identical [batch,time] shapes")
    mask = response_mask.to(dtype=rewards.dtype)
    masked = rewards * mask
    reverse_sum = torch.flip(torch.cumsum(torch.flip(masked, dims=[1]), dim=1), dims=[1])
    reverse_count = torch.flip(torch.cumsum(torch.flip(mask, dims=[1]), dim=1), dims=[1])
    future_sum = reverse_sum - masked
    future_count = (reverse_count - mask).clamp_min(1.0)
    future_mean = torch.where(reverse_count > 1.0, future_sum / future_count, torch.zeros_like(rewards))
    return (rewards + future_mean) * mask


def k1_policy_gradient_tokens(
    current_token_logp: torch.Tensor,
    old_student_token_logp: torch.Tensor,
    teacher_token_logp: torch.Tensor,
) -> torch.Tensor:
    """Negative K1 reverse-KL policy-gradient surrogate for sampled student tokens."""
    reward = (teacher_token_logp.float() - old_student_token_logp.float()).detach()
    return -reward * current_token_logp


def jensen_shannon_divergence_tokens(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    """Tokenwise Jensen-Shannon divergence between two full-vocabulary log distributions.

    Used by DemoPSD (Eq. 7) as the teacher-student disagreement
    ``d_t = JSD(pi_S || pi_T)``.  Both inputs are detached log-probability tensors
    shaped ``[..., vocab]``; the result drops the vocabulary axis.  The mixture is
    formed in log space with ``logaddexp`` for numerical stability, and the value
    is clamped at zero to remove tiny negative rounding residues.
    """
    if log_p.shape != log_q.shape:
        raise ValueError("JSD inputs must have identical shapes")
    log_p = log_p.float()
    log_q = log_q.float()
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    p = log_p.exp()
    q = log_q.exp()
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).clamp_min(0.0)


def demopsd_leakage_attenuation(
    disagreement: torch.Tensor,
    beta: float,
    alpha_max: float,
) -> torch.Tensor:
    """DemoPSD leakage-attenuation coefficient (Eq. 8).

    ``alpha_t = (sigmoid(beta * d_t) - 0.5) * 2 * alpha_max`` is the paper's
    remapped schedule: monotonically increasing in the disagreement ``d_t``,
    exactly zero at ``d_t = 0``, and saturating below ``alpha_max`` so the
    privileged teacher always retains at least ``1 - alpha_max`` of the mixture
    weight.  ``beta`` controls how sharply the gate responds to disagreement.
    """
    beta = float(beta)
    alpha_max = float(alpha_max)
    if beta <= 0.0:
        raise ValueError("DemoPSD beta must be positive")
    if not 0.0 <= alpha_max <= 1.0:
        raise ValueError("DemoPSD alpha_max must lie in [0, 1]")
    return (torch.sigmoid(beta * disagreement.float()) - 0.5) * (2.0 * alpha_max)


def demopsd_barycenter_target_log_probs(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """DemoPSD reverse-KL barycenter target (Eqs. 9-11).

    The target is the normalized geometric mixture
    ``pi_target ∝ pi_T^{1 - alpha_t} * pi_S^{alpha_t}``, equivalently the
    per-token interpolation of the two log distributions followed by a
    log-softmax renormalization.  ``alpha`` is per token (``[..., time]``) and
    everything is detached: the barycenter is a fixed distillation target and
    never a second gradient path.
    """
    if teacher_log_probs.shape != student_log_probs.shape:
        raise ValueError("DemoPSD teacher and student tensors must have identical shapes")
    if alpha.shape != teacher_log_probs.shape[:-1]:
        raise ValueError("DemoPSD alpha must match the token axes of the distributions")
    alpha = alpha.detach().float().clamp(min=0.0, max=1.0)
    mixture = (
        (1.0 - alpha)[..., None] * teacher_log_probs.detach().float()
        + alpha[..., None] * student_log_probs.detach().float()
    )
    return torch.log_softmax(mixture, dim=-1).detach()


def scope_group_weights(
    sequence_log_prob: torch.Tensor,
    lengths: torch.Tensor,
    temperature: float,
    mode: str,
) -> torch.Tensor:
    """SCOPE dual-path group-relative perplexity weights.

    Given sequence-level log probabilities ``log pi(y_i | x)`` and response
    lengths ``|y_i|`` for one prompt's trajectory group, the weights are the
    softmax of the length-normalized scores scaled by the temperature ``tau``:

    - ``mode="student"`` (correct set, Eq. 4):
      ``w_i ∝ exp(-log pi_S(y_i|x) / (tau |y_i|)) = PPL_S(y_i|x)^{1/tau}``,
      amplifying correct but low-confidence "unconventional valid paths".
    - ``mode="teacher"`` (incorrect set, Eq. 5):
      ``w_i ∝ exp(+log pi_T(y_i|x) / (tau |y_i|)) = PPL_T(y_i|x)^{-1/tau}``,
      down-weighting flawed prefixes on which the teacher is high-perplexity.

    Weights are nonnegative and sum to one within the group (Eqs. 18-19).
    """
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("SCOPE weight_temperature must be positive")
    if mode not in {"student", "teacher"}:
        raise ValueError("SCOPE weight mode must be 'student' or 'teacher'")
    if sequence_log_prob.shape != lengths.shape or sequence_log_prob.ndim != 1:
        raise ValueError("SCOPE group inputs must be matching one-dimensional tensors")
    scores = sequence_log_prob.float() / (temperature * lengths.float().clamp_min(1.0))
    if mode == "student":
        scores = -scores
    return torch.softmax(scores, dim=0)
