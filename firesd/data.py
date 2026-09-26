"""Hugging Face adapters for lightweight datasets with explicit reasoning traces."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from .utils import get_logger


def stable_example_seed(*parts: Any) -> int:
    """Deterministic 64-bit seed from stable string parts."""
    joined = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big")


def _choice_alternation(labels: Sequence[str] | None = None) -> str:
    label_list = [str(label).strip().upper() for label in (labels or ()) if str(label).strip()]
    if not label_list:
        label_list = ["A", "B", "C", "D", "E"]
    return "|".join(re.escape(label) for label in sorted(label_list, key=len, reverse=True))


def _marked_choice_label(text: str, labels: Sequence[str] | None = None) -> str:
    """Label taken from an explicit ``Answer: X`` marker, or ``""`` if absent."""
    alternation = _choice_alternation(labels)
    marked = re.findall(rf"ANSWER\s*:\s*\**\s*({alternation})(?![A-Z0-9])", text.upper())
    return marked[-1] if marked else ""


def _extract_choice_label(text: str, labels: Sequence[str] | None = None) -> str:
    alternation = _choice_alternation(labels)
    upper = text.upper()
    marked = _marked_choice_label(text, labels)
    if marked:
        return marked
    tokens = re.findall(rf"(?<![A-Z0-9])({alternation})(?![A-Z0-9])", upper)
    return tokens[-1] if tokens else ""


def _select_limit(ds, limit: int | None, seed: int):
    if limit is None:
        return ds
    return ds.shuffle(seed=seed).select(range(min(int(limit), len(ds))))


@dataclass(frozen=True)
class TaskExample:
    uid: str
    prompt: str
    answer: str
    task_type: str
    choices: tuple[str, ...] = ()
    # Optional benchmark-defined question type/subject. Kept after `choices` so
    # every legacy positional TaskExample construction remains unchanged.
    category: str = ""


class DatasetAdapter:
    name: str

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        raise NotImplementedError

    def extract_prediction(self, text: str, example: TaskExample) -> str:
        raise NotImplementedError

    def is_correct(self, text: str, example: TaskExample) -> bool:
        return self.extract_prediction(text, example) == self.normalize_gold(example.answer)

    def has_final_answer(self, text: str, example: TaskExample) -> bool:
        """Whether the rollout actually reached its canonical final-answer line.

        ``is_correct`` always returns a verdict, because every extractor falls
        back to the trailing text when the requested marker is absent.  That
        fallback is meaningless for a rollout that ran out of generation budget
        mid-derivation: for multiple choice it returns whichever option letter
        happened to appear last in the reasoning, so roughly one truncated
        rollout in ``len(choices)`` is scored correct by accident.  This hook
        reports whether there is a real final answer to grade, and is consulted
        only by ``generation.drop_unfinished_rollouts``.  ``is_correct`` and
        every published metric are deliberately left untouched.
        """
        return True

    def normalize_gold(self, answer: str) -> str:
        return re.sub(r"\s+", " ", answer.strip().lower())

    def render_answer(self, answer: str, example: TaskExample) -> str:
        return answer


class NumericReasoningAdapter(DatasetAdapter):
    """Shared exact-match handling for free-form numerical reasoning tasks."""

    @staticmethod
    def _last_numeric_token(text: str) -> str | None:
        values = re.findall(
            r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[-+]?\d+(?:\.\d+)?)?",
            text.replace(",", ""),
        )
        return values[-1] if values else None

    @staticmethod
    def _canonical_fraction(token: str) -> str:
        try:
            value = Fraction(token)
        except (ValueError, ZeroDivisionError):
            return token.strip().lower()
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"

    def extract_prediction(self, text: str, example: TaskExample) -> str:
        hashes = re.findall(r"####\s*([^\n]+)", text)
        candidate = hashes[-1] if hashes else text.splitlines()[-1] if text.strip() else ""
        token = self._last_numeric_token(candidate)
        return self._canonical_fraction(token) if token is not None else candidate.strip().lower()

    def normalize_gold(self, answer: str) -> str:
        token = self._last_numeric_token(answer)
        return self._canonical_fraction(token) if token is not None else answer.strip().lower()

    def has_final_answer(self, text: str, example: TaskExample) -> bool:
        marked = re.findall(r"####\s*([^\n]+)", text)
        if not marked:
            return False
        return self._last_numeric_token(marked[-1]) is not None

    def render_answer(self, answer: str, example: TaskExample) -> str:
        return f"#### {answer}"


class GSM8KAdapter(NumericReasoningAdapter):
    name = "gsm8k"

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        from datasets import load_dataset

        ds = _select_limit(load_dataset("openai/gsm8k", "main", split=split), limit, seed)
        output: list[TaskExample] = []
        for index, row in enumerate(ds):
            raw = str(row["answer"]).strip()
            match = re.search(r"####\s*([^\n]+)", raw)
            gold = match.group(1).strip() if match else raw
            output.append(TaskExample(str(index), str(row["question"]), gold, "math"))
        return output


class ASDivAdapter(NumericReasoningAdapter):
    """Numeric ASDiv-A subset with a deterministic 80/20 train/test partition."""

    name = "asdiv"

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        from datasets import load_dataset

        # The public HF conversion exposes one validation split. Use a fixed
        # partition independent of experiment seed to prevent train/test overlap.
        full = load_dataset("EleutherAI/asdiv", "asdiv", split="validation").shuffle(seed=17_029)
        split_name = str(split).lower()
        cut = int(0.8 * len(full))
        if split_name in {"train", "training"}:
            ds = full.select(range(cut))
        elif split_name in {"validation", "valid", "dev", "test"}:
            ds = full.select(range(cut, len(full)))
        else:
            raise ValueError("ASDiv split must be train, validation/dev, or test")
        ds = _select_limit(ds, limit, seed)

        output: list[TaskExample] = []
        for index, row in enumerate(ds):
            answer_raw = str(row.get("answer", "")).strip()
            token = self._last_numeric_token(answer_raw)
            # ASDiv includes a small number of comparison/text answers. The
            # arithmetic subset keeps evaluation consistent with the other
            # free-form numeric datasets.
            if token is None:
                continue
            body = str(row.get("body", "")).strip()
            question = str(row.get("question", "")).strip()
            prompt = f"{body}\n\n{question}" if body else question
            gold = self._canonical_fraction(token)
            output.append(TaskExample(f"asdiv-{index}", prompt, gold, "math"))
        return output


class MultipleChoiceAdapter(DatasetAdapter):
    def extract_prediction(self, text: str, example: TaskExample) -> str:
        return _extract_choice_label(text, example.choices)

    def has_final_answer(self, text: str, example: TaskExample) -> bool:
        # Only the explicit ``Answer: X`` marker counts.  The bare-letter
        # fallback inside ``_extract_choice_label`` returns whichever option
        # letter appeared last anywhere in the reasoning, which is not a
        # submitted answer at all for a rollout that never finished.
        return _marked_choice_label(text, example.choices) != ""

    def normalize_gold(self, answer: str) -> str:
        return re.sub(r"\s+", " ", answer.strip()).upper()

    def render_answer(self, answer: str, example: TaskExample) -> str:
        return f"Answer: {self.normalize_gold(answer)}"


class AQuARATAdapter(MultipleChoiceAdapter):
    name = "aqua_rat"

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        from datasets import load_dataset

        try:
            ds = load_dataset("deepmind/aqua_rat", "raw", split=split)
        except (ValueError, TypeError):
            # Compatibility with dataset revisions that expose raw as default.
            ds = load_dataset("deepmind/aqua_rat", split=split)
        ds = _select_limit(ds, limit, seed)

        output: list[TaskExample] = []
        for index, row in enumerate(ds):
            options = [str(value).strip() for value in row["options"]]
            labels: list[str] = []
            rendered_options: list[str] = []
            for option_index, option in enumerate(options):
                match = re.match(r"\s*([A-Ea-e])\s*[\)\.:]\s*(.*)", option)
                if match:
                    label, text = match.group(1).upper(), match.group(2).strip()
                else:
                    label, text = chr(ord("A") + option_index), option
                labels.append(label)
                rendered_options.append(f"{label}. {text}")
            prompt = f"{row['question']}\n\nChoices:\n" + "\n".join(rendered_options)
            correct = str(row["correct"]).strip().upper()
            output.append(
                TaskExample(
                    f"aqua-{index}",
                    prompt,
                    correct,
                    "multiple_choice",
                    tuple(labels),
                )
            )
        return output


class ARCChallengeAdapter(MultipleChoiceAdapter):
    """AI2 ARC-Challenge multiple-choice science questions."""

    name = "arc_challenge"

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
        ds = _select_limit(ds, limit, seed)
        output: list[TaskExample] = []
        for index, row in enumerate(ds):
            choices = row.get("choices") or {}
            labels_raw = [str(v).strip().upper() for v in (choices.get("label") or [])]
            texts = [str(v).strip() for v in (choices.get("text") or [])]
            if len(labels_raw) < 2 or len(labels_raw) != len(texts):
                continue
            # ARC occasionally uses numeric labels. Normalize every row to the
            # same A/B/C/... answer space so parsing is deterministic.
            labels = tuple(chr(ord("A") + i) for i in range(len(texts)))
            answer_key = str(row.get("answerKey", "")).strip().upper()
            if answer_key in labels_raw:
                answer = labels[labels_raw.index(answer_key)]
            elif answer_key.isdigit() and 1 <= int(answer_key) <= len(labels):
                answer = labels[int(answer_key) - 1]
            else:
                continue
            rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
            prompt = f"{str(row['question']).strip()}\n\nChoices:\n{rendered}"
            uid = str(row.get("id", f"arc-challenge-{split}-{index}"))
            output.append(TaskExample(uid, prompt, answer, "multiple_choice", labels, "science"))
        return output


class WinoGrandeAdapter(MultipleChoiceAdapter):
    """WinoGrande debiased binary commonsense/coreference benchmark."""

    name = "winogrande"

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        from datasets import load_dataset

        split_name = str(split).lower()
        if split_name in {"validation", "valid", "val", "dev", "test"}:
            hf_split = "validation"
        elif split_name in {"train", "training"}:
            hf_split = "train"
        else:
            raise ValueError("WinoGrande split must be train or validation/dev/test")

        ds = load_dataset("allenai/winogrande", "winogrande_debiased", split=hf_split)
        ds = _select_limit(ds, limit, seed)
        labels = ("A", "B")
        output: list[TaskExample] = []
        for index, row in enumerate(ds):
            sentence = str(row.get("sentence", "")).strip()
            options = [str(row.get("option1", "")).strip(), str(row.get("option2", "")).strip()]
            answer_raw = str(row.get("answer", "")).strip()
            if not sentence or any(not option for option in options) or answer_raw not in {"1", "2"}:
                continue
            prompt = (
                "Choose the option that best fills the blank in the sentence. "
                "Resolve the reference using commonsense reasoning.\n\n"
                f"Sentence:\n{sentence}\n\n"
                f"Choices:\nA. {options[0]}\nB. {options[1]}"
            )
            uid = str(row.get("idx", f"winogrande-{hf_split}-{index}"))
            output.append(
                TaskExample(
                    uid,
                    prompt,
                    labels[int(answer_raw) - 1],
                    "multiple_choice",
                    labels,
                    "coreference_commonsense",
                )
            )
        return output


class WiCAdapter(MultipleChoiceAdapter):
    """SuperGLUE WiC binary word-sense disambiguation benchmark."""

    name = "wic"

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        from datasets import load_dataset

        split_name = str(split).lower()
        if split_name in {"validation", "valid", "val", "dev", "test"}:
            hf_split = "validation"
        elif split_name in {"train", "training"}:
            hf_split = "train"
        else:
            raise ValueError("WiC split must be train or validation/dev/test")

        ds = load_dataset("aps/super_glue", "wic", split=hf_split)
        ds = _select_limit(ds, limit, seed)
        labels = ("A", "B")
        output: list[TaskExample] = []
        for index, row in enumerate(ds):
            word = str(row.get("word", "")).strip()
            sentence1 = str(row.get("sentence1", "")).strip()
            sentence2 = str(row.get("sentence2", "")).strip()
            if not word or not sentence1 or not sentence2:
                continue
            raw_label = row.get("label", -1)
            if isinstance(raw_label, bool):
                same_meaning = raw_label
            else:
                try:
                    label_int = int(raw_label)
                except (TypeError, ValueError):
                    raw = str(raw_label).strip().lower()
                    if raw in {"true", "1"}:
                        label_int = 1
                    elif raw in {"false", "0"}:
                        label_int = 0
                    else:
                        continue
                if label_int not in {0, 1}:
                    continue
                same_meaning = bool(label_int)
            prompt = (
                f"Consider the word '{word}' in the two sentences below. Determine whether it has "
                "the same meaning in both sentences.\n\n"
                f"Sentence 1:\n{sentence1}\n\n"
                f"Sentence 2:\n{sentence2}\n\n"
                "Choices:\nA. Different meaning\nB. Same meaning"
            )
            uid = str(row.get("idx", f"wic-{hf_split}-{index}"))
            output.append(
                TaskExample(
                    uid,
                    prompt,
                    "B" if same_meaning else "A",
                    "multiple_choice",
                    labels,
                    "word_sense",
                )
            )
        return output


class _MMLUProPartitionAdapter(MultipleChoiceAdapter):
    """Private full-pool partition helper for the six-category benchmark.

    MMLU-Pro ships only a large ``test`` split and a 70-row ``validation`` split,
    so a deterministic in-domain partition independent of the experiment seed is
    used to keep train and evaluation examples disjoint across seeds, exactly as
    for ASDiv.  Rows carry up to ten options, padded with ``N/A`` placeholders
    that are dropped before rendering.  The ``category`` field (math, physics,
    law, ...) becomes the evaluation slice.
    """

    _PARTITION_SEED = 31_337
    _LABELS = "ABCDEFGHIJ"

    @classmethod
    def _clean_options(cls, row) -> list[str]:
        options: list[str] = []
        for value in row.get("options") or []:
            text = str(value).strip()
            if not text or text.upper() == "N/A":
                # Trailing placeholders pad short questions out to ten slots.
                break
            options.append(text)
        return options

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        from datasets import load_dataset

        full = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

        def usable(row) -> bool:
            options = self._clean_options(row)
            index = row.get("answer_index")
            return (
                2 <= len(options) <= len(self._LABELS)
                and isinstance(index, int)
                and 0 <= index < len(options)
                and bool(str(row.get("question", "")).strip())
            )

        full = full.filter(usable).shuffle(seed=self._PARTITION_SEED)
        cut = int(0.8 * len(full))
        split_name = str(split).lower()
        if split_name in {"train", "training"}:
            ds = full.select(range(cut))
            partition = "train"
        elif split_name in {"validation", "valid", "val", "dev", "test"}:
            ds = full.select(range(cut, len(full)))
            partition = "validation"
        else:
            raise ValueError("MMLU-Pro split must be train or validation/dev/test")
        ds = _select_limit(ds, limit, seed)

        output: list[TaskExample] = []
        for index, row in enumerate(ds):
            options = self._clean_options(row)
            labels = tuple(self._LABELS[position] for position in range(len(options)))
            rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, options))
            prompt = f"{str(row['question']).strip()}\n\nChoices:\n{rendered}"
            answer = labels[int(row["answer_index"])]
            category = str(row.get("category", "")).strip()
            output.append(
                TaskExample(
                    f"mmlu-pro-{partition}-{index}",
                    prompt,
                    answer,
                    "multiple_choice",
                    labels,
                    category,
                )
            )
        return output


class MMLUProSixAdapter(_MMLUProPartitionAdapter):
    """Balanced six-category MMLU-Pro subset for the 14B experiments.

    The subset is one benchmark, not six separate datasets.  It keeps six broad
    disciplines with enough examples to support balanced held-out slices:
    Biology, Chemistry, Computer Science, Economics, Engineering, and Physics.
    ``limit`` is interpreted as a total budget and allocated as evenly as
    possible across categories; the returned examples are round-robin
    interleaved so every evaluation prefix remains category-balanced.

    MMLU-Pro has had reports of answer-option whitespace artifacts.  The parent
    adapter's ``_clean_options`` strips every option before rendering, so that
    formatting signal is intentionally unavailable to the model here.
    """

    name = "mmlu_pro_six"
    CATEGORIES = (
        "biology",
        "chemistry",
        "computer science",
        "economics",
        "engineering",
        "physics",
    )

    def load(self, split: str, limit: int | None, seed: int) -> list[TaskExample]:
        # Load the fixed parent partition without truncating it first; otherwise
        # a global random limit could make the six subject slices imbalanced.
        examples = super().load(split, None, seed)
        by_category: dict[str, list[TaskExample]] = {category: [] for category in self.CATEGORIES}
        for example in examples:
            category = str(example.category).strip().lower()
            if category in by_category:
                by_category[category].append(example)

        if limit is None:
            selected = by_category
        else:
            total = int(limit)
            if total < len(self.CATEGORIES):
                raise ValueError(
                    f"mmlu_pro_six limit must be at least {len(self.CATEGORIES)} to cover every category"
                )
            base, remainder = divmod(total, len(self.CATEGORIES))
            selected: dict[str, list[TaskExample]] = {}
            for category_index, category in enumerate(self.CATEGORIES):
                requested = base + int(category_index < remainder)
                pool = list(by_category[category])
                rng = __import__("random").Random(
                    stable_example_seed(self.name, split, seed, category)
                )
                rng.shuffle(pool)
                if len(pool) < requested:
                    raise ValueError(
                        f"MMLU-Pro category {category!r} has only {len(pool)} examples in {split}, "
                        f"but the balanced subset requested {requested}."
                    )
                selected[category] = pool[:requested]

        # Round-robin interleaving makes prefixes balanced too, which matters
        # because evaluate() intentionally consumes examples[:task_examples].
        output: list[TaskExample] = []
        max_len = max((len(items) for items in selected.values()), default=0)
        for index in range(max_len):
            for category in self.CATEGORIES:
                items = selected[category]
                if index < len(items):
                    output.append(items[index])
        return output


_REGISTRY: dict[str, DatasetAdapter] = {
    "gsm8k": GSM8KAdapter(),
    "asdiv": ASDivAdapter(),
    "aqua_rat": AQuARATAdapter(),
    "arc_challenge": ARCChallengeAdapter(),
    "winogrande": WinoGrandeAdapter(),
    "wic": WiCAdapter(),
    "mmlu_pro_six": MMLUProSixAdapter(),
}


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter {name!r}; available={sorted(_REGISTRY)}") from exc


def wilson_half_width(accuracy: float, total: int, z: float = 1.96) -> float:
    """95% Wilson half-width, used to report whether a split can resolve anything."""
    if total <= 0:
        return 1.0
    z2 = z * z
    denominator = 1.0 + z2 / total
    spread = z * math.sqrt((accuracy * (1.0 - accuracy) + z2 / (4.0 * total)) / total)
    return spread / denominator


def load_examples(data_cfg: dict[str, Any], split: str, seed: int) -> list[TaskExample]:
    adapter = get_adapter(str(data_cfg["adapter"]))
    split_key = "train_split" if split == "train" else "eval_split"
    limit_key = "train_limit" if split == "train" else "eval_limit"
    examples = adapter.load(str(data_cfg[split_key]), data_cfg.get(limit_key), seed)

    if split != "train":
        # An underpowered evaluation split silently produces a comparison table
        # that cannot separate its own rows.  Report the resolution explicitly,
        # and refuse to start when the config declares a floor that is not met.
        logger = get_logger("stablesd.data")
        half_width = wilson_half_width(0.5, len(examples))
        logger.info(
            "Evaluation split %s/%s holds %d examples; worst-case 95%% Wilson "
            "half-width is +/-%.3f accuracy.",
            data_cfg["adapter"],
            data_cfg[split_key],
            len(examples),
            half_width,
        )
        minimum = int(data_cfg.get("min_eval_examples", 0) or 0)
        if minimum and len(examples) < minimum:
            raise ValueError(
                f"Adapter {data_cfg['adapter']!r} yielded only {len(examples)} evaluation "
                f"examples but data.min_eval_examples={minimum}. At this size the 95% "
                f"confidence half-width is +/-{half_width:.3f}, which cannot resolve the "
                "differences this comparison matrix is meant to measure. Widen the pool "
                "or lower data.min_eval_examples "
                "deliberately."
            )
        if len(examples) < 192:
            logger.warning(
                "Only %d evaluation examples: differences smaller than about %.2f accuracy "
                "will not be distinguishable from noise.",
                len(examples),
                2.0 * half_width,
            )
    return examples
