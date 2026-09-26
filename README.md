# FIRE: Fisher-Informed Recalibration of Feedback-based On-Policy Self-Distillation of LLMs

This package contains the training and evaluation implementation, experiment
configurations, and dependencies for FIRE and its comparison methods.

## Setup

Use Python 3.10 or newer and a CUDA-enabled PyTorch installation. Run the
following commands from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The PyTorch installation must support the machine's CUDA environment. Model
weights and datasets are downloaded from Hugging Face on first use, so internet
access and sufficient cache space are required. The 14B experiments use NF4
quantization through `bitsandbytes` on Linux; the 1.5B and 3B experiments use
unquantized base weights. All configurations use LoRA adapters.

## Included experiments

| Configuration directory | Model checkpoint | Datasets |
| --- | --- | --- |
| `configs/experiments/qwen2p5_1p5b/` | `Qwen/Qwen2.5-1.5B-Instruct` | ASDiv, GSM8K, AQuA-RAT |
| `configs/experiments/qwen2p5_3b/` | `Qwen/Qwen2.5-3B-Instruct` | ARC-Challenge, WiC, WinoGrande |
| `configs/experiments/qwen2p5_14b/` | `Qwen/Qwen2.5-14B-Instruct` | MMLU-Pro, six categories |

The MMLU-Pro subset covers biology, chemistry, computer science, economics,
engineering, and physics. Its fixed partition and balanced category sampling
are implemented in `firesd/data.py`.

Each model/dataset directory includes `fire_sdpo.yaml` and the comparison
configurations `full_context_sdpo.yaml`, `on_policy_sft_sdpo.yaml`,
`veto_sdpo.yaml`, `topd_sdpo.yaml`, `tropd_sdpo.yaml`, `srpo_sdpo.yaml`,
`demopsd_sdpo.yaml`, and `scope_sdpo.yaml`.

The 1.5B and 3B directories also include the component ablations
`fire_no_attribution_sdpo.yaml`, `fire_no_projection_sdpo.yaml`, and
`hard_gradient_excision_sdpo.yaml`, plus FIRE learning-rate configurations
`fire_lr_0p5x_sdpo.yaml`, `fire_lr_1x_sdpo.yaml`, `fire_lr_2x_sdpo.yaml`, and
`fire_lr_4x_sdpo.yaml`.

## Start training

Run from this directory. These examples launch one FIRE experiment per model:

```bash
python -u scripts/train.py \
  --config configs/experiments/qwen2p5_1p5b/gsm8k/fire_sdpo.yaml \
  --set train.seed=0 --skip-existing

python -u scripts/train.py \
  --config configs/experiments/qwen2p5_3b/arc_challenge/fire_sdpo.yaml \
  --set train.seed=0 --skip-existing

python -u scripts/train.py \
  --config configs/experiments/qwen2p5_14b/mmlu_pro_six/fire_sdpo.yaml \
  --set train.seed=0 --skip-existing
```

Select another included YAML to run a different dataset, method, or ablation.
Use `CUDA_VISIBLE_DEVICES` to select the GPU when needed. Each invocation runs
one experiment and one seed on one device.

Configuration fragments under `configs/models/`, `configs/datasets/`,
`configs/tuning/`, `configs/contexts/`, and `configs/methods/` are composed by
the experiment YAML's `_base_` entries. Repeat `--set PATH=VALUE` for command-line
overrides, such as `--set train.seed=1` or `--set output.root=outputs_repeat`.
The resolved configuration records all effective settings.

To validate the configuration and dataset loading without loading model weights:

```bash
python scripts/train.py \
  --config configs/experiments/qwen2p5_1p5b/gsm8k/fire_sdpo.yaml \
  --dry-run --set output.root=outputs_preflight
```

Use a separate output directory for this check: a dry run creates artifacts,
and `--skip-existing` skips any nonempty run directory. This flag preserves
existing files; it does not resume training.

## Outputs

By default, each run writes to
`outputs/<experiment.name>/seed<train.seed>/`. The directory is created at
runtime. For example, the first command above writes to
`outputs/fire_sdpo_qwen2p5_1p5b_gsm8k/seed0/`.

| File | Contents |
| --- | --- |
| `resolved_config.json` | Complete configuration after composition and overrides |
| `source_config.yaml` | Copy of the selected experiment YAML |
| `latest_eval.json` | Most recent evaluation, updated after each evaluation |
| `metrics.json` | Training/evaluation history and final metrics, written at completion |
| `summary.json` | Resolved configuration and completed-run results; dry-run status for `--dry-run` |

Progress is printed to the terminal. Evaluation includes task accuracy and
routing diagnostics where applicable, with per-category accuracy for MMLU-Pro.
Checkpoints are disabled by default. With `--set output.save_checkpoints=true`,
each evaluation saves the student adapter, tokenizer, and EMA metadata under
`checkpoints/step_<step>/` within the run directory.
