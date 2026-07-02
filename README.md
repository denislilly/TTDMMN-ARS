TTDMMN-ARS
Tensor Train Decomposed Multi-Modal Neural Network with Adaptive Rank Selection for Efficient Fake News Detection
Official implementation of the paper "Tensor Train Decomposed Multi-Modal
Neural Network with Adaptive Rank Selection for Efficient Fake News
Detection" (Denis R., Scientific Reports).
TTDMMN-ARS fuses text (BERT), visual (ResNet-50), and social-propagation
graph (GCN/GraphSAGE/GAT) signals for fake-news classification, while
factorising the dominant weight matrices in every branch into Tensor-Train
(TT) format and learning the per-layer TT rank automatically via
Adaptive Rank Selection (ARS) — a nuclear-norm regulariser with a
cosine warm-up schedule.
> **Status: core architecture + training pipeline complete (milestone 1 of 2).**
> See [Repository status](#repository-status) below for exactly what is and
> isn't implemented yet.
Architecture
```
 text  ──▶ TCTE  (BERT-Base, TT-compressed attn/FFN)  ──▶ h_text   ∈ R^768
 image ──▶ VFE   (ResNet-50, Tucker-compressed convs)  ──▶ h_visual ∈ R^512
 graph ──▶ GSPM  (2-layer TT-weighted GCN/SAGE/GAT)     ──▶ h_social ∈ R^256
                                    │
                                    ▼
                    ACFL (cross-modal attention fusion) ──▶ h_fused ∈ R^256
                                    │
                                    ▼
                    TT-Linear classification head ──▶ ŷ ∈ [0,1]
```
Every dense weight matrix inside TCTE's attention/FFN projections, VFE's
`nn.Linear` head, and GSPM's graph-conv weights is replaced by a `TTLinear`
module (`models/tensor_train.py`) trained directly in TT-core form.
Convolutional kernels in the ResNet-50 backbone are Tucker-2 decomposed
(`models/vit_encoder.py::TuckerConv2d`) along the channel modes. `ARS`
(`models/adaptive_rank_selection.py`) adds a warmed-up nuclear-norm penalty
over every TT core's matricisation and truncates near-zero singular
directions after training.
Installation
```bash
git clone https://github.com/<org>/TTDMMN-ARS.git
cd TTDMMN-ARS
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate ttdmmn-ars
pip install -e .
```
Requires Python ≥3.11, PyTorch ≥2.3. GPU strongly recommended for BERT/ResNet
fine-tuning; CPU works for the unit tests and small-scale debugging.
Quick start
```python
from utils.config import load_config
from models.factory import build_model, build_trainer_config

cfg = load_config("configs/welfake.yaml")
model = build_model(cfg)               # full TTDMMN-ARS, TT-compressed
trainer_cfg = build_trainer_config(cfg)

print(model.compression_report())
# {'total_params': ..., 'tt_layer_params': ..., 'tt_dense_equivalent': ...,
#  'non_tt_params': ..., 'num_tt_layers': 88}
```
```python
import torch
from models.adaptive_rank_selection import AdaptiveRankSelector, ARSConfig
from losses.classification_loss import TTDMMNARSLoss

ars = AdaptiveRankSelector(model, ARSConfig(lambda_max=1e-3, warmup_steps=1000))
loss_fn = TTDMMNARSLoss(use_focal=False)

logits = model(input_ids, attention_mask=attention_mask)  # text-only dataset
loss, logs = loss_fn(logits, targets, rank_penalty=ars.penalty(step=0))
loss.backward()
```
Run the unit tests (no GPU / network required):
```bash
pytest tests/ -q
```
Configuration
`configs/default.yaml` holds every hyperparameter from the manuscript's
"Training Details" section (AdamW, cosine LR with warm-up, gradient
accumulation, ARS lambda schedule, etc.). Per-dataset configs
(`welfake.yaml`, `liar.yaml`, `politifact.yaml`, `gossipcop.yaml`,
`covid.yaml`) inherit it via a minimal `defaults: [default.yaml]`
convention (`utils/config.py`) and override only what differs (e.g.
GossipCop enables focal loss for its 76/24 class imbalance; PolitiFact/
GossipCop enable the graph branch since only FakeNewsNet ships propagation
graphs).
Repository status
Implemented and unit-tested (this milestone):
Component	File	Status
TT-SVD / TT-Linear core	`models/tensor_train.py`	✅ tested (reconstruction accuracy, param counts, autograd)
Adaptive Rank Selection	`models/adaptive_rank_selection.py`	✅ tested (warm-up schedule, penalty, truncation)
Text encoder (TCTE)	`models/bert_encoder.py`	✅ tested on synthetic BERT (network-free)
Visual encoder (VFE, Tucker convs)	`models/vit_encoder.py`	✅ tested (ResNet-50 + ViT-B alternative)
Graph encoder (GSPM: GCN/SAGE/GAT)	`models/graph_encoder.py`	✅ tested, all 3 backbones
Fusion (ACFL + early/late/tensor ablation variants)	`models/multimodal_fusion.py`	✅ tested
Classifier head	`models/classifier.py`	✅ tested
Full model wiring + modality-graceful degradation	`models/ttdmmn_ars.py`	✅ tested end-to-end (mixed missing-modality batches)
Losses (BCE + rank penalty, focal)	`losses/classification_loss.py`	✅ tested
Trainer (AdamW, cosine+warmup, grad accum, early stopping, checkpointing, ARS truncation)	`trainers/trainer.py`	✅ tested end-to-end on toy data
Evaluator + 5-fold CV runner	`trainers/evaluator.py`	✅ implemented
Metrics (Eq. 14-22)	`utils/metrics.py`	✅ tested, matches paper's reported CR/speedup figures
Config system (YAML + inheritance)	`utils/config.py`, `configs/*.yaml`	✅ tested
Reproducibility / logging	`utils/reproducibility.py`, `utils/logger.py`	✅ implemented
Packaging (setup.py, requirements, environment.yml, LICENSE, CITATION.cff)	—	✅
Not yet implemented (planned next milestone):
`datasets/` loaders for WELFake, LIAR, FakeNewsNet (PolitiFact/GossipCop),
COVID-19, and Kaggle download scripts.
`preprocessing/` (tokenizer wrapper, image transforms, graph builder from
raw FakeNewsNet propagation data).
`experiments/` scripts: `train.py`, `evaluate.py`, `cross_validation.py`,
`cross_domain.py`, `ablation.py`, `compression.py`, `rank_sensitivity.py`,
`explainability.py` (SHAP), `robustness.py`.
`supplementary/generate_all_figures.py`, `generate_all_tables.py` (all 15
figures / 10 tables listed in the spec, exported to CSV/Excel/LaTeX/MD).
`notebooks/visualization.ipynb`, `scripts/*.sh` orchestration.
`Dockerfile`, `docker-compose.yml`, GitHub Actions CI workflow.
`reproduce_paper.py` end-to-end pipeline script.
Example inference script and pretrained-checkpoint loading utility.
A note on reproducing exact paper numbers
This implementation faithfully reproduces the architecture and training
procedure described in the manuscript. Reproducing the exact reported
numbers (e.g. 97.58% Macro-F1 on WELFake) additionally requires: (1)
downloading the real Kaggle/FakeNewsNet datasets, (2) an NVIDIA A100-class
GPU for training and an RTX 4090-class GPU for the latency benchmarks (as
used in the paper, Section 3.10), and (3) the exact train/val/test splits
and preprocessing described in Section 4.2. None of that was available in
the environment this repository was built in — every module above was
instead validated with unit tests (synthetic data, small/CPU-friendly
configurations, no network calls) to confirm mathematical and
architectural correctness rather than to reproduce headline numbers.
Citation
See `CITATION.cff`.
License
MIT
