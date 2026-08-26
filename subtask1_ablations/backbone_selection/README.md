# Backbone selection experiments

The earliest Subtask 1 experiments — the ones behind **Table 6, Phase 1** of the
paper, where the backbone was chosen before any augmentation work began:

| Notebook | Backbone | Val F1 (paper) |
|---|---|:--:|
| `bert.ipynb` | mBERT-base | 0.7709 |
| `bert_multilingual.ipynb` | mBERT multilingual variants | – |
| `deberta.ipynb` | mDeBERTa-v3-base | **0.8034** |

mDeBERTa won consistently and became the primary backbone for the final system.
These predate the `st1-*.ipynb` ablation series in the parent folder.
