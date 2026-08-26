# SemEval-2026 Task 3 — DimABSA

**Habib University — Dhanani School of Science & Engineering, Karachi, Pakistan**

System, experiments, and published paper for **SemEval-2026 Task 3: Dimensional
Aspect-Based Sentiment Analysis (DimABSA)** — extracting structured sentiment elements
*and* regressing continuous **valence–arousal** scores across **6 languages** and **4 domains**.

Our modular four-stage pipeline placed **2nd for Tatar** and **6th for Russian** in
dimensional regression, achieving a best RMSE of **0.5333** (Subtask 1) and a best
cF1 of **0.5492** (Subtask 2).

---

## Paper

> **Habib University at SemEval-2026 Task 3: A Pipeline Approach for Dimensional Aspect-Based Sentiment Analysis**
> Muhammad Affan, Muhammad Hassan Shahzad, Mikaal Imam, Moiz Zulfiqar, Sandesh Kumar, Abdul Samad
> *Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)*, pages 3449–3459
> July 3–4, 2026 · Association for Computational Linguistics

📄 **[Read the paper](paper/Habib_University_SemEval2026_Task3_DimABSA.pdf)** ·
🔗 **[ACL Anthology](https://aclanthology.org/2026.semeval-1.428/)**

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{affan-etal-2026-habib,
    title     = "Habib University at {S}em{E}val-2026 Task 3: A Pipeline Approach
                 for Dimensional Aspect-Based Sentiment Analysis",
    author    = "Affan, Muhammad and Shahzad, Muhammad Hassan and Imam, Mikaal and
                 Zulfiqar, Moiz and Kumar, Sandesh and Samad, Abdul",
    booktitle = "Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)",
    month     = jul,
    year      = "2026",
    publisher = "Association for Computational Linguistics",
    pages     = "3449--3459",
    url       = "https://aclanthology.org/2026.semeval-1.428/"
}
```
</details>

---

## The Task

Traditional ABSA assigns **discrete** polarity labels — positive, negative, neutral.
DimABSA replaces them with **continuous valence–arousal (VA)** scores on a 1–9 scale
(Russell, 1980), so a system distinguishes not just polarity but *depth of emotion*.

Three subtasks form a progression of difficulty:

| Subtask | Name | Given | Predict |
|:--:|---|---|---|
| **ST1** | DimASR | sentence + aspect | continuous `(valence, arousal)` |
| **ST2** | DimASTE | sentence only | all `(aspect, opinion, VA)` **triplets** |
| **ST3** | DimASQP | sentence only | complete `(aspect, category, opinion, VA)` **quadruplets** |

**Scale:** 23,244 instances · 196 unique `Entity#Attribute` categories ·
6 languages (English, Chinese, Japanese, Russian, Tatar, Ukrainian) ·
4 domains (Restaurant, Laptop, Hotel, Finance).

**Why it's hard.** The data is doubly imbalanced. Valence skews mid-to-high (6–8) and
arousal clusters at 5–7, so extreme emotional states are barely represented. Categories
are worse: `FOOD#QUALITY` alone is **28.2%** of all instances, while **31%** of categories
each contribute under 2%.

---

## What We Did

A **four-stage pipeline** — Extractor → Pairer → Category Classifier → Regressor —
each stage trained independently, with each subtask using a different subset:

```
                    ┌─────────── ST1 uses only the Regressor ──────────┐
                    │                                                  │
sentence ──► Extractor ──► Pairer ──► Category Classifier ──► Regressor ──► (a, c, o, VA)
             (BIO tags)   (binary)    (entity → attribute)   (v, a)
             mDeBERTa     XLM-R       XLM-R ×2               mDeBERTa
                    │                                                  │
                    └────────── ST2 skips the Category stage ──────────┘
```

**Backbone selection.** mBERT, mDeBERTa, XLM-RoBERTa-base and XLM-RoBERTa-large were
benchmarked under one framework. **mDeBERTa-v3** won consistently — and notably,
switching to mDeBERTa beat *scaling up* to XLM-RoBERTa-large.

**Double [NULL] Token.** Standard BIO tagging cannot represent **implicit** aspects or
opinions — elements not literally present in the text. Rather than change the
architecture, we prepend two sentinel tokens:

```
S' = [NULL_asp, NULL_opi, w₁, …, wₙ]
```

When the gold aspect is implicit, `NULL_asp` receives the `B-ASP` label. Implicitness
handled with zero architectural modification.

**Regression stabilisation.** VA labels in `[1,9]` are linearly rescaled to `[-1,1]`:

$$y' = \frac{y-5}{4}$$

then perturbed with Gaussian noise $y_{noise} = y' + \epsilon,\ \epsilon \sim \mathcal{N}(0,\sigma^2)$
to reduce overfitting to exact annotation values. We also tried CCC loss and
Savitzky–Golay label smoothing — see the ablations below for why neither made the final system.

**Partitioned category prediction.** Rather than flat 196-class classification, ST3
decomposes into two sequential steps — an **Entity** model (`FOOD`, `SERVICE`) then an
**Attribute** model conditioned on the predicted entity (`QUALITY`, `PRICE`), assembled as
`C = ê#â`. This cuts cross-entity confusion (`FOOD#STYLE` vs `DRINKS#STYLE`) and lets each
attribute classifier train on a balanced, entity-specific label subset.

---

## Results

### Official leaderboard

| | Subtask 1 (RMSE ↓) | | Subtask 2 (cF1 ↑) | | Subtask 3 (cF1 ↑) | |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **Language / Domain** | **RMSE** | **Rank** | **cF1** | **Rank** | **cF1** | **Rank** |
| English Laptop | 1.3654 | 13 | 0.4770 | 15 | 0.0000 | 18 |
| English Restaurant | 1.3059 | 18 | 0.5202 | 17 | 0.0000 | 16 |
| Japanese Finance | 0.8907 | 8 | – | – | – | – |
| Japanese Hotel | 0.6680 | 9 | 0.3311 | 13 | 0.1853 | 9 |
| Russian Restaurant | 1.4344 | 6 | **0.5492** | 6 | 0.2963 | 11 |
| **Tatar Restaurant** | 1.6041 | **2** | 0.4839 | 5 | 0.2500 | 11 |
| Ukrainian Restaurant | 1.4661 | 7 | 0.5324 | 6 | 0.2938 | 12 |
| Chinese Restaurant | 0.9898 | 17 | 0.4622 | 11 | 0.3139 | 10 |
| **Chinese Finance** | **0.5333** | 8 | – | – | – | – |
| Chinese Laptop | 0.7311 | 13 | 0.4159 | 11 | 0.4199 | 10 |

The **low-resource results are the story**: Tatar (rank 2), Russian (rank 6) and Ukrainian
(rank 7) all place competitively despite minimal training data — the multilingual
architecture transfers.

### The English ST3 zero

English Laptop and Restaurant both scored **cF1 = 0.0000** in Subtask 3. This is *not* an
architectural collapse. Predictions were generated correctly by the pipeline; we attribute
the drop to a **silent serialization/formatting mismatch in the English submission file**
that caused the evaluation script to reject the quadruplets. The evidence: the same
English data scored a highly competitive **cF1 ≈ 0.50 in ST2**, which shares the entire
upstream pipeline. The failure is isolated to final file generation.

### Ablations

Two findings worth more than the leaderboard, both in the paper's appendices and
reproducible from [`evaluation/summary.txt`](evaluation/summary.txt):

**1. The Cross-Lingual Paradox.** Adding external Chinese SIGHAN-2024 data *degraded*
native Chinese performance (Chinese Restaurant RMSE 0.7619 → 0.7942) while acting as a
powerful cross-lingual regulariser everywhere else — English Laptop 0.9661 → 0.9150,
Russian Restaurant 1.2751 → 1.1958. Negative interference in-language, large gains
out-of-language.

| Lang | Domain | Base | +SIGHAN | +CCC | +2D Bin | **Final** |
|---|---|:--:|:--:|:--:|:--:|:--:|
| Eng | Laptop | 0.9661 | 0.9150 | 0.9821 | 2.2242 | **0.9542** |
| Eng | Rest. | 0.9352 | 0.9215 | 0.8721 | 1.6634 | **0.9111** |
| Jpn | Finance | 0.8353 | 0.8287 | 0.8607 | 0.9099 | **0.7520** |
| Rus | Rest. | 1.2751 | 1.1958 | 1.2268 | 1.2702 | **1.2554** |
| Zho | Rest. | 0.7619 | 0.7942 | 0.7338 | 0.7888 | **0.7536** |
| **Avg** | **All** | 0.9572 | 0.9462 | 0.9569 | 1.1718 | **0.9366** |

*(abridged — full 10-row table is Table 5 in the paper)*

**2. Contextual MLM actively harms sequence labeling.** We expected masked-language-model
augmentation to improve syntactic robustness. It did the opposite — F1 dropped
0.7846 → 0.7693 even with strict entity freezing. Perturbing the grammatical neighbourhood
around an aspect destroys the local syntactic cues BIO boundary detection depends on.
**Mention Replacement** (semantic substitution) proved far more robust, and the best
configuration (**F1 = 0.8531**) came from human-annotated SIGHAN data with *no* synthetic
MR augmentation at all — pure human annotation remains the strongest cross-lingual anchor.

---

## Repository Layout

```
.
├── paper/                  Published SemEval-2026 proceedings paper (PDF)
├── final_submission/       The submitted systems — 5 notebooks, ST1/ST2/ST3
├── subtask1_ablations/     11 ST1 training variants (one per ablation config)
├── augmentation/           Data augmentation pipelines
│   └── data/               Gemini-generated synthetic instances (eng + zho)
├── evaluation/             Official metric implementation + harness
│   ├── gold_data/          Gold labels, 10 language-domain pairs
│   ├── reports/            Official DimABSA evaluation result PDFs
│   └── summary.txt         Raw per-run scores behind the ablation tables
├── analysis/               Dataset analysis scripts
│   └── figures/            VA distributions, emotion quadrants, correlations
├── llm_baselines/          Generative-LLM comparison baselines
├── inference/              Prediction from published HF checkpoints
└── docs/                   Implementation notes & experimental setup
```

### The ablation notebooks

Each [`subtask1_ablations/`](subtask1_ablations) notebook is one row of the experiment
grid — they are kept separate deliberately, since each corresponds to a configuration
reported in the paper:

| Notebook | Configuration |
|---|---|
| `st1.ipynb` / `st1-copy.ipynb` | Baseline mDeBERTa regression |
| `st1-ccc.ipynb` | Concordance Correlation Coefficient loss |
| `st1-scale.ipynb` / `st1-normalize.ipynb` | VA target rescaling to `[-1,1]` |
| `st1-SG.ipynb` | Savitzky–Golay label smoothing |
| `st1-huber.ipynb` | Huber loss |
| `st1-reg.ipynb` | Regularisation sweep |
| `st1_decay.ipynb` | Weight-decay sweep |
| `st1-BT.ipynb` | Back-translation augmentation |
| `st1-new.ipynb` | Final combined configuration |

---

## Data

The **DimABSA 2026 dataset** is distributed by the task organisers:
**https://github.com/DimABSA/DimABSA2026/tree/main/task-dataset**

Each instance is an ID, a sentence, and a set of `(aspect, opinion, category, VA)`
quadruplets. Aspect and opinion may be *implicit*, represented as `NULL`.

| What's in this repo | Where |
|---|---|
| Gold labels (10 language-domain pairs) | [`evaluation/gold_data/`](evaluation/gold_data) |
| Gemini synthetic augmentation (eng, zho) | [`augmentation/data/`](augmentation/data) |
| Per-run evaluation scores | [`evaluation/summary.txt`](evaluation/summary.txt) |
| Official result reports | [`evaluation/reports/`](evaluation/reports) |

**External data used:** [SIGHAN-2024 dimABSA](https://github.com/NYCU-NLP/SIGHAN2024-dimABSA/blob/main/DataSets/dimABSA2024/Simplified) —
converted from parallel-list format into DimABSA quadruplets.

### Dataset distribution

| Language | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| English | Restaurant 2284 · Laptop 4076 | same | same |
| Japanese | Hotel 1600 · Finance 1024 | Hotel 1600 | Hotel 1600 |
| Russian | Restaurant 1240 | 1240 | 1240 |
| Tatar | Restaurant 1240 | 1240 | 1240 |
| Ukrainian | Restaurant 1240 | 1240 | 1240 |
| Chinese | Restaurant 6050 · Laptop 3490 · Finance 1000 | Rest. 6050 · Laptop 3490 | same |

---

## Reproducing

**Training setup** (identical across backbones for fair comparison): AdamW,
learning rate `2e-5`, batch size `16`, max sequence length `128`, internal dropout
disabled. ST1 trains 10 epochs with early stopping on RMSE; the ST2 extractor trains
15 epochs and the pairer 8, both early-stopped on validation F1. Split is 90/10
train/val, `seed = 42`.

**Evaluation:**

```bash
pip install -r evaluation/requirements.txt

# score one submission zip against gold
python evaluation/evaluate_single_zip.py evaluation/subtask_1_aug_rescale.zip

# score every submission zip in the folder
python evaluation/batch_evaluate_zips.py
```

Metrics: **PCC** for valence, **PCC** for arousal, and **RMSE** for ST1 (official ranking
is RMSE-based); **cF1 / cPrecision / cRecall** for ST2 and ST3.

**Inference from published checkpoints:**

```bash
export HF_TOKEN="..."
python inference/predict_from_hf.py
```

**Credentials.** Nothing is hardcoded — every script reads from the environment:

```bash
export HF_TOKEN="..."        # HuggingFace Hub (training + inference)
export GROQ_API_KEY="..."    # LLM-based augmentation
```

---

## Acknowledgments

We thank the organizers of SemEval-2026 Task 3 for curating the DimABSA dataset and
providing the comprehensive evaluation framework. We also gratefully acknowledge
computational support from **Google Colab** and **Kaggle**, which enabled our
fine-tuning processes.

---

## Key References

1. Yu et al. — *SemEval-2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA)*, SemEval-2026.
2. Lee et al. — [*DimABSA: Building multilingual and multidomain datasets for dimensional ABSA*](https://arxiv.org/abs/2601.23022), 2026.
3. Russell — *A circumplex model of affect*, J. Personality and Social Psychology, 1980.
4. Xu et al. — *HITSZ-HLT at SIGHAN-2024 dimABSA*, SIGHAN-10, 2024.
5. Atmaja & Akagi — *Evaluation of error and correlation-based loss functions for multitask learning dimensional speech emotion recognition*, 2021.
