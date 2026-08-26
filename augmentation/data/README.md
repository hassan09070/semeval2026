# Augmented training data

Synthetic and expanded training instances used for the DimABSA experiments.
All files are JSONL in the DimABSA quadruplet format:

```json
{"ID": "...", "Text": "...", "Quadruplet": [{"Aspect": "...", "Category": "ENTITY#ATTRIBUTE", "Opinion": "...", "VA": "valence#arousal"}]}
```

## `gemini_synthetic/`

LLM-generated instances from **Gemini 2.5 Flash** with batch prompting. These
preserve original aspect terms and category labels while diversifying sentence
text, opinion expressions, and VA scores — deliberately targeting the
underrepresented regions of the valence–arousal space.

| File | Instances | Notes |
|---|--:|---|
| `eng_raw.jsonl` | 2,876 | English, unfiltered generator output |
| `eng_cleaned.jsonl` | 2,032 | English, after filtering (drops VA values > 9.0 and malformed rows) |
| `zho_raw.jsonl` | 2,270 | Chinese, unfiltered generator output |

`eng_cleaned.jsonl` is the version actually used in training — see
[`docs/experimental_setup.md`](../../docs/experimental_setup.md).

## `train_alltasks/`

Expanded English training splits covering all three subtasks, built from the
augmentation pipeline in [`../`](..).

| File | Instances | Domain |
|---|--:|---|
| `eng_laptop.jsonl` | 4,222 | Laptop |
| `eng_restaurant.jsonl` | 2,780 | Restaurant |

## Generation

Produced by the scripts one level up — `augmentation.py`, `aug2.py`, `aug3.py`,
and `kaggle_absa_augmentation.py`. All read credentials from the environment:

```bash
export GROQ_API_KEY="..."
```
