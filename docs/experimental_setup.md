# Experimental Setup

## 1. Dataset

### 1.1 Data Sources
We combined multiple data sources to create our training dataset:

1. **DimABSA 2026 Dataset**: Our primary dataset includes six languages (English, Chinese, Japanese, Russian, Tatar, and Ukrainian) across four domains (restaurant, laptop, finance, and hotel). We utilized both task-specific and all-tasks training sets depending on domain availability.

2. **Augmented Gemini Data**: We incorporated augmented English data generated using Gemini, applying strict filtering to remove entries with valence or arousal values exceeding 9.0 to ensure data quality and consistency.

3. **SIGHAN 2024 Dataset**: We integrated two external Chinese restaurant domain datasets (TrainingSet1 and TrainingSet2) from SIGHAN 2024, transforming them into the quadruplet format to maintain structural consistency.

### 1.2 Data Preprocessing
- All datasets were normalized to a unified quadruplet format containing: Aspect, Category, Opinion, and VA (Valence#Arousal) fields
- Duplicate entries based on ID and Aspect were removed
- Malformed VA entries were filtered out during data loading
- Language and domain metadata were removed to ensure schema consistency
- Final split: 90% training, 10% validation (randomly sampled with seed 42)

### 1.3 Data Statistics
The combined dataset contains samples from:
- **Languages**: 6 (eng, zho, jpn, rus, tat, ukr)
- **Domains**: 4 (restaurant, laptop, finance, hotel)
- **VA Range**: Valence and Arousal values constrained to [1.0, 9.0]

## 2. Model Architecture

### 2.1 Base Model
We employed a transformer-based regression architecture using **microsoft/mdeberta-v3-base** as the backbone encoder. Alternative models tested included:
- XLM-RoBERTa-large
- XLM-RoBERTa-base  
- BERT-base-multilingual-cased

### 2.2 Architecture Design
Our `TransformerVARegressor` consists of:
1. **Backbone**: Pretrained transformer model for multilingual text encoding
2. **Dropout Layer**: Applied to the [CLS] token representation (p=0.1 for the regression head)
3. **Regression Head**: Linear layer mapping hidden states to 2 outputs (Valence and Arousal)

**Key Modification**: Following best practices from SIGHAN papers, we disabled all dropout layers within the transformer backbone by setting dropout probability to 0.0, preventing information loss during encoding.

### 2.3 Input Representation
Text inputs were formatted with explicit language and domain markers:
```
[LANGUAGE] [DOMAIN] aspect: text
```
For example: `[ENG] [LAPTOP] keyboard: The keyboard is good`

Tokenization specifications:
- Maximum sequence length: 128 tokens
- Padding: max_length
- Truncation: enabled

## 3. Training Configuration

### 3.1 Hyperparameters
- **Batch Size**: 16
- **Learning Rate**: 2×10⁻⁵
- **Epochs**: 10 (maximum)
- **Optimizer**: AdamW
- **Loss Function**: Mean Squared Error (MSE)
- **Gradient Clipping**: max_norm = 1.0

### 3.2 Learning Rate Scheduling
We implemented a linear learning rate schedule with warmup:
- **Warmup Steps**: 10% of total training steps
- **Schedule Type**: Linear decay after warmup
- **Update Frequency**: Per-batch (scheduler stepped with optimizer)

### 3.3 Regularization and Early Stopping
- Early stopping patience: 5 epochs
- Best model selection based on validation loss
- Model checkpointing: saved when validation loss improved

### 3.4 Hardware and Software
- **Framework**: PyTorch with Hugging Face Transformers (v4.41.2)
- **Device**: CUDA-enabled GPU (when available)
- **Additional Libraries**: 
  - scikit-learn for data splitting
  - pandas and numpy for data manipulation
  - scipy for evaluation metrics

## 4. Evaluation Metrics

We evaluated model performance using three metrics as specified by the task:

1. **PCC_V** (Pearson Correlation Coefficient for Valence): Measures linear correlation between predicted and gold valence scores

2. **PCC_A** (Pearson Correlation Coefficient for Arousal): Measures linear correlation between predicted and gold arousal scores

3. **RMSE_VA** (Root Mean Squared Error for VA): 
   ```
   RMSE_VA = √(Σ((V_pred + A_pred) - (V_gold + A_gold))² / N)
   ```

### 4.1 Prediction Post-processing
Final predictions were clipped to the valid range [1, 9] to ensure compliance with task requirements and prevent out-of-range predictions.

## 5. Experimental Procedure

### 5.1 Training Process
1. Data loaded from remote repositories via HTTPS
2. Data normalized and converted to pandas DataFrames
3. Train-validation split performed (90-10)
4. Model initialized with pretrained weights
5. Training loop with per-batch optimization and scheduling
6. Validation performed after each epoch
7. Best model saved based on validation loss
8. Early stopping triggered if no improvement for 5 consecutive epochs

### 5.2 Evaluation Process
1. Best checkpoint loaded from disk
2. Predictions generated on validation set for each language-domain combination
3. Metrics computed per language-domain pair
4. Predictions clipped to [1, 9] range
5. Results formatted as JSONL files with Aspect_VA structure

### 5.3 Submission Format
Predictions were formatted as JSONL files with the following structure:
```json
{
  "ID": "sample_id",
  "Aspect_VA": [
    {"Aspect": "aspect_term", "VA": "valence#arousal"}
  ]
}
```

Files were organized by subtask and compressed for submission.

## 6. Reproducibility

To ensure reproducibility:
- Random seed set to 42 for train-test split
- Model checkpoints saved to Hugging Face Hub
- All data sourced from public repositories with version control
- Complete training configuration preserved in code
- Token embeddings resized to accommodate tokenizer vocabulary

**Model Repository**: Models were uploaded to Hugging Face Hub under the identifier `hassanshahzad2003/{model_name}_aug_density` for public access and reproducibility.
