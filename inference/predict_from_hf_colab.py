"""
Standalone prediction script for Google Colab
Loads model from Hugging Face and generates predictions
"""

# ==================== COLAB SETUP ====================
# Run this cell first to install dependencies

# !pip install -q transformers torch scikit-learn pandas numpy tqdm scipy huggingface_hub

# Optional: Mount Google Drive to save results
# from google.colab import drive
# drive.mount('/content/drive')

# ==================== IMPORTS ====================
import json
import os
import re
import shutil
import zipfile
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr
import math

# ==================== CONFIG ====================
MODEL_REPO = "hassanshahzad2003/mdeberta-v3-base_aug_density"
SUBTASK = "subtask_1"
TASK = "task1"
LANGS = ["eng", "zho", "jpn", "rus", "tat", "ukr"]
DOMAINS = ["restaurant", "laptop", "finance", "hotel"]
BATCH_SIZE = 64
MAX_LEN = 128

# Change OUTPUT_DIR for Colab (Google Drive or /content)
OUTPUT_DIR = "/content"  # Change to "/content/drive/My Drive" to save to Drive
# OUTPUT_DIR = "/content/drive/My Drive/semeval_predictions"  # For Google Drive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")


# ==================== UTILITY FUNCTIONS ====================
def load_jsonl_url(url):
    """Fetches and parses a JSONL file from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = [json.loads(line) for line in response.text.strip().split('\n') if line]
        return data
    except Exception as e:
        print(f"Error loading JSONL from {url}: {e}")
        return None


def jsonl_to_df(data):
    """Convert JSONL data to DataFrame with normalized VA values."""
    if 'Quadruplet' in data[0]:
        df = pd.json_normalize(data, 'Quadruplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Category', 'Opinion'])
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')

    elif 'Triplet' in data[0]:
        df = pd.json_normalize(data, 'Triplet', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA', 'Opinion'])
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')

    elif 'Aspect' in data[0]:
        df = pd.json_normalize(data, 'Aspect', ['ID', 'Text'])
        df = df.rename(columns={df.columns[0]: "Aspect"})
        df['Valence'] = 0
        df['Arousal'] = 0

    elif 'Aspect_VA' in data[0]:
        df = pd.json_normalize(data, 'Aspect_VA', ['ID', 'Text'])
        df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        df = df.drop(columns=['VA'])
        df = df.drop_duplicates(subset=['ID', 'Aspect'], keep='first')
    else:
        raise ValueError("Invalid format: must include 'Quadruplet' or 'Triplet' or 'Aspect'")

    # Scale VA labels from 1-9 to -1 to +1
    if 'Valence' in df.columns:
        df['Valence'] = (df['Valence'] - 5) / 4
        df['Arousal'] = (df['Arousal'] - 5) / 4

    return df


def extract_num(s):
    """Extract trailing number from ID string."""
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1


def df_to_jsonl(df, out_path):
    """Convert predictions DataFrame to JSONL format."""
    df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    grouped = df_sorted.groupby("ID", sort=False)

    with open(out_path, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {
                "ID": gid,
                "Aspect_VA": []
            }
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append({
                    "Aspect": row["Aspect"],
                    "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                })
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==================== DATASET ====================
class VADataset(Dataset):
    """PyTorch Dataset for Valence-Arousal regression."""
    def __init__(self, dataframe, tokenizer, lang, domain, max_len=128):
        self.sentences = dataframe["Text"].tolist()
        self.aspects = dataframe["Aspect"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang
        self.domain = domain

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        text = f"[{self.lang.upper()}] [{self.domain.upper()}] {self.aspects[idx]}: {self.sentences[idx]}"

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


# ==================== MODEL ====================
class TransformerVARegressor(nn.Module):
    """BERT-based regressor for predicting Valence and Arousal scores."""
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self._disable_backbone_dropout()
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(self.backbone.config.hidden_size, 2)

    def _disable_backbone_dropout(self):
        """Set all dropout probabilities in backbone to zero."""
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(cls_output)
        return self.reg_head(x)


# ==================== PREDICTION ====================
def get_predictions(model, dataloader):
    """Get predictions from model."""
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask).cpu().numpy()
            all_preds.append(outputs)
    
    preds = np.vstack(all_preds)
    pred_v = preds[:, 0]
    pred_a = preds[:, 1]

    # Map predictions back to 1-9 scale and clip
    pred_v = np.clip(pred_v * 4 + 5, 1, 9)
    pred_a = np.clip(pred_a * 4 + 5, 1, 9)

    return pred_v, pred_a


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("Loading Model from Hugging Face")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
    print(f"✅ Tokenizer loaded from {MODEL_REPO}")

    # Load model
    model = TransformerVARegressor(model_name=MODEL_REPO).to(device)
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="full_model.bin")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded and ready for prediction")

    print("\n" + "=" * 60)
    print("Loading Data from GitHub")
    print("=" * 60)

    predict_raw = {}
    for lang in LANGS:
        for domain in DOMAINS:
            if domain == "finance":
                specified_task = "task1"
            else:
                specified_task = "alltasks"

            predict_url = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{SUBTASK}/{lang}/{lang}_{domain}_dev_{TASK}.jsonl"

            key = f"{lang}_{domain}"
            raw_predict_data = load_jsonl_url(predict_url)
            if raw_predict_data:
                predict_raw[key] = raw_predict_data
                print(f"✅ Loaded: {key}")
            else:
                print(f"⚠️  Skipped: {key}")

    print("\n" + "=" * 60)
    print("Generating Predictions")
    print("=" * 60)

    predict_df = {}
    for lang in LANGS:
        for domain in DOMAINS:
            try:
                key = f"{lang}_{domain}"
                predict_df[key] = jsonl_to_df(predict_raw[key])
            except KeyError:
                continue

    # Create output directory
    subtask_dir = os.path.join(OUTPUT_DIR, SUBTASK)
    os.makedirs(subtask_dir, exist_ok=True)

    # Generate predictions for each language-domain pair
    for lang in LANGS:
        for domain in DOMAINS:
            try:
                key = f"{lang}_{domain}"
                df = predict_df[key]
                
                # Create dataset and dataloader
                pred_dataset = VADataset(df, tokenizer, lang, domain, max_len=MAX_LEN)
                pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                # Get predictions
                pred_v, pred_a = get_predictions(model, pred_loader)
                
                # Update dataframe
                predict_df[key]["Valence"] = pred_v
                predict_df[key]["Arousal"] = pred_a
                
                # Save to JSONL
                output_file = os.path.join(subtask_dir, f"pred_{lang}_{domain}.jsonl")
                df_to_jsonl(predict_df[key], output_file)
                print(f"✅ Saved: {output_file}")
                
            except KeyError as e:
                print(f"⚠️  Skipped {lang}_{domain}: {e}")
                continue

    print("\n" + "=" * 60)
    print("Creating Submission ZIP")
    print("=" * 60)

    # Create ZIP file
    zip_filename = os.path.join(OUTPUT_DIR, f"{SUBTASK}.zip")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files_in_dir in os.walk(subtask_dir):
            for file in files_in_dir:
                path = os.path.join(root, file)
                zf.write(path, os.path.relpath(path, OUTPUT_DIR))

    print(f"✅ Created: {zip_filename}")
    
    print("\n" + "=" * 60)
    print("Done! Submission ready.")
    print("=" * 60)
    print(f"\nOutput location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
