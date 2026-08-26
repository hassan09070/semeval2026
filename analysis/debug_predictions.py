"""
Debug script to identify why predictions are all 9.00#9.00
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import requests

# ==================== CONFIG ====================
MODEL_REPO = "hassanshahzad2003/mdeberta-v3-base_aug_density"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


# ==================== DEBUG ====================
def main():
    print("\n" + "=" * 60)
    print("STEP 1: Loading Model and Tokenizer")
    print("=" * 60)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=False)
        print(f"✅ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return

    try:
        model = TransformerVARegressor(model_name=MODEL_REPO).to(device)
        print(f"✅ Model architecture created")
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return

    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename="full_model.bin")
        print(f"✅ Model path found: {model_path}")
        
        state_dict = torch.load(model_path, map_location=device)
        print(f"✅ State dict loaded, keys: {len(state_dict.keys())}")
        
        # Check what keys are in the state dict
        print("\nState dict keys sample:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  - {key}")
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Model weights loaded")
        print(f"   Missing keys: {len(missing)}")
        print(f"   Unexpected keys: {len(unexpected)}")
        
        if missing:
            print(f"   Missing: {missing[:3]}")
        if unexpected:
            print(f"   Unexpected: {unexpected[:3]}")
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return

    model.eval()
    print(f"✅ Model set to eval mode")

    # ==================== STEP 2: Test on sample data ====================
    print("\n" + "=" * 60)
    print("STEP 2: Testing on Sample Data")
    print("=" * 60)

    # Load sample data
    try:
        url = "https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl"
        response = requests.get(url)
        lines = [json.loads(line) for line in response.text.strip().split('\n') if line]
        print(f"✅ Loaded {len(lines)} samples from test data")
        
        # Convert to DataFrame
        if 'Quadruplet' in lines[0]:
            df = pd.json_normalize(lines, 'Quadruplet', ['ID', 'Text'])
            df[['Valence', 'Arousal']] = df['VA'].str.split('#', expand=True).astype(float)
        else:
            print("❌ Unexpected data format")
            return
            
        # Normalize VA to -1 to 1 range
        df['Valence'] = (df['Valence'] - 5) / 4
        df['Arousal'] = (df['Arousal'] - 5) / 4
        
        print(f"✅ DataFrame created: {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"\n   Sample row:")
        print(f"   ID: {df.iloc[0]['ID']}")
        print(f"   Text: {df.iloc[0]['Text'][:100]}...")
        print(f"   Aspect: {df.iloc[0]['Aspect']}")
        print(f"   Valence (gold): {df.iloc[0]['Valence']:.3f}")
        print(f"   Arousal (gold): {df.iloc[0]['Arousal']:.3f}")
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return

    # ==================== STEP 3: Make predictions ====================
    print("\n" + "=" * 60)
    print("STEP 3: Making Predictions")
    print("=" * 60)

    try:
        dataset = VADataset(df.head(5), tokenizer, "eng", "restaurant", max_len=128)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        print(f"✅ Dataset created with 5 samples")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Get raw predictions
                raw_output = model(input_ids, attention_mask)
                raw_preds = raw_output.cpu().numpy()
                
                # Apply scaling
                scaled_v = raw_preds[0, 0] * 4 + 5
                scaled_a = raw_preds[0, 1] * 4 + 5
                clipped_v = np.clip(scaled_v, 1, 9)
                clipped_a = np.clip(scaled_a, 1, 9)
                
                print(f"\n   Sample {i+1}:")
                print(f"   Raw output:  V={raw_preds[0, 0]:.6f}, A={raw_preds[0, 1]:.6f}")
                print(f"   Scaled:      V={scaled_v:.6f}, A={scaled_a:.6f}")
                print(f"   Clipped:     V={clipped_v:.2f}, A={clipped_a:.2f}")
                print(f"   Gold:        V={df.iloc[i]['Valence']:.6f}, A={df.iloc[i]['Arousal']:.6f}")
        
        print(f"\n✅ Predictions complete")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== DIAGNOSIS ====================
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("""
If you see:
- Raw output: close to 1.0 for everything → Model outputs are correct but always the same
- Scaled: 9.0 for everything → This is because 1.0 * 4 + 5 = 9.0 (after clipping)
- Clipped: 9.0 for everything → Model is likely not trained properly

POSSIBLE FIXES:
1. Check if model was actually trained (best_model.bin saved correctly)
2. Try retraining with better learning rate or more epochs
3. Verify the model_name in HuggingFace repo is correct
4. Check if backbone is properly initialized before training
    """)


if __name__ == "__main__":
    main()
