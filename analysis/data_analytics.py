# Load configurations
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns

subtask = "subtask_1"
task = "task1"
langs =  ["eng", "zho", "jpn", "rus", "tat", "ukr"]
domains = ["restaurant", "laptop", "finance", "hotel"]

# Global variables to store data
all_train = []
lang_data = {}

def load_jsonl_url(url):
    """Load JSONL data from URL."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return []
        lines = response.text.strip().split('\n')
        return [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

def load_all_languages():
    """Load DimABSA 2026 data for all languages and domains."""
    print("--- Loading DimABSA 2026 Data (All Languages) ---")
    for lang in langs:
        lang_samples = []
        for domain in domains:
            if domain == "finance":
                specified_task = "task1"
            else:
                specified_task = "alltasks"
            
            train_url = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_train_{specified_task}.jsonl"
            try:
                train_raw = load_jsonl_url(train_url)
                if train_raw:
                    lang_samples.extend(train_raw)
                    all_train.extend(train_raw)
                    print(f"✅ Loaded {lang}-{domain}: {len(train_raw)} samples")
            except Exception as e:
                print(f"⚠️ Skipped {lang}-{domain}: {e}")
        
        lang_data[lang] = lang_samples
        print(f"📊 {lang.upper()}: {len(lang_samples)} total samples\n")

# Load the data
load_all_languages()

# Extract VA scores
valence_scores = []
arousal_scores = []

for sample in all_train:
    for quad in sample.get('Quadruplet', []):
        va_str = quad.get('VA', '')
        if '#' in va_str:
            try:
                v, a = va_str.split('#')
                valence_scores.append(float(v))
                arousal_scores.append(float(a))
            except ValueError:
                continue

print(f"Total VA pairs extracted: {len(valence_scores)}")

# Plot VA score distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(valence_scores, bins=50, kde=True)
plt.title('Valence Score Distribution')
plt.xlabel('Valence Score')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(arousal_scores, bins=50, kde=True)
plt.title('Arousal Score Distribution')
plt.xlabel('Arousal Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('va_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some samples
print("--- Sample Data ---")
for lang in langs:
    if lang_data[lang]:
        print(f"\n{lang.upper()} samples:")
        for i, sample in enumerate(lang_data[lang][:3]):  # Print first 3 samples per language
            print(f"Sample {i+1}: {sample}")
        break  # Just print for first language to avoid too much output
