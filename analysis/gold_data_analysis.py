import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data from gold_data folder
gold_data_path = Path("/Users/hassan/Documents/code/semeval/evaluation/gold_data")
files = sorted(gold_data_path.glob("gold_*.jsonl"))

print("=" * 80)
print("GOLD DATA ANALYSIS - VA DISTRIBUTION")
print("=" * 80)
print(f"\nFound {len(files)} files:\n")
for f in files:
    print(f"  - {f.name}")

# Load all data
all_data = []
language_domain_va = defaultdict(lambda: {"valence": [], "arousal": []})

for file_path in files:
    file_name = file_path.stem  # e.g., gold_eng_laptop
    parts = file_name.replace("gold_", "").split("_")
    language = parts[0]  # e.g., eng
    domain = "_".join(parts[1:]) if len(parts) > 1 else "unknown"  # e.g., laptop
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            for aspect_va in record.get("Aspect_VA", []):
                va_str = aspect_va.get("VA", "")
                if "#" in va_str:
                    try:
                        valence, arousal = map(float, va_str.split("#"))
                        all_data.append({
                            "language": language,
                            "domain": domain,
                            "aspect": aspect_va.get("Aspect", ""),
                            "valence": valence,
                            "arousal": arousal,
                            "text": record.get("Text", "")
                        })
                        language_domain_va[f"{language}_{domain}"]["valence"].append(valence)
                        language_domain_va[f"{language}_{domain}"]["arousal"].append(arousal)
                    except ValueError:
                        continue

df = pd.DataFrame(all_data)

# ============================================================================
# OVERALL STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)

print(f"\nTotal records: {len(df)}")
print(f"Number of languages: {df['language'].nunique()}")
print(f"Number of domains: {df['domain'].nunique()}")

print("\n--- VALENCE DISTRIBUTION ---")
print(f"Mean: {df['valence'].mean():.2f}")
print(f"Median: {df['valence'].median():.2f}")
print(f"Std Dev: {df['valence'].std():.2f}")
print(f"Min: {df['valence'].min():.2f}")
print(f"Max: {df['valence'].max():.2f}")
print(f"Q1 (25%): {df['valence'].quantile(0.25):.2f}")
print(f"Q3 (75%): {df['valence'].quantile(0.75):.2f}")

print("\n--- AROUSAL DISTRIBUTION ---")
print(f"Mean: {df['arousal'].mean():.2f}")
print(f"Median: {df['arousal'].median():.2f}")
print(f"Std Dev: {df['arousal'].std():.2f}")
print(f"Min: {df['arousal'].min():.2f}")
print(f"Max: {df['arousal'].max():.2f}")
print(f"Q1 (25%): {df['arousal'].quantile(0.25):.2f}")
print(f"Q3 (75%): {df['arousal'].quantile(0.75):.2f}")

# ============================================================================
# LANGUAGE & DOMAIN BREAKDOWN
# ============================================================================
print("\n" + "=" * 80)
print("LANGUAGE-DOMAIN BREAKDOWN")
print("=" * 80)

lang_domain_stats = []
for key in sorted(language_domain_va.keys()):
    valences = language_domain_va[key]["valence"]
    arousals = language_domain_va[key]["arousal"]
    
    lang_domain_stats.append({
        "Language_Domain": key,
        "Count": len(valences),
        "Valence_Mean": np.mean(valences),
        "Valence_Std": np.std(valences),
        "Arousal_Mean": np.mean(arousals),
        "Arousal_Std": np.std(arousals)
    })

lang_domain_df = pd.DataFrame(lang_domain_stats)
print("\n" + lang_domain_df.to_string(index=False))

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("VA Distribution Analysis - Gold Data", fontsize=16, fontweight='bold')

# 1. Histogram - Valence
axes[0, 0].hist(df['valence'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Valence Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Valence Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df['valence'].mean(), color='red', linestyle='--', label=f"Mean: {df['valence'].mean():.2f}")
axes[0, 0].legend()

# 2. Histogram - Arousal
axes[0, 1].hist(df['arousal'], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Arousal Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Arousal Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['arousal'].mean(), color='red', linestyle='--', label=f"Mean: {df['arousal'].mean():.2f}")
axes[0, 1].legend()

# 3. Scatter - Valence vs Arousal
axes[0, 2].scatter(df['valence'], df['arousal'], alpha=0.5, s=20)
axes[0, 2].set_title('Valence vs Arousal', fontweight='bold')
axes[0, 2].set_xlabel('Valence Score')
axes[0, 2].set_ylabel('Arousal Score')
axes[0, 2].grid(True, alpha=0.3)

# 4. Box plot by Language
df_sorted = df.sort_values('language')
sns.boxplot(data=df_sorted, x='language', y='valence', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Valence by Language', fontweight='bold')
axes[1, 0].set_xlabel('Language')
axes[1, 0].set_ylabel('Valence Score')

# 5. Box plot by Domain
df_sorted = df.sort_values('domain')
sns.boxplot(data=df_sorted, x='domain', y='arousal', ax=axes[1, 1], palette='Set3')
axes[1, 1].set_title('Arousal by Domain', fontweight='bold')
axes[1, 1].set_xlabel('Domain')
axes[1, 1].set_ylabel('Arousal Score')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Count by Language-Domain
lang_domain_count = df.groupby(['language', 'domain']).size().reset_index(name='count')
pivot_count = lang_domain_count.pivot(index='language', columns='domain', values='count').fillna(0)
pivot_count.plot(kind='bar', ax=axes[1, 2], width=0.8)
axes[1, 2].set_title('Record Count by Language-Domain', fontweight='bold')
axes[1, 2].set_xlabel('Language')
axes[1, 2].set_ylabel('Count')
axes[1, 2].legend(title='Domain', fontsize=8, title_fontsize=8)
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/hassan/Documents/code/semeval/va_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: va_distribution_analysis.png")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

correlation = df['valence'].corr(df['arousal'])
print(f"\nValence-Arousal Correlation: {correlation:.4f}")

# Additional correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df[['valence', 'arousal']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Matrix - Valence & Arousal', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('/Users/hassan/Documents/code/semeval/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")

# ============================================================================
# DISTRIBUTION BY QUADRANT (Emotion Space)
# ============================================================================
print("\n" + "=" * 80)
print("EMOTION SPACE QUADRANTS")
print("=" * 80)

valence_median = df['valence'].median()
arousal_median = df['arousal'].median()

quadrants = {
    'High Valence + High Arousal (Happy)': len(df[(df['valence'] >= valence_median) & (df['arousal'] >= arousal_median)]),
    'High Valence + Low Arousal (Calm)': len(df[(df['valence'] >= valence_median) & (df['arousal'] < arousal_median)]),
    'Low Valence + High Arousal (Angry)': len(df[(df['valence'] < valence_median) & (df['arousal'] >= arousal_median)]),
    'Low Valence + Low Arousal (Sad)': len(df[(df['valence'] < valence_median) & (df['arousal'] < arousal_median)])
}

print(f"\nValence Median: {valence_median:.2f}")
print(f"Arousal Median: {arousal_median:.2f}\n")

for quadrant, count in sorted(quadrants.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(df)) * 100
    print(f"{quadrant}: {count:5d} ({percentage:5.1f}%)")

# Visualize quadrants
fig, ax = plt.subplots(figsize=(10, 8))
colors = df.apply(lambda row: 'green' if row['valence'] >= valence_median and row['arousal'] >= arousal_median
                  else 'yellow' if row['valence'] >= valence_median and row['arousal'] < arousal_median
                  else 'red' if row['valence'] < valence_median and row['arousal'] >= arousal_median
                  else 'blue', axis=1)

ax.scatter(df['valence'], df['arousal'], c=colors, alpha=0.6, s=30)
ax.axvline(valence_median, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(arousal_median, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Add quadrant labels
ax.text(0.98, 0.98, 'Happy\n(High V, High A)', transform=ax.transAxes, 
        ha='right', va='top', fontsize=10, color='green', fontweight='bold', alpha=0.7)
ax.text(0.02, 0.98, 'Calm\n(High V, Low A)', transform=ax.transAxes, 
        ha='left', va='top', fontsize=10, color='orange', fontweight='bold', alpha=0.7)
ax.text(0.98, 0.02, 'Angry\n(Low V, High A)', transform=ax.transAxes, 
        ha='right', va='bottom', fontsize=10, color='red', fontweight='bold', alpha=0.7)
ax.text(0.02, 0.02, 'Sad\n(Low V, Low A)', transform=ax.transAxes, 
        ha='left', va='bottom', fontsize=10, color='blue', fontweight='bold', alpha=0.7)

ax.set_xlabel('Valence Score', fontsize=12)
ax.set_ylabel('Arousal Score', fontsize=12)
ax.set_title('Emotion Space - Valence-Arousal Distribution', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/hassan/Documents/code/semeval/emotion_space_quadrants.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: emotion_space_quadrants.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  1. va_distribution_analysis.png - Main visualizations")
print("  2. correlation_heatmap.png - Correlation matrix")
print("  3. emotion_space_quadrants.png - Emotion quadrants")
print("\n" + "=" * 80)
