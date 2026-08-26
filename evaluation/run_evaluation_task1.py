import json
import os
import sys
import argparse
import urllib.request
from scipy.stats import pearsonr
import math

# Configuration
langs = ["eng", "zho", "jpn", "rus", "tat", "ukr"]
domains = ["restaurant", "laptop", "finance", "hotel"]

# Gold labels base URL
GOLD_BASE_URL = "https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/subtask_1"

# Mapping of prediction files to their lang/domain
PRED_MAPPING = {
    "pred_eng_laptop.jsonl": ("eng", "laptop"),
    "pred_eng_restaurant.jsonl": ("eng", "restaurant"),
    "pred_jpn_finance.jsonl": ("jpn", "finance"),
    "pred_jpn_hotel.jsonl": ("jpn", "hotel"),
    "pred_rus_restaurant.jsonl": ("rus", "restaurant"),
    "pred_tat_restaurant.jsonl": ("tat", "restaurant"),
    "pred_ukr_restaurant.jsonl": ("ukr", "restaurant"),
    "pred_zho_finance.jsonl": ("zho", "finance"),
    "pred_zho_laptop.jsonl": ("zho", "laptop"),
    "pred_zho_restaurant.jsonl": ("zho", "restaurant"),
}

key_name = {1: "Aspect_VA", 2: "Triplet", 3: 'Quadruplet'}

def download_gold_file(lang, domain, save_dir="gold_data"):
    """Download gold file from GitHub for specific lang/domain combination."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = f"gold_{lang}_{domain}.jsonl"
    filepath = os.path.join(save_dir, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        print(f"✓ Gold file already exists: {filename}")
        return filepath
    
    url = f"{GOLD_BASE_URL}/{lang}/{lang}_{domain}_dev_task1.jsonl"
    
    try:
        print(f"  Downloading: {filename} from {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Downloaded: {filename}")
        return filepath
    except urllib.error.HTTPError as e:
        print(f"✗ Failed to download {filename}: {e}")
        return None

def read_jsonl_file(file_path, task=3, data_type='pred'):
    """Reads a JSONL file from the specified path and processes each line."""
    output_key = key_name[task]
    input_key = key_name[3] if (data_type == 'gold' and task == 2) else key_name[task]
    
    data = []
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return data

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_data = json.loads(line)
                    entry = {
                        'ID': json_data.get('ID', f"Missing_ID_Line{line_num}"),
                        'Text': json_data.get('Text', ''),
                        'Aspect': json_data.get('Aspect', []),
                    }
                    if entry['ID'] == f"Missing_ID_Line{line_num}":
                        print(f"Error: ID value is missing at line {line_num}!")
                        continue
                    
                    quadruplets = json_data.get(input_key, [])
                    if data_type == 'gold' and len(quadruplets) == 0:
                        quadruplets = json_data.get(output_key, [])
                    
                    if isinstance(quadruplets, list):
                        parsed_quadruplets = []
                        for quad in quadruplets:
                            if not isinstance(quad, dict):
                                print(f"Warning: Item at line {line_num} is not a dictionary: {quad}")
                                continue

                            aspect = quad.get('Aspect', 'Unknown_Aspect')
                            va = quad.get('VA', '0.00#0.00')
                            
                            if va == '0.00#0.00':
                                print(f"Error: VA value is missing at line {line_num}!")
                                continue
                            if aspect == 'Unknown_Aspect':
                                print(f"Error: Aspect value is missing at line {line_num}!")
                                continue

                            parsed_quadruplets.append({
                                'Aspect': aspect.lower(),
                                'VA': va
                            })
                        entry[output_key] = parsed_quadruplets
                    else:
                        print(f"Warning: Item at line {line_num} is not a list type: {type(quadruplets)}")
                        entry[output_key] = []

                    data.append(entry)

                except json.JSONDecodeError as e:
                    print(f"JSON parsing error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"An unknown error occurred at line {line_num}: {e}")
                    continue

    except Exception as e:
        print(f"An error occurred while reading file '{file_path}': {e}")
        return data

    return data

def convert_task1_data(gold_data, pred_data):
    """Convert Task 1 data for evaluation (Pearson correlation)."""
    gold_data = {entry['ID']: entry for entry in gold_data}
    pred_data = {entry['ID']: entry for entry in pred_data}
    gold_v, gold_a, pred_v, pred_a = [], [], [], []
    
    for key, value in gold_data.items():
        gold_value = value["Aspect_VA"]
        if key not in pred_data:
            print(f"Warning: Missing prediction for ID {key}")
            continue
        
        pred_value = pred_data[key]["Aspect_VA"]
        pred_value = {entry['Aspect']: entry for entry in pred_value}
        
        for item in gold_value:
            gold_va = item['VA'].split("#")
            gold_v.append(float(gold_va[0]))
            gold_a.append(float(gold_va[1]))
            
            if item['Aspect'] in pred_value:
                pred_va = pred_value[item['Aspect']]["VA"].split("#")
                pred_v.append(float(pred_va[0]))
                pred_a.append(float(pred_va[1]))
            else:
                print(f"Warning: Missing prediction for aspect '{item['Aspect']}' in ID {key}")
                continue
    
    return gold_v, gold_a, pred_v, pred_a

def evaluate_predictions_task1(gold_data, pred_data, is_norm=True):
    """Evaluate Task 1 predictions using Pearson correlation and RMSE."""
    if not gold_data or not pred_data:
        print("Error: Failed to load one or both data files. Cannot perform evaluation.")
        return None
    
    gold_v, gold_a, pred_v, pred_a = convert_task1_data(gold_data, pred_data)
    
    if not gold_v or not pred_v:
        print("Error: No matching predictions found for evaluation.")
        return None
    
    if not (all(1 <= x <= 9 for x in pred_v) and all(1 <= x <= 9 for x in pred_a)):
        print(f"Warning: Some predicted values are out of the numerical range [1-9].")
    
    pcc_v = pearsonr(pred_v, gold_v)[0]
    pcc_a = pearsonr(pred_a, gold_a)[0]
    
    gold_va = gold_v + gold_a
    pred_va = pred_v + pred_a
    
    def rmse_norm(gold_va, pred_va, is_normalization=True):
        result = [(a - b)**2 for a, b in zip(gold_va, pred_va)]
        # if is_normalization:
        #     return math.sqrt(sum(result) / len(gold_v)) / math.sqrt(128)
        return math.sqrt(sum(result) / len(gold_v))
    
    rmse_va = rmse_norm(gold_va, pred_va, is_norm)
    
    return { 
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
        'num_samples': len(gold_v)
    }

def main():
    """Main function to download gold files and run evaluation for all prediction files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Task 1 predictions')
    parser.add_argument('--pred-dir', type=str, default="subtask_1", 
                        help='Directory containing prediction files (default: subtask_1)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("SemEval 2026 - Task 1 Evaluation (Subtask 1)")
    print("=" * 80)
    print(f"Using prediction directory: {args.pred_dir}\n")
    
    # Step 1: Download all gold files
    print("\n[STEP 1] Downloading Gold Label Files...")
    print("-" * 80)
    gold_files = {}
    for pred_file, (lang, domain) in PRED_MAPPING.items():
        gold_path = download_gold_file(lang, domain)
        if gold_path:
            gold_files[pred_file] = gold_path
    
    print(f"\n✓ Downloaded {len(gold_files)}/{len(PRED_MAPPING)} gold files")
    
    # Step 2: Run evaluation for each prediction file
    print("\n[STEP 2] Running Evaluation for Task 1...")
    print("-" * 80)
    
    results_summary = []
    pred_dir = args.pred_dir
    
    for pred_file in sorted(PRED_MAPPING.keys()):
        pred_path = os.path.join(pred_dir, pred_file)
        
        if pred_file not in gold_files:
            print(f"\n✗ Skipping {pred_file}: Gold file not available")
            continue
        
        gold_path = gold_files[pred_file]
        
        if not os.path.exists(pred_path):
            print(f"\n✗ Skipping {pred_file}: Prediction file not found at {pred_path}")
            continue
        
        print(f"\n--- Evaluating {pred_file} ---")
        print(f"  Gold: {gold_path}")
        print(f"  Pred: {pred_path}")
        
        print("  Loading gold data...")
        gold_data = read_jsonl_file(gold_path, task=1, data_type="gold")
        print(f"  Loaded {len(gold_data)} gold records")
        
        print("  Loading prediction data...")
        pred_data = read_jsonl_file(pred_path, task=1, data_type="pred")
        print(f"  Loaded {len(pred_data)} prediction records")
        
        if gold_data and pred_data:
            results = evaluate_predictions_task1(gold_data, pred_data, is_norm=True)
            if results:
                print(f"\n  Results:")
                print(f"    PCC_V (Pearson Valence):  {results['PCC_V']:.4f}")
                print(f"    PCC_A (Pearson Arousal):  {results['PCC_A']:.4f}")
                print(f"    RMSE_VA (Normalized):     {results['RMSE_VA']:.4f}")
                print(f"    Num Samples:              {results['num_samples']}")
                
                results_summary.append({
                    'file': pred_file,
                    'lang': PRED_MAPPING[pred_file][0],
                    'domain': PRED_MAPPING[pred_file][1],
                    'PCC_V': results['PCC_V'],
                    'PCC_A': results['PCC_A'],
                    'RMSE_VA': results['RMSE_VA'],
                    'samples': results['num_samples']
                })
        else:
            print(f"  ✗ Failed to load data files")
    
    # Step 3: Print summary table
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY - Task 1")
    print("=" * 80)
    print(f"{'File':<30} {'Lang':<6} {'Domain':<12} {'PCC_V':<10} {'PCC_A':<10} {'RMSE_VA':<10}")
    print("-" * 80)
    
    for result in results_summary:
        print(f"{result['file']:<30} {result['lang']:<6} {result['domain']:<12} "
              f"{result['PCC_V']:<10.4f} {result['PCC_A']:<10.4f} {result['RMSE_VA']:<10.4f}")
    
    if results_summary:
        avg_pcc_v = sum(r['PCC_V'] for r in results_summary) / len(results_summary)
        avg_pcc_a = sum(r['PCC_A'] for r in results_summary) / len(results_summary)
        avg_rmse = sum(r['RMSE_VA'] for r in results_summary) / len(results_summary)
        
        print("-" * 80)
        print(f"{'AVERAGE':<30} {'':<6} {'':<12} "
              f"{avg_pcc_v:<10.4f} {avg_pcc_a:<10.4f} {avg_rmse:<10.4f}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
