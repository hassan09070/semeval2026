#!/usr/bin/env python3
"""
Single zip file evaluation script.
Extracts a zip, evaluates predictions, and generates:
1. Text results file
2. JSON file with per-datapoint RMSE (sorted in decreasing order)
"""

import json
import os
import sys
import zipfile
import tempfile
import shutil
import subprocess
import math
from pathlib import Path
from io import StringIO
from scipy.stats import pearsonr

def extract_zip(zip_path, extract_to):
    """Extract zip file to temporary directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✓ Extracted: {zip_path}")
        return True
    except Exception as e:
        print(f"✗ Error extracting {zip_path}: {e}")
        return False

def find_pred_files(base_dir):
    """Find all pred_*.jsonl files in the directory structure."""
    pred_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("pred_") and file.endswith(".jsonl"):
                pred_files.append(os.path.join(root, file))
    return pred_files

def read_jsonl_file(file_path, task=1, data_type='pred'):
    """Reads a JSONL file and processes each line."""
    key_name = {1: "Aspect_VA", 2: "Triplet", 3: 'Quadruplet'}
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
                            
                            parsed_quadruplets.append(quad)
                        entry[output_key] = parsed_quadruplets
                    
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"An unknown error occurred while processing line {line_num}: {e}")
                    continue

    except Exception as e:
        print(f"An error occurred while reading file '{file_path}': {e}")
        return data

    return data

def convert_task1_data_with_datapoint_tracking(gold_data, pred_data):
    """Convert task 1 data with per-datapoint tracking."""
    gold_data_dict = {entry['ID']: entry for entry in gold_data}
    pred_data_dict = {entry['ID']: entry for entry in pred_data}
    
    # Store values per datapoint
    datapoint_values = {}
    
    for key, value in gold_data_dict.items():
        gold_value = value["Aspect_VA"]
        if key not in pred_data_dict:
            print(f"Error: Prediction missing for ID '{key}'!")
            continue
        
        pred_data_entry = pred_data_dict[key]
        pred_value = pred_data_entry["Aspect_VA"]
        pred_value_dict = {entry['Aspect']: entry for entry in pred_value}
        
        # Initialize structure for this datapoint
        datapoint_values[key] = {
            'text': value.get('Text', ''),  # Original text
            'gold_v': [],
            'gold_a': [],
            'pred_v': [],
            'pred_a': [],
            'gold_aspect_va': [],  # Full gold aspect_va
            'pred_aspect_va': []   # Full pred aspect_va
        }
        
        for item in gold_value:
            gold_va = item['VA'].split("#")
            v_val = float(gold_va[0])
            a_val = float(gold_va[1])
            aspect = item.get('Aspect', '')
            
            datapoint_values[key]['gold_v'].append(v_val)
            datapoint_values[key]['gold_a'].append(a_val)
            
            # Store complete gold aspect_va
            datapoint_values[key]['gold_aspect_va'].append({
                'Aspect': aspect,
                'VA': f"{v_val}#{a_val}"
            })
            
            if aspect in pred_value_dict:
                pred_item = pred_value_dict[aspect]
                pred_va = pred_item['VA'].split("#")
                pred_v = float(pred_va[0])
                pred_a = float(pred_va[1])
                
                datapoint_values[key]['pred_v'].append(pred_v)
                datapoint_values[key]['pred_a'].append(pred_a)
                
                # Store complete predicted aspect_va
                datapoint_values[key]['pred_aspect_va'].append({
                    'Aspect': aspect,
                    'VA': f"{pred_v}#{pred_a}"
                })
            else:
                print(f"Error: VA value is missing for ID '{key}', Aspect '{aspect}'!")
                continue
    
    return datapoint_values

def calculate_rmse_per_datapoint(datapoint_values, is_norm=True):
    """Calculate RMSE for each datapoint using the official formula."""
    
    rmse_summary = []
    D_max = math.sqrt(128)
    
    for datapoint_id, values in datapoint_values.items():
        gold_v = values['gold_v']
        gold_a = values['gold_a']
        pred_v = values['pred_v']
        pred_a = values['pred_a']
        
        if not gold_v:
            continue
        
        # Combine V and A values (official formula)
        gold_va = gold_v + gold_a
        pred_va = pred_v + pred_a
        
        # Calculate squared errors
        squared_errors = [(pred - gold) ** 2 for pred, gold in zip(pred_va, gold_va)]
        
        # RMSE formula: sqrt(sum(squared_errors) / len(gold_v))
        mse = sum(squared_errors) / len(gold_v)
        rmse = math.sqrt(mse)
        
        # Normalize if needed
        if is_norm:
            rmse = rmse / D_max
        
        rmse_summary.append({
            'ID': datapoint_id,
            'Text': values['text'],
            'Gold_Labels': values['gold_aspect_va'],
            'Predicted_Labels': values['pred_aspect_va'],
            'RMSE': round(rmse, 6)
        })
    
    # Sort by RMSE in descending order (highest RMSE first)
    rmse_summary.sort(key=lambda x: x['RMSE'], reverse=True)
    
    return rmse_summary

def run_evaluation(pred_dir, gold_dir):
    """Run the evaluation and extract per-datapoint metrics."""
    try:
        import sys
        python_exe = sys.executable
        
        # Find all pred_*.jsonl files
        pred_files = find_pred_files(pred_dir)
        
        if not pred_files:
            return None, "No prediction files found"
        
        # Initialize overall results
        all_rmse_data = []
        
        # Process each prediction file
        for pred_file in pred_files:
            filename = os.path.basename(pred_file)
            print(f"Processing: {filename}")
            
            # Extract lang and domain from filename (e.g., pred_eng_laptop.jsonl)
            # Format: pred_{lang}_{domain}.jsonl
            parts = filename.replace('pred_', '').replace('.jsonl', '').split('_')
            if len(parts) >= 2:
                lang = parts[0]
                domain = '_'.join(parts[1:])  # Handle multi-word domains
            else:
                lang = 'unknown'
                domain = 'unknown'
            
            # Find corresponding gold file
            gold_file = os.path.join(gold_dir, filename.replace('pred_', 'gold_'))
            
            if not os.path.exists(gold_file):
                print(f"Warning: Gold file not found for {filename}, skipping...")
                continue
            
            # Read data
            print(f"  Loading gold data...")
            gold_data = read_jsonl_file(gold_file, task=1, data_type='gold')
            print(f"  Loading prediction data...")
            pred_data = read_jsonl_file(pred_file, task=1, data_type='pred')
            
            # Convert and evaluate with per-datapoint tracking
            datapoint_values = convert_task1_data_with_datapoint_tracking(gold_data, pred_data)
            
            if not datapoint_values:
                print(f"Warning: No valid data extracted from {filename}")
                continue
            
            # Calculate per-datapoint RMSE
            rmse_data = calculate_rmse_per_datapoint(datapoint_values, is_norm=True)
            
            # Add lang and domain to each result
            for item in rmse_data:
                item['lang'] = lang
                item['domain'] = domain
            
            all_rmse_data.extend(rmse_data)
            
            print(f"  ✓ Processed {len(rmse_data)} datapoints (lang={lang}, domain={domain})")
        
        # Sort all RMSE data by RMSE (descending)
        all_rmse_data.sort(key=lambda x: x['RMSE'], reverse=True)
        
        return all_rmse_data, None
        
    except Exception as e:
        return None, str(e)

def evaluate_single_zip(zip_path):
    """Evaluate a single zip file and generate results."""
    
    if not os.path.exists(zip_path):
        print(f"✗ Zip file not found: {zip_path}")
        return
    
    zip_name = Path(zip_path).stem
    txt_output = f"{zip_name}.txt"
    json_output = f"{zip_name}_rmse.json"
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(zip_path).name}")
    print(f"{'='*60}")
    
    # Create temporary extraction directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract zip
        if not extract_zip(zip_path, temp_dir):
            return
        
        # Find pred directory
        pred_dir = None
        if os.path.exists(os.path.join(temp_dir, "subtask_1")):
            pred_dir = os.path.join(temp_dir, "subtask_1")
        else:
            if any(f.startswith("pred_") and f.endswith(".jsonl") for f in os.listdir(temp_dir)):
                pred_dir = temp_dir
        
        if not pred_dir:
            print(f"✗ No subtask_1 folder or pred files found in {Path(zip_path).name}")
            return
        
        pred_files = find_pred_files(pred_dir)
        if not pred_files:
            print(f"✗ No pred_*.jsonl files found in {Path(zip_path).name}")
            return
        
        print(f"Found {len(pred_files)} prediction file(s)")
        
        # Get gold data directory
        gold_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(gold_dir, "gold_data")):
            os.makedirs(os.path.join(gold_dir, "gold_data"), exist_ok=True)
        gold_data_dir = os.path.join(gold_dir, "gold_data")
        
        # Run evaluation
        print(f"Running evaluation...")
        rmse_results, error = run_evaluation(pred_dir, gold_data_dir)
        
        if error:
            print(f"✗ Evaluation error: {error}")
            return
        
        # Save text results
        with open(txt_output, 'w') as f:
            f.write(f"Evaluation Results for: {Path(zip_path).name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Extracted from: {Path(zip_path).name}\n")
            f.write(f"Prediction files found: {len(pred_files)}\n")
            f.write(f"{'='*60}\n\n")
            if rmse_results:
                f.write(f"Per-Datapoint RMSE (sorted by highest RMSE first):\n")
                f.write(f"{'='*60}\n")
                for item in rmse_results:
                    num_aspects = len(item.get('Gold_Labels', []))
                    lang = item.get('lang', 'unknown')
                    domain = item.get('domain', 'unknown')
                    f.write(f"ID: {item['ID']:<30} Lang: {lang:<6} Domain: {domain:<12} RMSE: {item['RMSE']:.6f} ({num_aspects} aspects)\n")
        
        print(f"✓ Results saved to: {txt_output}")
        
        # Save JSON results
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(rmse_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ RMSE data saved to: {json_output}")
        print(f"\n{'='*60}")
        print(f"Evaluation completed!")
        print(f"Total datapoints: {len(rmse_results)}")
        if rmse_results:
            print(f"Highest RMSE: {rmse_results[0]['RMSE']:.6f}")
            print(f"Lowest RMSE: {rmse_results[-1]['RMSE']:.6f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    # Hardcoded zip file path
    zip_path = "/Users/hassan/Documents/code/semeval/evaluation/subtask_1_aug_rescale.zip"
    
    evaluate_single_zip(zip_path)
