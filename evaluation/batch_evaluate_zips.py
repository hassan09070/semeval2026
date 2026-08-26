#!/usr/bin/env python3
"""
Batch evaluation script for multiple subtask_1_*.zip files.
Each zip is extracted, evaluated, and results saved to a corresponding .txt file.
"""

import json
import os
import sys
import zipfile
import tempfile
import shutil
import subprocess
from pathlib import Path
from io import StringIO

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

def run_evaluation(pred_dir):
    """
    Run the evaluation script on pred files in the given directory.
    Returns the output as a string.
    """
    try:
        # Get the Python executable from the venv
        import sys
        python_exe = sys.executable
        
        # Run the main evaluation script with the pred directory
        result = subprocess.run(
            [python_exe, "run_evaluation_task1.py", "--pred-dir", pred_dir],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # If the script doesn't support --pred-dir flag, we'll use the old method
        if result.returncode != 0 and "unrecognized arguments" in result.stderr:
            # Copy pred files to subtask_1 temporarily
            target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subtask_1_temp")
            os.makedirs(target_dir, exist_ok=True)
            
            for file in os.listdir(pred_dir):
                if file.startswith("pred_") and file.endswith(".jsonl"):
                    src = os.path.join(pred_dir, file)
                    dst = os.path.join(target_dir, file)
                    shutil.copy2(src, dst)
            
            # Run evaluation
            result = subprocess.run(
                [python_exe, "run_evaluation_task1.py"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # Clean up temp directory
            shutil.rmtree(target_dir, ignore_errors=True)
        
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "✗ Evaluation timed out (>300 seconds)"
    except Exception as e:
        return f"✗ Error running evaluation: {e}"

def batch_evaluate_zips(evaluation_dir):
    """Find and evaluate all subtask_1_*.zip files."""
    os.chdir(evaluation_dir)
    
    # Find all subtask_1_*.zip files
    zip_files = sorted(Path(".").glob("subtask_1_*.zip"))
    
    if not zip_files:
        print("✗ No zip files found starting with 'subtask_1_'")
        return
    
    print(f"Found {len(zip_files)} zip file(s) to evaluate\n")
    
    for zip_path in zip_files:
        zip_name = zip_path.stem  # Name without .zip extension
        txt_output = f"{zip_name}.txt"
        
        print(f"\n{'='*60}")
        print(f"Processing: {zip_path.name}")
        print(f"{'='*60}")
        
        # Create temporary extraction directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract zip
            if not extract_zip(str(zip_path), temp_dir):
                continue
            
            # Find pred files in extracted content
            # They might be in subtask_1 subfolder or directly at root
            pred_dir = None
            if os.path.exists(os.path.join(temp_dir, "subtask_1")):
                pred_dir = os.path.join(temp_dir, "subtask_1")
            else:
                # Check if pred files are at root
                if any(f.startswith("pred_") and f.endswith(".jsonl") for f in os.listdir(temp_dir)):
                    pred_dir = temp_dir
            
            if not pred_dir:
                print(f"✗ No subtask_1 folder or pred files found in {zip_path.name}")
                continue
            
            pred_files = find_pred_files(pred_dir)
            if not pred_files:
                print(f"✗ No pred_*.jsonl files found in {zip_path.name}")
                continue
            
            print(f"Found {len(pred_files)} prediction file(s)")
            
            # Run evaluation
            print(f"Running evaluation...")
            eval_output = run_evaluation(pred_dir)
            
            # Save results to txt file
            with open(txt_output, 'w') as f:
                f.write(f"Evaluation Results for: {zip_path.name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Extracted from: {zip_path.name}\n")
                f.write(f"Prediction files found: {len(pred_files)}\n")
                f.write(f"{'='*60}\n\n")
                f.write(eval_output)
            
            print(f"✓ Results saved to: {txt_output}")

if __name__ == "__main__":
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    batch_evaluate_zips(eval_dir)
    print(f"\n{'='*60}")
    print("Batch evaluation completed!")
    print(f"{'='*60}")
