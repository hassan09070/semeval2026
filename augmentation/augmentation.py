
import json
import time
import requests
import os
import shutil
import random
from openai import OpenAI, APIError, RateLimitError
from IPython.display import FileLink

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = os.environ["GROQ_API_KEY"]  # <--- PASTE KEY HERE
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-70b-versatile"

# 1. LOOK AT YOUR RIGHT SIDEBAR TO FIND THE EXACT PATH
# It will be something like: /kaggle/input/my-laptop-data/balanced_aug_laptop_hybrid_v3.jsonl
SOURCE_PATH = "/kaggle/input/data-synth-part1/balanced_aug_laptop_hybrid_v4_FINAL (1).jsonl" 

# 2. WHERE WE WILL WRITE NEW DATA (Writable Folder)
OUTPUT_FILE = "/kaggle/working/balanced_aug_laptop_hybrid.jsonl"

TARGET_COUNT = 200  
DATA_URL = "https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/subtask_2/eng/eng_laptop_train_alltasks.jsonl"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# PART 1: RESTORE THE FILE
# ==========================================
def restore_file():
    if os.path.exists(OUTPUT_FILE):
        print(f"✅ Writable file already exists. Size: {os.path.getsize(OUTPUT_FILE)} bytes")
        return
        
    if os.path.exists(SOURCE_PATH):
        print(f"🔄 Copying file from Input to Working directory...")
        shutil.copy(SOURCE_PATH, OUTPUT_FILE)
        print("✅ Restore Complete! Ready to append new data.")
    else:
        print(f"⚠️ WARNING: Could not find source file at: {SOURCE_PATH}")
        print("   If this is a fresh start, that is fine. If you expected to resume, CHECK THE PATH.")

# ==========================================
# PART 2: GENERATION LOGIC
# ==========================================
def get_sentiment_desc(v, a):
    v, a = float(v), float(a)
    v_desc = "Neutral"
    if v <= 3.5: v_desc = "Negative / Unhappy / Critical"
    elif v >= 6.5: v_desc = "Positive / Happy / Praising"
    
    a_desc = "Moderate Intensity"
    if a <= 3.5: a_desc = "Calm / Passive / Bored"
    elif a >= 6.5: a_desc = "Excited / Intense / Urgent"
    return v_desc, a_desc

def generate_with_backoff(prompt, retries=5):
    wait_time = 2
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful data augmentation assistant. Output only JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            return json.loads(response.choices[0].message.content)
        except RateLimitError:
            print(f"    [Limit] Rate limited. Sleeping {wait_time}s...")
            time.sleep(wait_time)
            wait_time *= 2
        except Exception as e:
            print(f"    [Error] {e}")
            time.sleep(1)
            return None
    return None

def get_bin_data():
    print("Downloading original dataset...")
    response = requests.get(DATA_URL)
    bin_storage = {} 
    
    for line in response.text.splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        for q_idx, quad in enumerate(obj["Quadruplet"]):
            v, a = map(float, quad["VA"].split('#'))
            v_idx = min(8, int(v - 1)) if v < 10 else 8
            a_idx = min(8, int(a - 1)) if a < 10 else 8
            if (v_idx, a_idx) not in bin_storage: bin_storage[(v_idx, a_idx)] = []
            bin_storage[(v_idx, a_idx)].append((obj, q_idx))
    return bin_storage

def load_existing_progress(filepath):
    progress_counts = {}
    if not os.path.exists(filepath): return {}, 0
    
    print("Scanning existing file for progress...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                # Use first quad for binning (same as main logic)
                first_quad = obj["Quadruplet"][0]
                v, a = map(float, first_quad["VA"].split('#'))
                v_idx = min(8, int(v - 1)) if v < 10 else 8
                a_idx = min(8, int(a - 1)) if a < 10 else 8
                
                if (v_idx, a_idx) not in progress_counts: progress_counts[(v_idx, a_idx)] = 0
                progress_counts[(v_idx, a_idx)] += 1
            except: continue
    return progress_counts, 0

def rewrite_sample(seed_tuple):
    seed_obj, target_q_idx = seed_tuple 
    original_text = seed_obj["Text"]
    quad = seed_obj["Quadruplet"][target_q_idx]
    aspect = quad["Aspect"]
    opinion = quad["Opinion"] 
    va_score = quad["VA"]
    v_val, a_val = map(float, va_score.split('#'))
    v_desc, a_desc = get_sentiment_desc(v_val, a_val)
    
    prompt = f"""
    Rewrite the sentence.
    CRITICAL CONSTRAINTS:
    1. Aspect "{aspect}" must appear exactly.
    2. Opinion "{opinion}" must appear exactly.
    3. Tone must be "{v_desc}" (Valence {v_val} and Arousal {a_val}).
    
    Input: "{original_text}"
    Output JSON: {{ "enhanced_sentence": "..." }}
    """
    res = generate_with_backoff(prompt)
    if res and "enhanced_sentence" in res:
        new_text = res["enhanced_sentence"]
        if aspect.lower() not in new_text.lower(): return None
        if opinion.lower() not in new_text.lower(): return None
        return {"text": new_text, "aspect": aspect, "opinion": opinion, "va": va_score, "category": quad["Category"]}
    return None

def generate_fresh(target_v, target_a):
    v_desc, a_desc = get_sentiment_desc(target_v, target_a)
    prompt = f"""
    Generate a Laptop review.
    TARGET: Valence {target_v} ({v_desc}), Arousal {target_a} ({a_desc}).
    MUST contain explicit aspect and opinion.
    Output JSON: {{ "enhanced_sentence": "...", "aspect": "...", "opinion_term": "..." }}
    """
    res = generate_with_backoff(prompt)
    if res and "enhanced_sentence" in res:
        return {"text": res["enhanced_sentence"], "aspect": res.get("aspect", "general"), "opinion": res.get("opinion_term", "unknown"), "va": f"{target_v}#{target_a}", "category": "LAPTOP#GENERAL"}
    return None

def main():
    restore_file() # <--- STEP 1: RESTORE
    
    bin_storage = get_bin_data()
    progress_map, _ = load_existing_progress(OUTPUT_FILE)
    
    total_generated = 0
    print(f"\nResuming Generation...")
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for v_idx in range(9):
            for a_idx in range(9):
                original_samples = bin_storage.get((v_idx, a_idx), [])
                count_original = len(original_samples)
                count_generated = progress_map.get((v_idx, a_idx), 0)
                current_total = count_original + count_generated
                
                if current_total >= TARGET_COUNT: continue
                
                needed = TARGET_COUNT - current_total
                if needed > 0: print(f"Bin ({v_idx+1},{a_idx+1}) | Need: {needed}")
                
                center_v = v_idx + 1.5
                center_a = a_idx + 1.5
                
                for _ in range(needed):
                    if count_original > 0:
                        res = rewrite_sample(random.choice(original_samples))
                        method = "rewrite"
                    else:
                        res = generate_fresh(center_v, center_a)
                        method = "synth"
                    
                    if res:
                        record = {
                            "ID": f"aug_{method}_{int(time.time()*10000)}",
                            "Text": res["text"],
                            "Quadruplet": [{
                                "Aspect": res["aspect"], "Category": res["category"],
                                "Opinion": res["opinion"], "VA": res["va"]
                            }]
                        }
                        f_out.write(json.dumps(record) + "\n")
                        f_out.flush() 
                        total_generated += 1
                        print(".", end="", flush=True) 
                    time.sleep(0.5) 

    print(f"\nDONE. Added {total_generated} NEW samples.")

if __name__ == "__main__":
    main()