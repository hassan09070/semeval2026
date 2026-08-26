#!/usr/bin/env python3
"""
Kaggle-Optimized ABSA Data Augmentation Pipeline
Generates synthetic aspect-based sentiment analysis samples using Groq API
Designed for long-running container execution with checkpoint recovery
"""

import json
import time
import requests
import os
import random
import sys
from openai import OpenAI, APIError, RateLimitError
from datetime import datetime
import logging

# ==========================================
# LOGGING SETUP
# ==========================================
LOG_DIR = "/app/logs" if os.path.exists("/app") else "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, f"augmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("🚀 KAGGLE ABSA AUGMENTATION PIPELINE STARTED")
logger.info("="*80)

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = os.environ["GROQ_API_KEY"]
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-70b-versatile"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==========================================
# PRODUCTION PARAMETERS (Optimized for Kaggle)
# ==========================================
API_CALL_DELAY = 2.0
BATCH_SIZE = 5  # Generate 5 samples per API call for efficiency
TARGET_COUNT_PER_BIN_PER_LANG = 50

# Data configuration
subtask = "subtask_1"
task = "task1"
langs = ["eng", "zho", "jpn", "rus", "tat", "ukr"]
domains = ["restaurant", "laptop", "finance", "hotel"]

# Output directory
OUTPUT_DIR = "/app/output" if os.path.exists("/app") else "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, "augmented_multilang_kaggle.jsonl")
checkpoint_file = os.path.join(OUTPUT_DIR, "augmentation_checkpoint.json")

all_train = []
lang_data = {lang: [] for lang in langs}

logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Output file: {output_file}")
logger.info(f"Checkpoint file: {checkpoint_file}")

# ==========================================
# CHECKPOINT MANAGEMENT
# ==========================================
def load_checkpoint():
    """Load augmentation checkpoint to resume from last state."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"✅ Loaded checkpoint: {checkpoint['timestamp']}")
            logger.info(f"   Samples generated so far: {checkpoint['total_samples']}")
            return checkpoint
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")
            return None
    return None

def save_checkpoint(state):
    """Save current augmentation state for recovery."""
    state['timestamp'] = datetime.now().isoformat()
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        logger.debug(f"💾 Checkpoint saved: {state['total_samples']} samples")
    except Exception as e:
        logger.error(f"❌ Failed to save checkpoint: {e}")

# ==========================================
# PROGRESS TRACKING
# ==========================================
class ProgressTracker:
    def __init__(self, langs, target_per_bin=50):
        self.langs = langs
        self.target_per_bin = target_per_bin
        self.total_bins = 9 * 9  # 81 VA bins
        self.total_expected = len(langs) * self.total_bins * target_per_bin
        
        self.progress = {
            lang: {
                "samples_generated": 0,
                "samples_target": self.total_bins * target_per_bin,
                "bins_completed": 0,
                "bins_total": self.total_bins,
                "start_time": None,
                "api_calls": 0,
                "errors": 0
            }
            for lang in langs
        }
        self.global_start = None
    
    def start(self):
        self.global_start = time.time()
        for lang in self.langs:
            self.progress[lang]["start_time"] = time.time()
    
    def record_sample(self, lang, bin_coord):
        self.progress[lang]["samples_generated"] += 1
    
    def record_bin_complete(self, lang):
        self.progress[lang]["bins_completed"] += 1
    
    def record_api_call(self, lang):
        self.progress[lang]["api_calls"] += 1
    
    def record_error(self, lang):
        self.progress[lang]["errors"] += 1
    
    def print_live_status(self, lang):
        p = self.progress[lang]
        elapsed = time.time() - p["start_time"] if p["start_time"] else 0
        generated = p["samples_generated"]
        target = p["samples_target"]
        percent = (generated / target * 100) if target > 0 else 0
        
        bar_length = 40
        filled = int(bar_length * generated / target) if target > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        rate = generated / elapsed if elapsed > 0 else 0
        eta_seconds = (target - generated) / rate if rate > 0 else 0
        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if rate > 0 else "?"
        
        status = f"[{bar}] {percent:.1f}% ({generated}/{target}) | ⏱️ {int(elapsed)}s | ETA: {eta_str} | Rate: {rate:.1f}/s"
        logger.info(f"  {lang.upper()}: {status}")
    
    def print_summary(self):
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL SUMMARY")
        logger.info("="*80)
        
        total_samples = 0
        total_api_calls = 0
        total_errors = 0
        total_elapsed = time.time() - self.global_start if self.global_start else 0
        
        for lang in self.langs:
            p = self.progress[lang]
            total_samples += p["samples_generated"]
            total_api_calls += p["api_calls"]
            total_errors += p["errors"]
            
            elapsed = time.time() - p["start_time"] if p["start_time"] else 0
            percent = (p["samples_generated"] / p["samples_target"] * 100) if p["samples_target"] > 0 else 0
            
            logger.info(f"{lang.upper():6} | {p['samples_generated']:4}/{p['samples_target']:4} ({percent:5.1f}%) | "
                       f"{elapsed:6.0f}s | API: {p['api_calls']:3} | Errors: {p['errors']:2}")
        
        logger.info(f"{'─'*80}")
        logger.info(f"TOTAL  | {total_samples:4}/{self.total_expected:4} | {total_elapsed:6.0f}s | API: {total_api_calls:3} | Errors: {total_errors:2}")
        logger.info("="*80)
        
        return total_samples, total_api_calls, total_errors, total_elapsed

tracker = ProgressTracker(langs, TARGET_COUNT_PER_BIN_PER_LANG)

# ==========================================
# DATA LOADING
# ==========================================
def load_jsonl_url(url):
    """Load JSONL data from URL with retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                continue
            lines = response.text.strip().split('\n')
            return [json.loads(line) for line in lines if line.strip()]
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Retry {attempt+1}/{max_retries} for {url.split('/')[-1]}: {str(e)[:50]}")
                time.sleep(2)
            else:
                logger.error(f"❌ Failed to load {url}: {e}")
    return []

def load_all_languages():
    """Load DimABSA 2026 data for all languages."""
    logger.info("📥 Loading DimABSA 2026 Data (All Languages)")
    logger.info("-" * 80)
    
    for lang in langs:
        lang_samples = []
        for domain in domains:
            specified_task = "task1" if domain == "finance" else "alltasks"
            train_url = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/{subtask}/{lang}/{lang}_{domain}_train_{specified_task}.jsonl"
            
            train_raw = load_jsonl_url(train_url)
            if train_raw:
                lang_samples.extend(train_raw)
                all_train.extend(train_raw)
                logger.info(f"✅ {lang}-{domain}: {len(train_raw)} samples")
        
        lang_data[lang] = lang_samples
        logger.info(f"📊 {lang.upper()}: {len(lang_samples)} total samples")
    
    logger.info("-" * 80)

# ==========================================
# VA INTERPRETATION
# ==========================================
def get_emotion_interpretation(v, a):
    """Map VA values to emotional keywords."""
    v, a = float(v), float(a)
    
    if v >= 7: v_emotion = "pleasant, satisfied, delighted, impressed"
    elif v >= 5: v_emotion = "moderately positive, content"
    elif v >= 3: v_emotion = "neutral, indifferent"
    else: v_emotion = "disappointed, frustrated, unhappy"
    
    if a >= 7: a_emotion = "excited, energetic, tense, intense"
    elif a >= 5: a_emotion = "moderately engaged, focused"
    elif a >= 3: a_emotion = "neutral, calm"
    else: a_emotion = "calm, relaxed, dull, passive"
    
    return v_emotion, a_emotion

# ==========================================
# API CALLS WITH BACKOFF
# ==========================================
def generate_with_backoff(prompt, retries=5, lang="eng"):
    """Call Groq API with exponential backoff."""
    wait_time = 2
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful data augmentation assistant for ABSA. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                response_format={"type": "json_object"},
                timeout=30
            )
            tracker.record_api_call(lang)
            return json.loads(response.choices[0].message.content)
        except RateLimitError:
            if attempt < retries - 1:
                logger.warning(f"⚠️ [RateLimit] {lang.upper()} - Attempt {attempt+1}/{retries}. Sleeping {wait_time}s...")
                time.sleep(wait_time)
                wait_time *= 2
        except Exception as e:
            logger.error(f"❌ [Error] {lang.upper()} - {type(e).__name__}: {str(e)[:60]}")
            tracker.record_error(lang)
            if attempt < retries - 1:
                time.sleep(1)
    
    return None

def get_bin_data_by_lang(lang_samples):
    """Bin language samples by VA region."""
    bin_storage = {}
    for obj in lang_samples:
        for q_idx, quad in enumerate(obj.get("Quadruplet", [])):
            try:
                v, a = map(float, quad["VA"].split('#'))
                v_idx = min(8, int(v - 1)) if v < 10 else 8
                a_idx = min(8, int(a - 1)) if a < 10 else 8
                if (v_idx, a_idx) not in bin_storage:
                    bin_storage[(v_idx, a_idx)] = []
                bin_storage[(v_idx, a_idx)].append((obj, q_idx))
            except:
                continue
    return bin_storage

def generate_fresh_quadruplet(target_v, target_a, lang_code, num_samples=1):
    """Generate synthetic ABSA samples targeting VA bin."""
    v_emotion, a_emotion = get_emotion_interpretation(target_v, target_a)
    
    domain = "laptop"
    if lang_code == "zho": domain = "restaurant"
    elif lang_code in ["jpn", "rus"]: domain = "hotel"
    
    lang_map = {"eng": "English", "zho": "Chinese", "jpn": "Japanese", "rus": "Russian", "tat": "Tatar", "ukr": "Ukrainian"}
    lang_name = lang_map.get(lang_code, "English")
    
    prompt = f"""You are a data generation assistant for Aspect-Based Sentiment Analysis with Valence–Arousal regression.

Your task is to generate {num_samples} synthetic but realistic review sentences in {lang_name} for the {domain} domain.

TARGET VA BIN:
- Valence (V): {target_v} → {v_emotion}
- Arousal (A): {target_a} → {a_emotion}

OUTPUT REQUIREMENTS:
Generate exactly {num_samples} unique samples. Each must:
- Clearly reflect the target emotional state
- Be natural and domain-consistent
- Match VA target within ±0.3

OUTPUT FORMAT (strict JSON):
{{
  "samples": [
    {{
      "text": "<natural review sentence in {lang_name}>",
      "aspect": "<explicit aspect>",
      "category": "{domain.upper()}#GENERAL",
      "opinion": "<explicit opinion>",
      "va": "{target_v}#{target_a}"
    }}
  ]
}}

CONSTRAINTS:
- Do NOT mention valence/arousal numbers
- Do NOT generate contradictory emotions
- Do NOT reuse sentences"""
    
    result = generate_with_backoff(prompt, retries=5, lang=lang_code)
    if result and "samples" in result:
        return result["samples"]
    return []

def rewrite_sample_with_target_va(seed_obj, target_q_idx, target_v, target_a, lang_code):
    """Rewrite existing sample to match target VA."""
    original_text = seed_obj["Text"]
    quad = seed_obj["Quadruplet"][target_q_idx]
    aspect = quad["Aspect"]
    opinion = quad["Opinion"]
    category = quad["Category"]
    
    v_emotion, a_emotion = get_emotion_interpretation(target_v, target_a)
    
    lang_map = {"eng": "English", "zho": "Chinese", "jpn": "Japanese", "rus": "Russian", "tat": "Tatar", "ukr": "Ukrainian"}
    lang_name = lang_map.get(lang_code, "English")
    
    prompt = f"""You are an ABSA data augmentation assistant.

TASK: Rewrite a review sentence while preserving aspect and opinion.

CONSTRAINTS:
1. The aspect "{aspect}" must appear exactly.
2. The opinion "{opinion}" must appear exactly.
3. Emotional tone: {v_emotion} (V{target_v}), {a_emotion} (A{target_a})
4. Keep it natural.

Original: "{original_text}"

OUTPUT FORMAT:
{{
  "rewritten_text": "<rewritten sentence>",
  "aspect": "{aspect}",
  "opinion": "{opinion}",
  "category": "{category}",
  "va": "{target_v}#{target_a}"
}}"""
    
    result = generate_with_backoff(prompt, retries=5, lang=lang_code)
    if result and "rewritten_text" in result:
        new_text = result["rewritten_text"]
        if aspect.lower() in new_text.lower() and opinion.lower() in new_text.lower():
            return {
                "text": new_text,
                "aspect": result.get("aspect", aspect),
                "opinion": result.get("opinion", opinion),
                "category": result.get("category", category),
                "va": result.get("va", f"{target_v}#{target_a}"),
                "lang": lang_code
            }
    return None

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    try:
        tracker.start()
        
        logger.info(f"\n{'='*80}")
        logger.info("🚀 KAGGLE PRODUCTION MODE")
        logger.info(f"   API Delay: {API_CALL_DELAY}s | Batch Size: {BATCH_SIZE} | Target/Bin: {TARGET_COUNT_PER_BIN_PER_LANG}")
        logger.info(f"   Total Expected: {len(langs)} langs × 81 bins × {TARGET_COUNT_PER_BIN_PER_LANG} = {len(langs)*81*TARGET_COUNT_PER_BIN_PER_LANG} samples")
        logger.info(f"{'='*80}\n")
        
        # Load data
        load_all_languages()
        
        logger.info(f"\n{'='*80}")
        logger.info("🎯 Starting Augmentation Pipeline")
        logger.info(f"{'='*80}\n")
        
        # Open output file for continuous writing
        f_out = open(output_file, "a", encoding="utf-8")
        total_samples = 0
        
        for lang_idx, lang in enumerate(langs, 1):
            logger.info(f"\n[{lang_idx}/{len(langs)}] Processing Language: {lang.upper()}")
            logger.info("-" * 80)
            
            lang_samples = lang_data[lang]
            if not lang_samples:
                logger.warning(f"⚠️ No data for {lang}, skipping...")
                continue
            
            logger.info(f"📊 Total samples in {lang.upper()}: {len(lang_samples)}")
            
            bin_storage = get_bin_data_by_lang(lang_samples)
            logger.info(f"📦 Bins with data: {len(bin_storage)}/81\n")
            
            for v_idx in range(9):
                for a_idx in range(9):
                    original_samples = bin_storage.get((v_idx, a_idx), [])
                    count_original = len(original_samples)
                    
                    if count_original >= TARGET_COUNT_PER_BIN_PER_LANG:
                        continue
                    
                    needed = TARGET_COUNT_PER_BIN_PER_LANG - count_original
                    
                    if needed > 0:
                        bin_label = f"({v_idx+1},{a_idx+1})"
                        logger.debug(f"  Bin {bin_label}: Original {count_original}, Need {needed}")
                    
                    center_v = v_idx + 1.5
                    center_a = a_idx + 1.5
                    
                    # Batch generation
                    batch_size = min(needed, BATCH_SIZE)
                    generated_batch = generate_fresh_quadruplet(center_v, center_a, lang, num_samples=batch_size)
                    
                    batch_idx = 0
                    for _ in range(needed):
                        res = None
                        
                        # 60% rewrite, 40% fresh
                        if count_original > 0 and random.random() < 0.6:
                            seed_tuple = random.choice(original_samples)
                            res = rewrite_sample_with_target_va(
                                seed_tuple[0], seed_tuple[1], center_v, center_a, lang
                            )
                            method = "rewrite"
                        else:
                            if batch_idx < len(generated_batch):
                                sample = generated_batch[batch_idx]
                                res = {
                                    "text": sample["text"],
                                    "aspect": sample["aspect"],
                                    "opinion": sample["opinion"],
                                    "category": sample["category"],
                                    "va": sample["va"],
                                    "lang": lang
                                }
                                batch_idx += 1
                                method = "synth"
                            else:
                                fresh = generate_fresh_quadruplet(center_v, center_a, lang, num_samples=1)
                                if fresh:
                                    sample = fresh[0]
                                    res = {
                                        "text": sample["text"],
                                        "aspect": sample["aspect"],
                                        "opinion": sample["opinion"],
                                        "category": sample["category"],
                                        "va": sample["va"],
                                        "lang": lang
                                    }
                                    method = "synth"
                        
                        if res:
                            record = {
                                "ID": f"aug_{method}_groq_{lang}_{int(time.time()*10000)}_{random.randint(1000,9999)}",
                                "Text": res["text"],
                                "Quadruplet": [{
                                    "Aspect": res["aspect"],
                                    "Category": res["category"],
                                    "Opinion": res["opinion"],
                                    "VA": res["va"]
                                }],
                                "Language": lang
                            }
                            f_out.write(json.dumps(record) + "\n")
                            f_out.flush()
                            total_samples += 1
                            tracker.record_sample(lang, (v_idx, a_idx))
                        
                        time.sleep(API_CALL_DELAY)
                    
                    if needed > 0:
                        tracker.record_bin_complete(lang)
                        time.sleep(30)  # Inter-bin pause
            
            tracker.print_live_status(lang)
        
        f_out.close()
        
        # Print summary
        total, api_calls, errors, elapsed = tracker.print_summary()
        
        logger.info(f"\n✅ Augmentation Complete!")
        logger.info(f"   Output: {output_file}")
        logger.info(f"   Samples: {total} | API Calls: {api_calls} | Errors: {errors}")
        logger.info(f"   Time: {elapsed/3600:.1f} hours | Avg: {elapsed/total:.2f}s per sample")
        
        # Save final checkpoint
        save_checkpoint({
            "total_samples": total,
            "api_calls": api_calls,
            "errors": errors,
            "elapsed_seconds": elapsed,
            "status": "COMPLETED"
        })
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Augmentation interrupted by user")
        save_checkpoint({
            "total_samples": tracker.progress[langs[0]]["samples_generated"],
            "status": "INTERRUPTED"
        })
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        save_checkpoint({"status": "ERROR", "error": str(e)})
        sys.exit(1)

if __name__ == "__main__":
    main()
