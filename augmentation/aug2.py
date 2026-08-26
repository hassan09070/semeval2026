import json
import time
import requests
import os
import random
import asyncio
from openai import AsyncOpenAI, APIError, RateLimitError
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# Using Groq API (better free tier than Gemini)
API_KEY = os.environ["GROQ_API_KEY"]  # Groq API Key
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME = "llama-3.3-70b-versatile"

# Async client (will be initialized in main)
async_client = None

# 🧪 TEST MODE CONFIGURATION
TEST_MODE = False  # Set to False for production
TEST_LANGS = ["eng"]  # Test only 1 language
TEST_BINS = 2  # Test only 2 VA bins instead of 81
TEST_SAMPLES_PER_BIN = 5  # Test 5 samples per bin instead of 50

# Production parameters - GROQ RATE LIMITS (Optimized for TPM bottleneck)
# RPM = 30 requests/min (safe limit)
# TPM = 12,000 tokens/min (PRIMARY BOTTLENECK)
# Strategy: Batch 10 samples/request (~620 tokens each) with 8 concurrent workers
# Result: 8 workers × 60s ÷ 5.15s per batch ≈ 93 batches/min ≈ 930 requests/min (way over RPM!)
# So we limit to: 10-12 concurrent at 5s inter-batch delay = safe
API_CALL_DELAY = 0.5  # Inter-request delay (seconds) - for concurrency spacing
MAX_CONCURRENT_REQUESTS = 8  # 8 concurrent workers (safe under 12k TPM)
BATCH_SIZE = 10  # Generate 10 samples per request (not 1!)
TARGET_COUNT_PER_BIN_PER_LANG = 50 if not TEST_MODE else TEST_SAMPLES_PER_BIN
BATCH_WAIT_SECONDS = 0.1  # Small delay between batch submissions

# Load configurations
subtask = "subtask_1"
task = "task1"
langs = TEST_LANGS if TEST_MODE else ["eng", "zho", "jpn", "rus", "tat", "ukr"]
domains = ["restaurant", "laptop", "finance", "hotel"]

all_train = []
lang_data = {lang: [] for lang in langs}  # Separate storage per language

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
        """Record a successfully generated sample."""
        self.progress[lang]["samples_generated"] += 1
    
    def record_bin_complete(self, lang):
        """Record when a bin reaches target."""
        self.progress[lang]["bins_completed"] += 1
    
    def record_api_call(self, lang):
        """Track API calls per language."""
        self.progress[lang]["api_calls"] += 1
    
    def record_error(self, lang):
        """Track errors per language."""
        self.progress[lang]["errors"] += 1
    
    def print_live_status(self, lang):
        """Print real-time status for current language."""
        p = self.progress[lang]
        elapsed = time.time() - p["start_time"] if p["start_time"] else 0
        generated = p["samples_generated"]
        target = p["samples_target"]
        percent = (generated / target * 100) if target > 0 else 0
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * generated / target) if target > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        
        rate = generated / elapsed if elapsed > 0 else 0
        eta_seconds = (target - generated) / rate if rate > 0 else 0
        eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if rate > 0 else "?"
        
        print(f"\n  [{bar}] {percent:.1f}% ({generated}/{target})")
        print(f"  ⏱️  Elapsed: {int(elapsed)}s | ETA: {eta_str} | Rate: {rate:.1f} samples/sec")
        print(f"  📊 Bins: {p['bins_completed']}/{p['bins_total']} | API Calls: {p['api_calls']} | Errors: {p['errors']}")
    
    def print_summary(self):
        """Print final summary across all languages."""
        print(f"\n\n{'='*80}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*80}\n")
        
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
            
            print(f"{lang.upper():6} | {p['samples_generated']:4}/{p['samples_target']:4} samples ({percent:5.1f}%) " +
                  f"| {elapsed:6.0f}s | {p['api_calls']:3} API calls | {p['errors']:2} errors")
        
        print(f"\n{'─'*80}")
        print(f"TOTAL  | {total_samples:4}/{self.total_expected:4} samples " +
              f"| {total_elapsed:6.0f}s | {total_api_calls:3} API calls | {total_errors:2} errors")
        print(f"{'='*80}\n")
        
        return total_samples, total_api_calls, total_errors, total_elapsed

tracker = ProgressTracker(langs, TARGET_COUNT_PER_BIN_PER_LANG)

# ==========================================
# VALIDATION HELPER
# ==========================================
def is_valid_record(record):
    """Check if a record has non-empty text, aspect, and opinion."""
    try:
        text = record.get("Text", "").strip()
        quadruplet = record.get("Quadruplet", [{}])[0]
        aspect = quadruplet.get("Aspect", "").strip()
        opinion = quadruplet.get("Opinion", "").strip()
        
        return bool(text and aspect and opinion)
    except:
        return False

# ==========================================
# DATA LOADING (MULTI-LANGUAGE)
# ==========================================
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

def load_existing_augmented_data(output_file):
    """Load already generated augmented samples from file (for resuming)."""
    existing_by_lang = {lang: [] for lang in langs}
    existing_by_lang_bin = {lang: {} for lang in langs}
    
    if not os.path.exists(output_file):
        print(f"  ℹ️ No existing file found: {output_file} (starting fresh)")
        return existing_by_lang, existing_by_lang_bin, 0
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            total_loaded = 0
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    
                    # SKIP EMPTY RECORDS
                    if not is_valid_record(record):
                        continue
                    
                    lang = record.get("Language", "unknown")
                    
                    if lang not in existing_by_lang:
                        continue
                    
                    # Track by language
                    existing_by_lang[lang].append(record)
                    
                    # Track by bin (1-9 scale)
                    quad = record.get("Quadruplet", [{}])[0]
                    va_str = quad.get("VA", "0#0")
                    try:
                        v, a = map(float, va_str.split('#'))
                        v_idx = max(0, min(8, int(v - 1)))
                        a_idx = max(0, min(8, int(a - 1)))
                        
                        if (v_idx, a_idx) not in existing_by_lang_bin[lang]:
                            existing_by_lang_bin[lang][(v_idx, a_idx)] = 0
                        existing_by_lang_bin[lang][(v_idx, a_idx)] += 1
                    except:
                        pass
                    
                    total_loaded += 1
                except json.JSONDecodeError:
                    pass
        
        print(f"  ✅ Loaded {total_loaded} existing augmented samples from {output_file}")
        return existing_by_lang, existing_by_lang_bin, total_loaded
    
    except Exception as e:
        print(f"  ⚠️ Error loading existing file: {e}")
        return existing_by_lang, existing_by_lang_bin, 0

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

# ==========================================
# VA INTERPRETATION & EMOTION GROUNDING
# ==========================================
def get_emotion_interpretation(v, a):
    """Map numeric VA values to emotional keywords."""
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
# ASYNC GENERATION LOGIC (CONCURRENT API CALLS)
# ==========================================
async def generate_with_backoff_async(prompt, semaphore, retries=5, lang="eng", max_tokens=600):
    """Call Groq API asynchronously with semaphore for concurrency control."""
    async with semaphore:
        wait_time = 2
        for attempt in range(retries):
            try:
                response = await async_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful data augmentation assistant for ABSA. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    timeout=30
                )
                tracker.record_api_call(lang)
                return json.loads(response.choices[0].message.content)
            except RateLimitError:
                if attempt < retries - 1:
                    await asyncio.sleep(wait_time)
                    wait_time *= 2
            except Exception as e:
                tracker.record_error(lang)
                if attempt < retries - 1:
                    await asyncio.sleep(1)
        return None

def get_bin_data_by_lang(lang_samples):
    """Bin language-specific samples by VA region (1-9 scale)."""
    bin_storage = {}
    for obj in lang_samples:
        for q_idx, quad in enumerate(obj.get("Quadruplet", [])):
            try:
                v, a = map(float, quad["VA"].split('#'))
                # VA values are 1-9, map to bin indices 0-8
                v_idx = max(0, min(8, int(v - 1)))
                a_idx = max(0, min(8, int(a - 1)))
                if (v_idx, a_idx) not in bin_storage:
                    bin_storage[(v_idx, a_idx)] = []
                bin_storage[(v_idx, a_idx)].append((obj, q_idx))
            except:
                continue
    return bin_storage

async def generate_fresh_quadruplet(target_v, target_a, lang_code, semaphore, num_samples=10):
    """Generate synthetic ABSA samples targeting VA bin (batched, async)."""
    v_emotion, a_emotion = get_emotion_interpretation(target_v, target_a)
    
    # Domain selection
    domain = "laptop"
    if lang_code == "zho": domain = "restaurant"
    elif lang_code in ["jpn", "rus"]: domain = "hotel"
    
    lang_map = {"eng": "English", "zho": "Chinese", "jpn": "Japanese", "rus": "Russian", "tat": "Tatar", "ukr": "Ukrainian"}
    lang_name = lang_map.get(lang_code, "English")
    
    # OPTIMIZED PROMPT (~300 tokens, high quality)
    prompt = f"""Generate {num_samples} natural {lang_name} review sentences for {domain}.
TARGET: V={target_v} ({v_emotion}), A={target_a} ({a_emotion})

Rules:
- Aspect: explicit, {domain}-relevant
- Opinion: explicit 1-3 word phrase
- Text: 15-30 words, natural & realistic (NOT terse)
- Categories: {domain.upper()}#GENERAL, {domain.upper()}#DESIGN_FEATURES, {domain.upper()}#PERFORMANCE, {domain.upper()}#PRICE, {domain.upper()}#USABILITY

JSON:
{{"samples": [{{"text": "...", "aspect": "...", "opinion": "...", "category": "...", "va": "{target_v}#{target_a}"}}]}}"""
    
    result = await generate_with_backoff_async(prompt, semaphore, retries=5, lang=lang_code, max_tokens=600)
    if result and "samples" in result:
        return result["samples"]
    return []

async def rewrite_sample_with_target_va(seed_obj, target_q_idx, target_v, target_a, lang_code, semaphore):
    """Rewrite existing sample to match target VA, preserving aspect/opinion (async)."""
    original_text = seed_obj["Text"]
    quad = seed_obj["Quadruplet"][target_q_idx]
    aspect = quad["Aspect"]
    opinion = quad["Opinion"]
    category = quad["Category"]
    
    v_emotion, a_emotion = get_emotion_interpretation(target_v, target_a)
    
    lang_map = {"eng": "English", "zho": "Chinese", "jpn": "Japanese", "rus": "Russian", "tat": "Tatar", "ukr": "Ukrainian"}
    lang_name = lang_map.get(lang_code, "English")
    
    # OPTIMIZED REWRITE PROMPT (~250 tokens)
    prompt = f"""Rewrite in {lang_name}, preserving aspect & opinion:
Aspect: "{aspect}"
Opinion: "{opinion}"
Target: V={target_v} ({v_emotion}), A={target_a} ({a_emotion})
Original: "{original_text}"

Keep aspect & opinion. Match emotional tone. Make it 15-30 words, natural (NOT terse).

JSON: {{"rewritten_text": "...", "aspect": "{aspect}", "opinion": "{opinion}", "category": "{category}", "va": "{target_v}#{target_a}"}}"""
    
    result = await generate_with_backoff_async(prompt, semaphore, retries=5, lang=lang_code, max_tokens=150)
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
# MAIN EXECUTION (ASYNC)
# ==========================================
async def main():
    global async_client
    
    # Initialize async client
    async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # Initialize tracking
    tracker.start()
    
    # Print configuration
    print(f"\n{'='*80}")
    if TEST_MODE:
        print("🧪 TEST MODE ACTIVE")
        print(f"   Languages: {TEST_LANGS}")
        print(f"   Bins to test: {TEST_BINS}/81")
        print(f"   Samples per bin: {TEST_SAMPLES_PER_BIN}/50")
        print(f"   Rate strategy: Concurrent (TPM-optimized)")
        print(f"   Workers: {MAX_CONCURRENT_REQUESTS} | Batch size: {BATCH_SIZE}")
    else:
        print("🚀 PRODUCTION MODE (TPM-Optimized Concurrent)")
        print(f"   Groq limits: RPM=30, TPM=12,000 (TPM is bottleneck)")
        print(f"   Strategy: {MAX_CONCURRENT_REQUESTS} workers × {BATCH_SIZE} samples/batch")
        print(f"   Per-request tokens: ~600 (under 12k TPM limit)")
        print(f"   Estimated throughput: ~1,600 samples/day")
    print(f"{'='*80}\n")
    
    # Load all languages
    print(f"\n{'='*80}")
    print("📥 Loading DimABSA 2026 Data")
    print(f"{'='*80}\n")
    load_all_languages()
    
    # Load existing augmented data (resume capability) - will check per-language files
    print(f"\n{'='*80}")
    print("📂 Checking for existing augmented data")
    print(f"{'='*80}\n")
    existing_by_lang, existing_by_lang_bin = {lang: [] for lang in langs}, {lang: {} for lang in langs}
    total_existing = 0
    
    for lang in langs:
        lang_output_file = os.path.join(os.path.dirname(__file__), 
                                        f"augmented_{lang}_gemini_test.jsonl" if TEST_MODE else f"augmented_{lang}_gemini.jsonl")
        lang_existing, lang_existing_bin, lang_count = load_existing_augmented_data(lang_output_file)
        existing_by_lang[lang] = lang_existing[lang]
        existing_by_lang_bin[lang] = lang_existing_bin[lang]
        total_existing += lang_count
    
    print(f"\n{'='*80}")
    print("🎯 Multi-Language VA Bin-Count-Fill Augmentation (Sequential, TPM-Aware)")
    print(f"{'='*80}")
    if not TEST_MODE:
        total_expected = len(langs) * 81 * TARGET_COUNT_PER_BIN_PER_LANG
        still_needed = total_expected - total_existing
        print(f"📊 Total expected: {total_expected} samples")
        print(f"✅ Already done: {total_existing} samples")
        print(f"📋 Still needed: {still_needed} samples\n")
    else:
        print(f"🧪 Test mode\n")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Generate per language
    total_generated = 0
    
    for lang_idx, lang in enumerate(langs, 1):
        # Output file for this language
        output_file = os.path.join(os.path.dirname(__file__), 
                                  f"augmented_{lang}_gemini_test.jsonl" if TEST_MODE else f"augmented_{lang}_gemini.jsonl")
        f_out = open(output_file, "a", encoding="utf-8")  # Append mode - will resume
        print(f"\n{'='*80}")
        print(f"[{lang_idx}/{len(langs)}] Processing Language: {lang.upper()}")
        print(f"{'='*80}")
        
        lang_samples = lang_data[lang]
        if not lang_samples:
            print(f"⚠️ No data for {lang}, skipping...")
            continue
        
        print(f"📊 Total original samples in {lang.upper()}: {len(lang_samples)}")
        
        bin_storage = get_bin_data_by_lang(lang_samples)
        print(f"📦 Bins with original data: {len(bin_storage)}/81")
        
        # Check existing augmented samples for this language
        existing_count = len(existing_by_lang[lang])
        print(f"✅ Existing augmented samples: {existing_count}")
        print(f"📋 Existing bins covered: {len(existing_by_lang_bin[lang])}/81\n")
        
        bin_processed = 0
        bins_tested = 0
        total_successful = 0
        total_failed = 0
        
        for v_idx in range(9):
            if TEST_MODE and bins_tested >= TEST_BINS:
                print(f"\n🧪 Test limit reached ({TEST_BINS} bins tested)")
                break
                
            for a_idx in range(9):
                if TEST_MODE and bins_tested >= TEST_BINS:
                    break
                    
                bin_processed += 1
                original_samples = bin_storage.get((v_idx, a_idx), [])
                count_original = len(original_samples)
                
                # Add existing augmented samples to the count
                count_existing_aug = existing_by_lang_bin[lang].get((v_idx, a_idx), 0)
                count_total = count_original + count_existing_aug
                
                if count_total >= TARGET_COUNT_PER_BIN_PER_LANG:
                    continue
                
                needed = TARGET_COUNT_PER_BIN_PER_LANG - count_total
                
                if needed > 0:
                    bin_label = f"({v_idx+1},{a_idx+1})"
                    print(f"\n  🔄 Bin {bin_label:8} | Original: {count_original:2}, Existing: {count_existing_aug:2}, Need: {needed:2} ", end="", flush=True)
                    bins_tested += 1
                
                center_v = v_idx + 1.5
                center_a = a_idx + 1.5
                
                # Batch generation: generate in groups of BATCH_SIZE (e.g., 10)
                successful_this_bin = 0
                remaining = needed
                
                while remaining > 0:
                    batch_size = min(BATCH_SIZE, remaining)
                    actual_generated = 0  # Track actual count
                    
                    # Decide: 60% rewrite, 40% fresh
                    if count_original > 0 and random.random() < 0.6:
                        # Batch rewrite: generate BATCH_SIZE rewrites concurrently
                        tasks = []
                        for _ in range(batch_size):
                            seed_obj, q_idx = random.choice(original_samples)
                            tasks.append(rewrite_sample_with_target_va(
                                seed_obj, q_idx, center_v, center_a, lang, semaphore
                            ))
                        
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for res in results:
                            if isinstance(res, dict) and res:
                                record = {
                                    "ID": f"aug_rewrite_groq_{lang}_{int(time.time()*10000)}_{random.randint(1000, 9999)}",
                                    "Text": res["text"],
                                    "Quadruplet": [{
                                        "Aspect": res["aspect"],
                                        "Category": res["category"],
                                        "Opinion": res["opinion"],
                                        "VA": res["va"]
                                    }],
                                    "Language": lang
                                }
                                # SKIP EMPTY RECORDS
                                if is_valid_record(record):
                                    f_out.write(json.dumps(record) + "\n")
                                    f_out.flush()
                                    tracker.record_sample(lang, (v_idx, a_idx))
                                    successful_this_bin += 1
                                    total_successful += 1
                                    total_generated += 1
                                    actual_generated += 1
                                    print("✓", end="", flush=True)
                                else:
                                    total_failed += 1
                                    print("✗", end="", flush=True)
                            else:
                                total_failed += 1
                                print("✗", end="", flush=True)
                    else:
                        # Batch fresh: 1 request generates BATCH_SIZE samples
                        samples = await generate_fresh_quadruplet(center_v, center_a, lang, semaphore, num_samples=batch_size)
                        actual_generated = len(samples)
                        
                        for sample in samples[:batch_size]:  # Ensure we don't exceed batch_size
                            res = {
                                "text": sample["text"],
                                "aspect": sample["aspect"],
                                "opinion": sample["opinion"],
                                "category": sample["category"],
                                "va": sample["va"],
                                "lang": lang
                            }
                            record = {
                                "ID": f"aug_synth_groq_{lang}_{int(time.time()*10000)}_{random.randint(1000, 9999)}",
                                "Text": res["text"],
                                "Quadruplet": [{
                                    "Aspect": res["aspect"],
                                    "Category": res["category"],
                                    "Opinion": res["opinion"],
                                    "VA": res["va"]
                                }],
                                "Language": lang
                            }
                            # SKIP EMPTY RECORDS
                            if is_valid_record(record):
                                f_out.write(json.dumps(record) + "\n")
                                f_out.flush()
                                tracker.record_sample(lang, (v_idx, a_idx))
                                successful_this_bin += 1
                                total_successful += 1
                                total_generated += 1
                                print("✓", end="", flush=True)
                        
                        # Count failures for samples we didn't get
                        if actual_generated < batch_size:
                            total_failed += (batch_size - actual_generated)
                            for _ in range(batch_size - actual_generated):
                                print("✗", end="", flush=True)
                    
                    # Decrement by ACTUAL generated, not requested batch size
                    # This ensures we retry if we didn't get enough
                    remaining -= actual_generated
                    await asyncio.sleep(API_CALL_DELAY)  # Spacing between batches
                
                if needed > 0:
                    tracker.record_bin_complete(lang)
                    print(f" {successful_this_bin}/{needed}")
        
        # Print progress after each language
        print(f"\n")
        tracker.print_live_status(lang)
        
        print(f"\n  ✅ Batch execution complete for {lang.upper()}", flush=True)
        
        # Close file for this language
        try:
            if not f_out.closed:
                f_out.close()
        except Exception:
            pass
        
        print(f"✅ Output saved to: {output_file}")
    
    # Print final summary
    total_samples, total_api_calls, total_errors, total_elapsed = tracker.print_summary()
    
    print(f"📈 Avg time per sample: {total_elapsed/total_samples:.2f}s" if total_samples > 0 else "")
    print(f"🔗 API Efficiency: {total_samples/total_api_calls:.2f} samples per API call" if total_api_calls > 0 else "")
    
    if TEST_MODE:
        print(f"\n🧪 Test complete! If everything looks good, set TEST_MODE = False for production.")
    
    print(f"\n📁 All language-specific files saved in workspace directory")

    await async_client.close()
    
    print(f"\n")

if __name__ == "__main__":
    asyncio.run(main())