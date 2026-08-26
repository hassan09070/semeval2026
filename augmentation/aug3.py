import json
import random
from pathlib import Path


def randomize_va_decimals(va_string):
    """
    Randomize the decimal part of VA values.
    
    Input: "1.5#2.5" (variance#arousal)
    Output: "1.3#2.7" (with random decimals between 0.2-0.8)
    
    Args:
        va_string (str): VA string in format "variance#arousal"
    
    Returns:
        str: VA string with randomized decimals
    """
    try:
        variance_str, arousal_str = va_string.split('#')
        variance = float(variance_str)
        arousal = float(arousal_str)
        
        # Extract integer parts
        variance_int = int(variance)
        arousal_int = int(arousal)
        
        # Generate random decimals between 0.2 and 0.8
        random_decimal_v = round(random.uniform(0.2, 0.8), 1)
        random_decimal_a = round(random.uniform(0.2, 0.8), 1)
        
        # Combine integer + random decimal
        new_variance = variance_int + random_decimal_v
        new_arousal = arousal_int + random_decimal_a
        
        return f"{new_variance}#{new_arousal}"
    except Exception as e:
        print(f"Error processing VA string '{va_string}': {e}")
        return va_string


def should_keep_record(record):
    """
    Check if record should be kept (not filtered out).
    Remove records where variance > 9 OR arousal > 9
    
    Args:
        record (dict): JSONL record with Quadruplet containing VA
    
    Returns:
        bool: True if record should be kept, False if it should be removed
    """
    try:
        quadruplets = record.get("Quadruplet", [])
        if not quadruplets:
            return True
        
        quad = quadruplets[0]
        va_str = quad.get("VA", "0#0")
        variance_str, arousal_str = va_str.split('#')
        
        variance = float(variance_str)
        arousal = float(arousal_str)
        
        # Remove if variance > 9 OR arousal > 9
        if variance > 9 or arousal > 9:
            return False
        
        return True
    except Exception as e:
        print(f"Error checking record: {e}")
        return True


def clean_and_randomize_va(input_file, output_file):
    """
    Process JSONL file to:
    1. Remove records where variance > 9 or arousal > 9
    2. Randomize decimal part of both VA values (0.2-0.8)
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    
    Returns:
        dict: Statistics about processing
    """
    stats = {
        "total_records": 0,
        "records_kept": 0,
        "records_removed": 0,
        "errors": 0
    }
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue
                
                stats["total_records"] += 1
                
                try:
                    record = json.loads(line)
                    
                    # Step 1: Filter records
                    if not should_keep_record(record):
                        stats["records_removed"] += 1
                        continue
                    
                    # Step 2: Randomize VA decimals
                    quadruplets = record.get("Quadruplet", [])
                    if quadruplets:
                        quad = quadruplets[0]
                        old_va = quad.get("VA", "0#0")
                        new_va = randomize_va_decimals(old_va)
                        quad["VA"] = new_va
                    
                    # Write modified record
                    outfile.write(json.dumps(record) + '\n')
                    stats["records_kept"] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: JSON decode error - {e}")
                    stats["errors"] += 1
                except Exception as e:
                    print(f"Line {line_num}: Error processing record - {e}")
                    stats["errors"] += 1
        
        return stats
    
    except Exception as e:
        print(f"Error reading/writing files: {e}")
        return stats


def process_multiple_files(input_files, output_dir):
    """
    Process multiple JSONL files with the same cleaning/randomization logic.
    
    Args:
        input_files (list): List of input JSONL file paths
        output_dir (str): Directory to save output files (will be created if needed)
    
    Returns:
        dict: Statistics for all files combined
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_stats = {
        "total_records": 0,
        "records_kept": 0,
        "records_removed": 0,
        "errors": 0,
        "files_processed": 0
    }
    
    for input_file in input_files:
        if not Path(input_file).exists():
            print(f"❌ File not found: {input_file}")
            continue
        
        # Generate output filename
        input_name = Path(input_file).stem
        output_file = output_path / f"{input_name}_cleaned.jsonl"
        
        print(f"\n📄 Processing: {input_file}")
        print(f"   Output: {output_file}")
        
        stats = clean_and_randomize_va(input_file, str(output_file))
        
        # Update totals
        total_stats["total_records"] += stats["total_records"]
        total_stats["records_kept"] += stats["records_kept"]
        total_stats["records_removed"] += stats["records_removed"]
        total_stats["errors"] += stats["errors"]
        total_stats["files_processed"] += 1
        
        # Print stats for this file
        print(f"   ✅ Kept: {stats['records_kept']} | ❌ Removed: {stats['records_removed']} | ⚠️  Errors: {stats['errors']}")
    
    return total_stats


# ==================== USAGE EXAMPLES ====================
if __name__ == "__main__":
    
    # Example 1: Process a single file
    print("=" * 80)
    print("CLEANING & RANDOMIZING VA VALUES")
    print("=" * 80)
    
    input_file = "/Users/hassan/Documents/code/office/augmented_eng_gemini.jsonl"
    output_file = "/Users/hassan/Documents/code/office/augmented_eng_gemini_cleaned.jsonl"
    
    if Path(input_file).exists():
        stats = clean_and_randomize_va(input_file, output_file)
        
        print(f"\n📊 STATISTICS")
        print(f"   Total Records: {stats['total_records']}")
        print(f"   Records Kept: {stats['records_kept']}")
        print(f"   Records Removed: {stats['records_removed']}")
        print(f"   Errors: {stats['errors']}")
        print(f"\n✅ Cleaned file saved to: {output_file}")
    else:
        print(f"❌ Input file not found: {input_file}")
    
    
    # Example 2: Process multiple files at once (uncomment to use)
    # input_files = [
    #     "/Users/hassan/Documents/code/office/augmented_eng_gemini.jsonl",
    #     "/Users/hassan/Documents/code/office/augmented_zho_gemini.jsonl"
    # ]
    # output_dir = "/Users/hassan/Documents/code/office/cleaned_data"
    # 
    # total_stats = process_multiple_files(input_files, output_dir)
    # 
    # print(f"\n📊 TOTAL STATISTICS")
    # print(f"   Files Processed: {total_stats['files_processed']}")
    # print(f"   Total Records: {total_stats['total_records']}")
    # print(f"   Total Records Kept: {total_stats['records_kept']}")
    # print(f"   Total Records Removed: {total_stats['records_removed']}")
    # print(f"   Total Errors: {total_stats['errors']}")
