import json
import os

key_name = {1: "Aspect_VA", 2: "Triplet", 3: 'Quadruplet'}

def read_jsonl_file(file_path, task=3, data_type='pred'):
    """
    Reads a JSONL file from the specified path and processes each line.

    Args:
        file_path (str): The path to the JSONL file.
        type (str): pred or gold.

    Returns:
        list: A list of dictionaries containing all successfully parsed lines. 
              Returns an empty list if the file does not exist or cannot be read.
    """
    output_key = key_name[task]
    input_key = key_name[3] if (data_type == 'gold' and task == 2) else key_name[task]
    
    data = []
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return data  # Return empty list on failure instead of exiting

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    # Parse JSON line
                    json_data = json.loads(line)

                    # Extract basic fields (ID, Text), which are usually required
                    entry = {
                        'ID': json_data.get('ID', f"Missing_ID_Line{line_num}"),  # Use line number if ID is missing
                        'Text': json_data.get('Text', ''),
                        'Aspect': json_data.get('Aspect', []),
                    }
                    if entry['ID'] == f"Missing_ID_Line{line_num}":
                        print(f"Error: ID value is missing at line {line_num}!")
                        continue
                    # Handle Quadruplet field (might not exist or be an empty list)
                    quadruplets = json_data.get(input_key, [])  # Default to empty list
                    if data_type == 'gold' and len(quadruplets) == 0:
                        quadruplets = json_data.get(output_key, [])
                    
                    if isinstance(quadruplets, list):
                        # Process each quadruplet
                        parsed_quadruplets = []
                        for quad in quadruplets:
                            # Ensure quad is a dictionary
                            if not isinstance(quad, dict):
                                print(f"Warning: Quadruplet at line {line_num} contains non-dictionary item: {quad}")
                                continue

                            # Extract parts of the quadruplet, handle possible missing values
                            aspect = quad.get('Aspect', 'Unknown_Aspect')
                            category = quad.get('Category', 'Unknown_Category')
                            opinion = quad.get('Opinion', 'Unknown_Opinion')
                            va = quad.get('VA', '0.00#0.00')  # Default value if VA is missing
                            if va == '0.00#0.00':
                                print(f"Error: VA value is missing at line {line_num}!")
                                continue
                            if aspect == 'Unknown_Aspect':
                                print(f"Error: {input_key}-Aspect value is missing at line {line_num}!")
                                continue
                            if opinion == 'Unknown_Opinion' and (task == 2 or task == 3):
                                print(f"Error: {input_key}-Opinion value is missing at line {line_num}!")
                                continue
                            if category == 'Unknown_Category' and task == 3:
                                print(f"Error: {input_key}-Category value is missing at line {line_num}!")
                                continue

                            # Add parsed quadruplet to list
                            parsed_quadruplets.append({
                                'Aspect': aspect.lower(),
                                'Category': category.lower(),
                                'Opinion': opinion.lower(),
                                'VA': va
                            })
                        entry[output_key] = parsed_quadruplets
                    else:
                        # If Quadruplet exists but is not a list (e.g., null or other types), log warning and set to empty list
                        print(f"Warning: Quadruplet at line {line_num} is not a list type: {type(quadruplets)}")
                        entry[output_key] = []

                    # Add parsed entry to data list
                    data.append(entry)

                except json.JSONDecodeError as e:
                    print(f"JSON parsing error at line {line_num}: {e}")
                    # Can choose to skip problematic lines or record errors
                    continue
                except Exception as e:
                    print(f"An unknown error occurred while processing line {line_num}: {e}")
                    continue

    except Exception as e:
        print(f"An error occurred while reading file '{file_path}': {e}")
        return data  # Return empty list on failure instead of exiting

    return data

def print_data_summary(data, task=3):
    """
    Prints a brief summary of the loaded data.

    Args:
        data (list): The list of data entries obtained from the read_jsonl_file function.
    """
    print(f"\n--- Data Summary ---")
    print(f"Successfully loaded {len(data)} valid records.")

    if data:
        print(f"\nSample Data:")
        for i, entry in enumerate(data[:3]):  # Print the first 3 entries as examples
            print(f"  Record {i+1}:")
            print(f"    ID: {entry['ID']}")
            print(f"    Text: {entry['Text']}")
            print(f"    Quadruplets ({len(entry[key_name[task]])}):")
            for quad in entry[key_name[task]]:
                print(f"      - Aspect: '{quad['Aspect']}', Category: '{quad['Category']}', "
                      f"Opinion: '{quad['Opinion']}', VA: '{quad['VA']}'")
            if i < 2 and len(data) > 3:  # Add separator if there are more records
                print("    ...")

pred_dir = 'evaluation/subtask_1/'

for file in sorted(os.listdir(pred_dir)):
    if file.endswith('.jsonl'):
        path = os.path.join(pred_dir, file)
        print(f"\n--- Testing {file} ---")
        data = read_jsonl_file(path, task=1, data_type='pred')
        if data:
            print_data_summary(data, task=1)
        else:
            print("Failed to load data.")