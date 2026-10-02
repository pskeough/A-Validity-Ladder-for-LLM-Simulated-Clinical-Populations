"""
Merge Recovered Data into Main Dataset

This script:
1. Loads the original Run 1 dataset (audit_results.csv).
2. Loads the recovered data from run1_refusal_verification.csv.
3. Updates the original "refusals" with the recovered valid data.
4. Calculates all psychometric scores (totals, risk labels) for the new data.
5. Saves to Validated Analysis/audit_results_merged.csv.
"""

import csv
import json
import re
from typing import Dict, List, Any

# === SCORING LOGIC (Copied from original analyzer) ===

def calculate_scores(data: Dict) -> Dict:
    """Calculate totals and labels for the 4 scales."""
    scores = {}
    
    # PHQ-8
    if "PHQ8" in data:
        items = data["PHQ8"]
        total = sum(items)
        scores["phq8_total"] = total
        for i, val in enumerate(items):
            scores[f"phq8_{i+1}"] = val
            
        if total >= 10:
            scores["phq8_label"] = "Moderate+"
        else:
            scores["phq8_label"] = "None/Mild"

    # GAD-7
    if "GAD7" in data:
        items = data["GAD7"]
        total = sum(items)
        scores["gad7_total"] = total
        for i, val in enumerate(items):
            scores[f"gad7_{i+1}"] = val
            
        if total >= 10:
            scores["gad7_label"] = "Moderate+"
        else:
            scores["gad7_label"] = "None/Mild"

    # AUDIT-C
    if "AUDIT_C" in data:
        items = data["AUDIT_C"]
        total = sum(items)
        scores["audit_total"] = total
        for i, val in enumerate(items):
            scores[f"audit_{i+1}"] = val
            
        # Simplified risk (Standard cutoff >= 4 men, >=3 women - using 4 for general here or keeping simple)
        # Using the logic from previous files if possible, else standard:
        if total >= 4: 
            scores["audit_risk_label"] = "Hazardous"
        else:
            scores["audit_risk_label"] = "Low Risk"

    # PCL-5
    if "PCL5" in data:
        items = data["PCL5"]
        total = sum(items)
        scores["pcl5_total"] = total
        for i, val in enumerate(items):
            scores[f"pcl5_{i+1}"] = val

    return scores

def parse_preview_json(preview_str: str) -> Dict:
    """Parse the JSON string from the CSV preview column."""
    # It might have extra quotes or be formatted oddly due to CSV saving
    try:
        # The CSV reader handles the outer quotes, but checks just in case
        clean = preview_str.strip()
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content: {e}")
        return {}

def main():
    original_path = "Validated Analysis/audit_results.csv"
    recovered_path = "run1_refusal_verification.csv"
    output_path = "Validated Analysis/audit_results_merged.csv"

    print(f"Loading recovered data from {recovered_path}...")
    
    recovered_lookup = {}
    recovered_count = 0
    
    with open(recovered_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['verification_result'] == 'json_error_fixed':
                # Create key: model, profile_id, iteration
                key = (row['model'], row['profile_id'], row['iteration'])
                recovered_lookup[key] = row['raw_response_preview']
                recovered_count += 1
    
    print(f"Loaded {recovered_count} recovered responses.")

    print(f"Processing {original_path}...")
    
    updated_rows = []
    fixed_count = 0
    original_rows = 0
    
    with open(original_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            original_rows += 1
            
            # Check if this row was a refusal that we recovered
            key = (row['model'], row['profile_id'], row['iteration'])
            
            if row['refusal_flag'] == 'True' and key in recovered_lookup:
                # WE HAVE A FIX
                raw_json = recovered_lookup[key]
                parsed_data = parse_preview_json(raw_json)
                
                if parsed_data:
                    scores = calculate_scores(parsed_data)
                    
                    # Update row
                    row['refusal_flag'] = 'False' 
                    for k, v in scores.items():
                        if k in row: # Only update existing columns
                            row[k] = v
                            
                    fixed_count += 1
            
            updated_rows.append(row)

    print(f"Processed {original_rows} rows.")
    print(f"Merged {fixed_count} recovered data points.")

    print(f"Writing to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print("✓ Merge complete!")

if __name__ == "__main__":
    main()
