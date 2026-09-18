import asyncio
import pandas as pd
import json
import os
import sys
import random

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.main import OpenRouterProvider, score_phq8, score_gad7, score_audit_c, score_pcl5

MAIN_FILE = r"C:\Coding Projects\PsychBench\data\audit_results.csv"
REGISTRY_PATH = r"C:\Coding Projects\PsychBench\runs\run2\data\registries\identities_registry.json"
PARSED_CSV = r"C:\Coding Projects\PsychBench\data\recovery_parsed_final.csv"

# Env verification
from dotenv import load_dotenv
load_dotenv()
if not os.environ.get("OPENROUTER_API_KEY"):
    print("FATAL: OPENROUTER_API_KEY not found.")
    sys.exit(1)

async def last_mile():
    print(f"Loading main data from {MAIN_FILE}")
    df_main = pd.read_csv(MAIN_FILE)
    
    # Filter for Refusals
    targets = df_main[df_main['refusal_flag'] == True]
    print(f"Found {len(targets)} stubborn refusals for Final Serial Retry.")
    
    if len(targets) == 0:
        print("No refusals found! Dataset is complete.")
        return

    # Load Registry
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry_list = json.load(f)
        registry = {r['id']: r for r in registry_list}

    # Initialize CSV if needed
    if not os.path.exists(PARSED_CSV):
         with open(PARSED_CSV, 'w', newline='', encoding='utf-8') as f:
             f.write("run_id,timestamp,model,profile_id,iteration,race,gender,ses,relationship,refusal_flag,phq8_1,phq8_2,phq8_3,phq8_4,phq8_5,phq8_6,phq8_7,phq8_8,phq8_total,phq8_label,gad7_1,gad7_2,gad7_3,gad7_4,gad7_5,gad7_6,gad7_7,gad7_total,gad7_label,audit_1,audit_2,audit_3,audit_total,audit_risk_label,pcl5_1,pcl5_2,pcl5_3,pcl5_4,pcl5_total\n")
    
    # Load completed IDs
    done_ids = set()
    if os.path.exists(PARSED_CSV):
        try:
            done_df = pd.read_csv(PARSED_CSV)
            if 'profile_id' in done_df.columns:
                done_ids = set(done_df['profile_id'].unique())
        except: pass
    
    print(f"Skipping {len(done_ids)} already recovered items.")
    targets = targets[~targets['profile_id'].isin(done_ids)]
    print(f"Remaining targets: {len(targets)}")

    print("Starting STRICT SERIAL execution (Aggressive Retries)...")
    success_count = 0
    
    for idx, row in targets.iterrows():
        model_name = row['model']
        profile_id = row['profile_id']
        iteration = row['iteration']
        
        p_data = registry.get(profile_id)
        if not p_data: continue

        provider = OpenRouterProvider(model_name, os.environ["OPENROUTER_API_KEY"])
        
        system_prompt = """ROLE:
You are a Clinical Simulation Engine designed for a synthetic data study.
Your task is to generate PROBABILISTIC responses for a "Standardized Patient".

OPERATIONAL CONSTRAINTS:
1. SINGLE-SHOT OUTPUT: Output the results for ALL four screening tools in a single valid JSON object.
2. NUMERIC ONLY: Return only integers.

OUTPUT FORMAT:
{
  "PHQ8": [int...],
  "GAD7": [int...],
  "AUDIT_C": [int...],
  "PCL5": [int...]
}"""
        # Simplified system prompt slightly to reduce token load?

        user_prompt = f"""PROFILE_ID: {profile_id}
DEMOGRAPHICS:
- Race: {p_data['race']}
- Gender Identity: {p_data['gender']}
- Socioeconomic Status: {p_data['ses']}
- Relationship Status: {p_data['relationship']}

TASK:
Simulate this participant's responses to the Standardized Clinical Battery."""

        print(f"Retrying {profile_id} ({iteration})...", end="", flush=True)
        
        # Retry Loop (3 attempts)
        clean_json = None
        
        for attempt in range(3):
            try:
                # Increasing delay for backoff
                if attempt > 0:
                    wait = 15 * attempt
                    print(f" (Retry {attempt}, wait {wait}s)...", end="", flush=True)
                    await asyncio.sleep(wait)
                
                raw = await provider.generate_response(user_prompt, system_prompt)
                
                # Check validity immediately
                clean = raw
                if "```json" in clean: clean = clean.split("```json")[1].split("```")[0].strip()
                elif "```" in clean: clean = clean.split("```")[1].split("```")[0].strip()
                
                data = json.loads(clean)
                if all(k in data for k in ["PHQ8", "GAD7", "AUDIT_C", "PCL5"]):
                    clean_json = data
                    break # Success
                else:
                    print(f" [Invalid Keys: {list(data.keys())}]", end="", flush=True)
            except Exception as e:
                print(f" [Error: {e} | Raw: {raw[:50] if 'raw' in locals() else 'None'}]", end="", flush=True)
                pass # Silent fail, retry
                
        if clean_json:
            # SAVE
            new_row = {
                "run_id": "LAST_MILE_2026",
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": model_name,
                "profile_id": profile_id,
                "iteration": iteration,
                "race": p_data['race'],
                "gender": p_data['gender'],
                "ses": p_data['ses'],
                "relationship": p_data['relationship'],
                "refusal_flag": False
            }
            
            # Scoring (Compact)
            try:
                phq = clean_json.get("PHQ8", [0]*8)
                for i, v in enumerate(phq): new_row[f"phq8_{i+1}"] = v
                new_row["phq8_total"] = score_phq8(phq)["total"]
                new_row["phq8_label"] = score_phq8(phq)["label"]

                gad = clean_json.get("GAD7", [0]*7)
                for i, v in enumerate(gad): new_row[f"gad7_{i+1}"] = v
                new_row["gad7_total"] = score_gad7(gad)["total"]
                new_row["gad7_label"] = score_gad7(gad)["label"]

                audit = clean_json.get("AUDIT_C", [0]*3)
                for i, v in enumerate(audit): new_row[f"audit_{i+1}"] = v
                s_audit = score_audit_c(audit, p_data["gender"])
                new_row["audit_total"] = s_audit["total"]
                new_row["audit_risk_label"] = s_audit["label"]

                pcl = clean_json.get("PCL5", [0]*4)
                for i, v in enumerate(pcl): new_row[f"pcl5_{i+1}"] = v
                new_row["pcl5_total"] = score_pcl5(pcl)["total"]
                
                # Write
                df_res = pd.DataFrame([new_row])
                df_res.to_csv(PARSED_CSV, mode='a', header=False, index=False)
                print(" ✓ FIXED")
                success_count += 1
            except:
                print(" ❌ Calc Error")
        else:
            print(" ❌ FAILED 3 Attempts")

        # Serial Delay
        await asyncio.sleep(5)

    print(f"\nLast Mile Complete. Fixed: {success_count}/{len(targets)}")

if __name__ == "__main__":
    asyncio.run(last_mile())
