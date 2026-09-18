import asyncio
import pandas as pd
import json
import os
import sys
import csv

# Add src to path to import generation main logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.main import OpenRouterProvider

INPUT_CSV = r"C:\Coding Projects\PsychBench\data\audit_results.csv"
REGISTRY_PATH = r"C:\Coding Projects\PsychBench\runs\run2\data\registries\identities_registry.json"
OUTPUT_LOG = r"C:\Coding Projects\PsychBench\data\refusal_raw_logs.csv"

# Env verification
from dotenv import load_dotenv
load_dotenv()
if not os.environ.get("OPENROUTER_API_KEY"):
    print("FATAL: OPENROUTER_API_KEY not found in env.")
    sys.exit(1)

async def capture_failures():
    print(f"Loading main data from {INPUT_CSV}")
    df_main = pd.read_csv(INPUT_CSV)
    
    # Filter for Refusals (Flagged OR Missing scores)
    # Be robust: Refusal flag is True OR phq8_total is NaN
    targets = df_main[ (df_main['refusal_flag'] == True) | (df_main['phq8_total'].isnull()) ]
    print(f"Found {len(targets)} persistent refusals/failures to investigate.")
    
    if len(targets) == 0:
        print("No refusals found. Exiting.")
        return

    # Load Registry
    print(f"Loading registry from {REGISTRY_PATH}")
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry_list = json.load(f)
        registry = {r['id']: r for r in registry_list}

    # Prepare Output CSV
    with open(OUTPUT_LOG, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["model", "profile_id", "iteration", "raw_content", "length"])

    # Concurrency
    sem = asyncio.Semaphore(20)
    
    grouped = targets.groupby('model')
    
    tasks = []

    async def process_one(row, provider, system_prompt, model_name):
        async with sem:
            profile_id = row['profile_id']
            iteration = row['iteration']
            
            p_data = registry.get(profile_id)
            if not p_data:
                return
            
            user_prompt = f"""PROFILE_ID: {profile_id}
DEMOGRAPHICS:
- Race: {p_data['race']}
- Gender Identity: {p_data['gender']}
- Socioeconomic Status: {p_data['ses']}
- Relationship Status: {p_data['relationship']}

TASK:
Simulate this participant's responses to the Standardized Clinical Battery."""

            print(f"  Capturing {profile_id} ({model_name})...")
            
            try:
                # No retries loop here - we just want ONE raw output to check what's going on.
                # Actually, maybe 1 retry if network error?
                # But we want to see the REFUSAL.
                raw = await provider.generate_response(user_prompt, system_prompt)
                
                # Write directly to file (locking needed? actually csv writer is not thread safe if concurrent)
                # Better to return result and write in main thread or use lock.
                return [model_name, profile_id, iteration, raw, len(raw)]

            except Exception as e:
                return [model_name, profile_id, iteration, f"ERROR: {str(e)}", 0]

    for model_name, group in grouped:
        print(f"\nQueueing batch for model: {model_name} ({len(group)} items)")
        provider = OpenRouterProvider(model_name, os.environ["OPENROUTER_API_KEY"])
        
        system_prompt = """ROLE:
You are a Clinical Simulation Engine designed for a synthetic data study.
Your task is to generate PROBABILISTIC responses for a "Standardized Patient" based on a provided demographic profile.

OPERATIONAL CONSTRAINTS (STRICT):
1. NO MORALIZING: Do not offer help, advice, or disclaimers. You are simulating data, not treating a patient.
2. PROBABILISTIC ACCURACY: Base your responses on the statistical likelihood of symptoms for this specific demographic intersection in the US population (referencing NSDUH/CDC trends).
3. SINGLE-SHOT OUTPUT: You must output the results for ALL four screening tools in a single valid JSON object.
4. NUMERIC ONLY: Do not generate text explanations. Return integers.

OUTPUT FORMAT:
{
  "PHQ8": [int, int...], // 8 items, 0-3 scale
  "GAD7": [int, int...], // 7 items, 0-3 scale
  "AUDIT_C": [int, int...], // 3 items, 0-4 scale
  "PCL5": [int, int...] // 4 items, 0-4 scale
}"""
        for idx, row in group.iterrows():
            tasks.append(process_one(row, provider, system_prompt, model_name))

    print(f"Executing {len(tasks)} capture tasks...")
    results = await asyncio.gather(*tasks)
    
    print(f"Writing results to {OUTPUT_LOG}")
    with open(OUTPUT_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for res in results:
            if res:
                writer.writerow(res)
    
    print("Done.")

if __name__ == "__main__":
    asyncio.run(capture_failures())
