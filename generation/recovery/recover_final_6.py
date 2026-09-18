import asyncio
import pandas as pd
import json
import os
import sys

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.main import OpenRouterProvider

MAIN_PATH = r"C:\Coding Projects\PsychBench\data\audit_results.csv"
REGISTRY_PATH = r"C:\Coding Projects\PsychBench\runs\run2\data\registries\identities_registry.json"
OUTPUT_LOG = r"C:\Coding Projects\PsychBench\data\recovery_final_6.jsonl"

# Env verification
from dotenv import load_dotenv
load_dotenv()

async def recover_final_6():
    print("--- Recovering Final 6 Failures (Aggressive 10x Retry) ---")
    
    # 1. Load Targets
    df = pd.read_csv(MAIN_PATH)
    targets = df[df['refusal_flag'] == True]
    print(f"Found {len(targets)} persistent items.")
    
    if len(targets) == 0:
        print("No failures left!")
        return

    # 2. Load Registry
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry_list = json.load(f)
        registry = {r['id']: r for r in registry_list}

    # 3. Setup Provider
    # Note: All 6 are likely GLM based on previous check, but we use row['model'] to be safe.
    
    # 4. Process
    sem = asyncio.Semaphore(3) # 3 concurrent
    
    async def process_item(row):
        async with sem:
            pid = row['profile_id']
            iteration = row['iteration']
            model = row['model']
            p_data = registry.get(pid)
            
            if not p_data: return

            provider = OpenRouterProvider(model, os.environ["OPENROUTER_API_KEY"])
            
            system_prompt = """ROLE:
You are a Clinical Simulation Engine.
TASK: Generate PROBABILISTIC responses for a "Standardized Patient".
OUTPUT: JSON with ARRAYS of integers.
FORMAT:
{
  "PHQ8": [0, 1, ...],
  "GAD7": [0, 1, ...],
  "AUDIT_C": [0, 1, 2],
  "PCL5": [0, 1, ...]
}
NO TEXT EXPLANATIONS. ONLY JSON."""

            user_prompt = f"""PROFILE: {pid}
Race: {p_data['race']}
Gender: {p_data['gender']}
SES: {p_data['ses']}
Rel: {p_data['relationship']}

TASK: Simulate Clinical Screening Responses."""

            print(f"[{pid} iter {iteration}] Starting 10 retries...")
            
            for attempt in range(1, 11):
                try:
                    # 60s timeout per attempt
                    raw = await asyncio.wait_for(provider.generate_response(user_prompt, system_prompt), timeout=60.0)
                    
                    # Validate
                    clean = raw
                    if "```json" in clean: clean = clean.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean: clean = clean.split("```")[1].split("```")[0].strip()
                    
                    try:
                        data = json.loads(clean)
                        if all(k in data for k in ["PHQ8", "GAD7", "AUDIT_C", "PCL5"]):
                            # SUCCESS
                            print(f" ✓ [{pid} iter {iteration}] Success on Attempt {attempt}")
                            
                            # Log
                            entry = {
                                "profile_id": pid,
                                "iteration": iteration,
                                "raw_response": raw, # Keep original
                                "status": "success",
                                "attempt": attempt
                            }
                            
                            with open(OUTPUT_LOG, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(entry) + "\n")
                            return # Exit function for this item
                        else:
                            print(f" ? [{pid} iter {iteration}] Attempt {attempt} missing keys")
                            
                    except json.JSONDecodeError:
                        print(f" ? [{pid} iter {iteration}] Attempt {attempt} invalid JSON")
                        
                except Exception as e:
                    print(f" ❌ [{pid} iter {iteration}] Attempt {attempt} Error: {e}")
                    
                # Wait briefly between retries
                await asyncio.sleep(2)
            
            print(f" ☠️ [{pid} iter {iteration}] FAILED all 10 attempts.")

    tasks = [process_item(row) for _, row in targets.iterrows()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    if os.path.exists(OUTPUT_LOG):
        os.remove(OUTPUT_LOG)
    asyncio.run(recover_final_6())
