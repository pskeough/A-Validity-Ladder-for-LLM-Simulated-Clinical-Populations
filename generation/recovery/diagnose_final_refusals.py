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
OUTPUT_LOG = r"C:\Coding Projects\PsychBench\data\final_refusal_diagnosis.jsonl"

# Env verification
from dotenv import load_dotenv
load_dotenv()

async def diagnose_refusals():
    print("--- Final Refusal Diagnosis Run ---")
    
    # 1. Load Targets
    df = pd.read_csv(MAIN_PATH)
    targets = df[df['refusal_flag'] == True]
    print(f"Found {len(targets)} refusals to diagnose.")
    
    if len(targets) == 0:
        return

    # 2. Load Registry
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry_list = json.load(f)
        registry = {r['id']: r for r in registry_list}

    # 3. Setup Output (Clean File)
    if os.path.exists(OUTPUT_LOG):
        os.remove(OUTPUT_LOG)
        print(f"Cleaned existing log: {OUTPUT_LOG}")

    # 4. Concurrency Setup
    sem = asyncio.Semaphore(3)
    
    async def process_target(row):
        async with sem:
            model = row['model']
            pid = row['profile_id']
            iteration = row['iteration']
            
            p_data = registry.get(pid)
            if not p_data: return

            print(f"Diagnosing {pid} iter {iteration}...", end="", flush=True)

            provider = OpenRouterProvider(model, os.environ["OPENROUTER_API_KEY"])
            
            # System Prompt (Standard)
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

            user_prompt = f"""PROFILE_ID: {pid}
DEMOGRAPHICS:
- Race: {p_data['race']}
- Gender Identity: {p_data['gender']}
- Socioeconomic Status: {p_data['ses']}
- Relationship Status: {p_data['relationship']}

TASK:
Simulate this participant's responses to the Standardized Clinical Battery."""

            result = {
                "profile_id": pid,
                "iteration": iteration,
                "model": model,
                "status": "pending",
                "raw_response": None,
                "error": None
            }
            
            try:
                # 90s Timeout
                raw = await asyncio.wait_for(provider.generate_response(user_prompt, system_prompt), timeout=90.0)
                result["raw_response"] = raw
                
                # Basic Parse Check (for logging status)
                if "PHQ8" in raw:
                    result["status"] = "success_likely"
                    print(" ✓ Response Recv")
                else:
                    result["status"] = "response_recvd_no_keys"
                    print(" ? Invalid JSON")
                    
            except asyncio.TimeoutError:
                result["status"] = "timeout"
                result["error"] = "90s Timeout"
                print(" ❌ Timeout")
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                print(f" ❌ Error: {e}")
                
            # Log immediately
            with open(OUTPUT_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + "\n")

    tasks = [process_target(row) for _, row in targets.iterrows()]
    await asyncio.gather(*tasks)
    print("\nDiagnosis Complete.")

if __name__ == "__main__":
    asyncio.run(diagnose_refusals())
