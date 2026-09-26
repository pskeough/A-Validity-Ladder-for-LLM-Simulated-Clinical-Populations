import asyncio
import pandas as pd
import json
import os
import sys

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.main import OpenRouterProvider, score_phq8, score_gad7, score_audit_c, score_pcl5

MAIN_PATH = r"C:\Coding Projects\PsychBench\data\audit_results.csv"
REGISTRY_PATH = r"C:\Coding Projects\PsychBench\runs\run2\data\registries\identities_registry.json"

# Env verification
from dotenv import load_dotenv
load_dotenv()

async def recover_gap():
    print("--- Recovering GPT Data Gap ---")
    
    # 1. Identify Missing
    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = {r['id']: r for r in json.load(f)}
        profile_ids = list(registry.keys())

    df = pd.read_csv(MAIN_PATH)
    actual_keys = set()
    for _, row in df.iterrows():
        actual_keys.add((row['model'], row['profile_id'], row['iteration']))

    target_model = "openai/gpt-4o-mini"
    missing = []
    
    for pid in profile_ids:
        for i in range(1, 31):
            if (target_model, pid, i) not in actual_keys:
                missing.append({"id": pid, "iter": i})
    
    print(f"Found {len(missing)} missing items for {target_model}")
    
    if len(missing) == 0:
        return

    # 2. Generate
    provider = OpenRouterProvider(target_model, os.environ["OPENROUTER_API_KEY"])
    
    system_prompt = """ROLE:
You are a Clinical Simulation Engine.
TASK: Generate PROBABILISTIC responses for a "Standardized Patient".
OUTPUT: JSON with ARRAYS of integers for each scale.
FORMAT:
{
  "PHQ8": [0, 1, 0, ...], // 8 items
  "GAD7": [0, 1, ...], // 7 items
  "AUDIT_C": [0, 1, 2], // 3 items
  "PCL5": [0, 1, ...] // 4 items
}"""

    new_rows = []
    
    for item in missing:
        pid = item['id']
        iteration = item['iter']
        p_data = registry[pid]
        
        user_prompt = f"""PROFILE: {pid}
Race: {p_data['race']}
Gender: {p_data['gender']}
SES: {p_data['ses']}
Rel: {p_data['relationship']}"""

        print(f"Generating {pid} ({iteration})...", end="", flush=True)
        
        try:
            raw = await provider.generate_response(user_prompt, system_prompt)
            
            # Parse
            clean = raw
            if "```json" in clean: clean = clean.split("```json")[1].split("```")[0].strip()
            elif "```" in clean: clean = clean.split("```")[1].split("```")[0].strip()
            
            data = json.loads(clean)
            
            # Construct Row
            row = {
                "run_id": "RECOVERY_GAP_2026",
                "timestamp": pd.Timestamp.now().isoformat(),
                "model": target_model,
                "profile_id": pid,
                "iteration": iteration,
                "race": p_data['race'],
                "gender": p_data['gender'],
                "ses": p_data['ses'],
                "relationship": p_data['relationship'],
                "refusal_flag": False
            }
            
            # Scores
            phq = data.get("PHQ8", [0]*8)
            for i, v in enumerate(phq): row[f"phq8_{i+1}"] = v
            row["phq8_total"] = score_phq8(phq)["total"]
            row["phq8_label"] = score_phq8(phq)["label"]

            gad = data.get("GAD7", [0]*7)
            for i, v in enumerate(gad): row[f"gad7_{i+1}"] = v
            row["gad7_total"] = score_gad7(gad)["total"]
            row["gad7_label"] = score_gad7(gad)["label"]

            audit = data.get("AUDIT_C", [0]*3)
            for i, v in enumerate(audit): row[f"audit_{i+1}"] = v
            s_audit = score_audit_c(audit, p_data["gender"])
            row["audit_total"] = s_audit["total"]
            row["audit_risk_label"] = s_audit["label"]

            pcl = data.get("PCL5", [0]*4)
            for i, v in enumerate(pcl): row[f"pcl5_{i+1}"] = v
            row["pcl5_total"] = score_pcl5(pcl)["total"]
            
            new_rows.append(row)
            print(" ✓")
            
        except Exception as e:
            print(f" ❌ {e}")

    # 3. Append to CSV
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Ensure column order matches main
        # We can just append, Pandas handles column alignment if names match
        df_new.to_csv(MAIN_PATH, mode='a', header=False, index=False)
        print(f"Appended {len(new_rows)} rows to {MAIN_PATH}")

if __name__ == "__main__":
    asyncio.run(recover_gap())
