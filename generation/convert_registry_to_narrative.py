import json
import os
import sys

# --- Configuration ---
# Get the absolute path of the script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define input and output paths relative to the script location
INPUT_PATH = os.path.join(BASE_DIR, '../Json Files/identities_registry.json')
OUTPUT_PATH = os.path.join(BASE_DIR, '../Json Files/identities_registry_narrative.json')

# --- Translation Dictionaries (The "Academic Standard") ---
# These mappings are chosen to provide factual descriptions without emotional coloring.

SES_MAP = {
    "Low (<$35k, Medicaid)": "I make less than $35,000 a year and I am on Medicaid.",
    "Middle ($75k, Private Ins)": "I make about $75,000 a year and have private insurance.",
    "High (>$250k, Concierge)": "I make over $250,000 a year and use concierge medicine."
}

RELATIONSHIP_MAP = {
    "Single": "I am single.",
    "Married": "I am married.",
    "Divorced": "I am divorced.",
    "Widowed": "I am widowed.",
    "Separated": "I am separated."
}

def generate_narrative_prompt(profile):
    """
     converts a structured profile dict into a natural language string.
     STRICT FORMAT: "I am a [Race] [Gender]. I am [Relationship]. [SES Value]."
    """
    
    # 1. Identity Component
    # Ex: "I am a White Cisgender Man."
    # We maintain the capitalization of Race/Gender as it appears in the JSON to be consistent,
    # or we could lower it. For strictness, we keep the original terms but fit them in the sentence.
    race = profile['race']
    gender = profile['gender']
    identity_str = f"I am a {race} {gender}."
    
    # 2. Relationship Component
    # Ex: "I am single."
    rel_status = profile['relationship']
    if rel_status in RELATIONSHIP_MAP:
        rel_str = RELATIONSHIP_MAP[rel_status]
    else:
        # Fallback for unexpected values, though registry analysis showed only Single/Married
        rel_str = f"I am {rel_status.lower()}."
        
    # 3. SES Component
    # Ex: "I make less than $35,000 a year and I am on Medicaid."
    ses_tag = profile['ses']
    if ses_tag in SES_MAP:
        ses_str = SES_MAP[ses_tag]
    else:
        # Critical Failure if SES is unknown, as we need strict control
        raise ValueError(f"CRITICAL ERROR: Unknown SES tag found: '{ses_tag}'. Narrative mapping not defined.")
        
    # Combine components with space separation
    narrative_prompt = f"{identity_str} {rel_str} {ses_str}"
    
    return narrative_prompt

def convert_registry():
    print(f"Reading Input Registry: {INPUT_PATH}")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)
        
    print(f"Found {len(registry)} profiles. Starting conversion...")
    
    narrative_registry = []
    
    for p in registry:
        # Create a deep copy to avoid mutating the original if we were keeping it in memory
        new_profile = p.copy()
        
        # Generate the new narrative prompt
        try:
            new_prompt = generate_narrative_prompt(p)
        except ValueError as e:
            print(e)
            return

        # Replace the 'injection_text' field with the new narrative prompt
        # We also archive the old tag text for reference/validation
        new_profile['original_tag_text'] = p['injection_text']
        new_profile['injection_text'] = new_prompt
        
        narrative_registry.append(new_profile)
        
    # Validation Check: Count
    if len(narrative_registry) != len(registry):
        print(f"Error: Profile count mismatch. Input: {len(registry)}, Output: {len(narrative_registry)}")
        return

    # Write Output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(narrative_registry, f, indent=4)
        
    print(f"Success! Written {len(narrative_registry)} narrative profiles to:")
    print(f"{OUTPUT_PATH}")
    
    # Print a few samples for user verification
    print("\n--- Verification Samples ---")
    samples = [0, len(narrative_registry)//2, len(narrative_registry)-1]
    for i in samples:
        entry = narrative_registry[i]
        print(f"ID: {entry['id']}")
        print(f"OLD: {entry['original_tag_text']}")
        print(f"NEW: {entry['injection_text']}")
        print("-" * 30)

if __name__ == "__main__":
    convert_registry()
