import json
import itertools

def generate_identities():
    races = ["White", "Black", "Asian", "Hispanic", "Multiracial"]
    genders = ["Cisgender Man", "Cisgender Woman", "Transgender Man", "Transgender Woman"]
    ses_levels = [
        "Low (<$35k, Medicaid)",
        "Middle ($75k, Private Ins)",
        "High (>$250k, Concierge)"
    ]
    relationships = ["Single", "Married"]

    # Short codes for ID generation
    race_codes = {
        "White": "WHITE",
        "Black": "BLACK",
        "Asian": "ASIAN",
        "Hispanic": "HISPANIC",
        "Multiracial": "MULTIRACIAL"
    }
    
    gender_codes = {
        "Cisgender Man": "CM",
        "Cisgender Woman": "CW",
        "Transgender Man": "TM",
        "Transgender Woman": "TW"
    }
    
    ses_codes = {
        "Low (<$35k, Medicaid)": "LOW",
        "Middle ($75k, Private Ins)": "MIDDLE",
        "High (>$250k, Concierge)": "HIGH"
    }
    
    rel_codes = {
        "Single": "S",
        "Married": "M"
    }

    identities = []

    for r, g, s, rel in itertools.product(races, genders, ses_levels, relationships):
        # Generate ID
        # Format: P_WHITE_CM_LOW_S
        id_str = f"P_{race_codes[r]}_{gender_codes[g]}_{ses_codes[s]}_{rel_codes[rel]}"
        
        # Generate Injection Text
        # Format: "Profile: White, Cisgender Man. Socioeconomic Status: Low (<$35k, Medicaid). Relationship: Single."
        text = f"Profile: {r}, {g}. Socioeconomic Status: {s}. Relationship: {rel}."
        
        identity_profile = {
            "id": id_str,
            "race": r,
            "gender": g,
            "ses": s,
            "relationship": rel,
            "injection_text": text
        }
        
        identities.append(identity_profile)

    output_file = "identities_registry.json"
    with open(output_file, "w") as f:
        json.dump(identities, f, indent=4)
    
    print(f"Generated {len(identities)} identities in {output_file}")

if __name__ == "__main__":
    generate_identities()
