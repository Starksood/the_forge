
import json
import re

def attempt_fix_unquoted_keys(text):
    # Matches words followed by a colon if they are after { or ,
    return re.sub(r'([{,]\s*)([a-z_][a-z0-9_]*)\s*:', r'\1"\2":', text)

bad_json = '{   "chunk_boundaries": [     {       offset: 0,       offset_type: "new_entity"     },     {       offset: 1240,       offset_type: "new_entity"     } ] }'

print(f"Original: {bad_json}")
fixed = attempt_fix_unquoted_keys(bad_json)
print(f"Fixed: {fixed}")

try:
    data = json.loads(fixed)
    print("Successfully parsed!")
    print(data)
except Exception as e:
    print(f"Failed to parse: {e}")
