import pandas as pd
import json

# ==========================
# CONFIG
# ==========================

INPUT_FILE = "outputs/comparison_reports/flag_files.csv"
OUTPUT_FILE = "outputs/comparison_reports/flagged_files.json"

# ==========================
# READ CSV
# ==========================

df = pd.read_csv(INPUT_FILE)

# ==========================
# EXTRACT FILE NAMES
# ==========================

flagged_files = df["file"].tolist()

# ==========================
# CREATE JSON FORMAT
# ==========================

output = {
    "flagged_files": flagged_files
}

# ==========================
# SAVE JSON
# ==========================

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=4)

print(f"Saved: {OUTPUT_FILE}")
print(output)