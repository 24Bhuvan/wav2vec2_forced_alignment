import json
from pathlib import Path

# Folder containing JSON files
INPUT_DIR = Path("wav2vec2_outputs")

# Folder to save TXT annotation files
OUTPUT_DIR = Path("audacity_formated")
OUTPUT_DIR.mkdir(exist_ok=True)

json_files = sorted(INPUT_DIR.glob("*.json"))

print(f"Found {len(json_files)} JSON files")

for json_file in json_files:
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        output_file = OUTPUT_DIR / f"{json_file.stem}.txt"

        with open(output_file, "w", encoding="utf-8") as out:

            for item in data:
                start = float(item["start"])
                end = float(item["end"])
                word = str(item["word"]).upper()

                # Format:
                # 0.469029    1.320000    PREPARATION
                out.write(
                    f"{start:.6f}\t{end:.6f}\t{word}\n"
                )

        print(f"Converted: {json_file.name}")

    except Exception as e:
        print(f"Error in {json_file.name}: {e}")

print("\nDone.")
print(f"TXT files saved to: {OUTPUT_DIR}")
