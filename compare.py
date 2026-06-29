import json
import csv
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

HUMAN_DIR = Path("data/human_annotations")
WAV2VEC_DIR = Path("wav2vec2_outputs")

THRESHOLD_MS = 90
THRESHOLD_SEC = THRESHOLD_MS / 1000.0

# =====================================================
# STATS
# =====================================================

total_files = 0
red_flag_files = 0

total_words = 0
flagged_words = 0

total_matching_words = 0
total_start_error_sec = 0.0
total_end_error_sec = 0.0
total_combined_error_sec = 0.0

# =====================================================
# REPORTS
# =====================================================

red_flag_report = []
word_error_report = []

# =====================================================
# PARSE FUNCTIONS
# =====================================================

def load_human_annotation(txt_file):
    words = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            start = float(parts[0])
            end = float(parts[1])
            word = " ".join(parts[2:]).upper()
            words.append({"word": word, "start": start, "end": end})
    return words


def load_wav2vec_annotation(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for item in data:
        words.append({
            "word": item["word"].upper(),
            "start": float(item["start"]),
            "end": float(item["end"])
        })
    return words


# =====================================================
# MAIN COMPARISON
# =====================================================

for human_file in sorted(HUMAN_DIR.glob("*_Annotated.txt")):
    stem = human_file.stem.replace("_Annotated", "")
    wav_file = WAV2VEC_DIR / f"{stem}.json"

    if not wav_file.exists():
        print(f"Missing wav2vec file: {stem}")
        continue

    total_files += 1

    human_words = load_human_annotation(human_file)
    wav_words = load_wav2vec_annotation(wav_file)

    # Flag for tracking file issues
    file_has_flag = False
    file_flagged_words = 0

    # 1. Word count mismatch
    if len(human_words) != len(wav_words):
        red_flag_files += 1
        red_flag_report.append({
            "file": stem,
            "reason": "WORD_COUNT_MISMATCH",
            "human_words": len(human_words),
            "wav_words": len(wav_words),
            "flagged_words": 0,
            "total_words": len(human_words)
        })
        print(f"[FLAG] {stem} word count mismatch ({len(human_words)} vs {len(wav_words)})")
        continue

    # 2. Word by word comparison
    for idx in range(len(human_words)):
        h = human_words[idx]
        w = wav_words[idx]

        total_words += 1

        # Text mismatch
        if h["word"] != w["word"]:
            file_has_flag = True
            file_flagged_words += 1
            flagged_words += 1

            word_error_report.append({
                "file": stem,
                "word_index": idx,
                "human_word": h["word"],
                "wav_word": w["word"],
                "start_diff_ms": "",
                "end_diff_ms": "",
                "reason": "WORD_MISMATCH"
            })
            continue

        # Timing difference evaluation (70ms Threshold)
        start_diff = abs(h["start"] - w["start"])
        end_diff = abs(h["end"] - w["end"])
        combined_diff = start_diff + end_diff

        total_matching_words += 1
        total_start_error_sec += start_diff
        total_end_error_sec += end_diff
        total_combined_error_sec += combined_diff

        if start_diff > THRESHOLD_SEC or end_diff > THRESHOLD_SEC:
            file_has_flag = True
            file_flagged_words += 1
            flagged_words += 1

            word_error_report.append({
                "file": stem,
                "word_index": idx,
                "human_word": h["word"],
                "wav_word": w["word"],
                "start_diff_ms": round(start_diff * 1000, 2),
                "end_diff_ms": round(end_diff * 1000, 2),
                "reason": "TIMESTAMP_DIFF"
            })

    # Record flagged files
    if file_has_flag:
        red_flag_files += 1
        red_flag_report.append({
            "file": stem,
            "reason": "WORD_OR_TIMESTAMP_ERRORS",
            "human_words": len(human_words),
            "wav_words": len(wav_words),
            "flagged_words": file_flagged_words,
            "total_words": len(human_words)
        })

# =====================================================
# SUMMARY & OUTPUT
# =====================================================

percentage = (flagged_words / total_words * 100) if total_words > 0 else 0.0

mean_start_error_ms = (total_start_error_sec / total_matching_words * 1000) if total_matching_words > 0 else 0.0
mean_end_error_ms = (total_end_error_sec / total_matching_words * 1000) if total_matching_words > 0 else 0.0
mean_combined_error_ms = (total_combined_error_sec / total_matching_words * 1000) if total_matching_words > 0 else 0.0

summary = {
    "files_compared": total_files,
    "red_flag_files": red_flag_files,
    "total_words": total_words,
    "flagged_words": flagged_words,
    "percentage_flagged_words": round(percentage, 4),
    "mean_start_error_ms": round(mean_start_error_ms, 2),
    "mean_end_error_ms": round(mean_end_error_ms, 2),
    "mean_combined_error_ms": round(mean_combined_error_ms, 2),
    "matching_words_evaluated": total_matching_words
}

with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

with open("red_flag_files.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["file", "reason", "human_words", "wav_words", "flagged_words", "total_words"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(red_flag_report)

with open("word_level_errors.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["file", "word_index", "human_word", "wav_word", "start_diff_ms", "end_diff_ms", "reason"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(word_error_report)

print("\n" + "=" * 50)
print(f"Files compared       : {total_files}")
print(f"Red flag files       : {red_flag_files}")
print(f"Total words          : {total_words}")
print(f"Words > 90 ms        : {flagged_words}")
print(f"Percentage flagged   : {percentage:.2f}%")
print(f"Mean Start Error     : {mean_start_error_ms:.2f} ms")
print(f"Mean End Error       : {mean_end_error_ms:.2f} ms")
print("=" * 50)