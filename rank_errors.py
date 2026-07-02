import pandas as pd

# ==========================
# CONFIG
# ==========================

INPUT_FILE = "outputs/comparison_reports/flag_words.csv"
OUTPUT_FILE = "outputs/comparison_reports/flag_words_ranked.csv"

# ==========================
# LOAD CSV
# ==========================

df = pd.read_csv(INPUT_FILE)

# ==========================
# CALCULATE COMBINED ERROR
# ==========================

df["combined_error_ms"] = (
    df["start_diff_ms"] + df["end_diff_ms"]
)

# ==========================
# SORT DESCENDING
# ==========================

df_sorted = df.sort_values(
    by="combined_error_ms",
    ascending=False
)

# ==========================
# SAVE RESULT
# ==========================

df_sorted.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved sorted file: {OUTPUT_FILE}")

# Show top errors
print(df_sorted.head(20))