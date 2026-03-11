import argparse
import pandas as pd
import glob
import os

# Step 1: Find all your CSV files
parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["rest", "locomotion"])
args = parser.parse_args()

folder_path = os.path.join(".", f"results_{args.condition}") 
csv_files = glob.glob(os.path.join(folder_path, "output_ID_*.csv"))

# Step 2: Collect transformed rows
rows = []

for file in csv_files:
    df = pd.read_csv(file)

    # Drop 'Stderr' column
    df = df.drop(columns=["Stderr"])

    # Pivot: One row per file, columns from 'Parameter', values from 'Value'
    df_wide = df.pivot(index="ID", columns="Parameter", values="Value").reset_index()

    rows.append(df_wide)

# Step 3: Combine all rows
combined_df = pd.concat(rows, ignore_index=True)

# Step 4: filter based on threshold

THRESHOLDS = {
"rest": 3,
"locomotion": 10.0,
}


ChiThresh = THRESHOLDS[args.condition]

filtered = combined_df[combined_df["redchi"] < ChiThresh]

filename = f"good_fits_{args.condition}.csv"

filtered.to_csv(filename, index=False)