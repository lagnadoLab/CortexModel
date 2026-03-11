import argparse
import pandas as pd
import glob
import os

# Step 1: Find all your CSV files
parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["rest", "locomotion"])
parser.add_argument("--pc_type", required=True, choices=["sensitizers", "intermediates", "depressors"])
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

filename = f"good_fits_{args.condition}_{args.pc_type}.csv"

combined_df.to_csv(filename, index=False)