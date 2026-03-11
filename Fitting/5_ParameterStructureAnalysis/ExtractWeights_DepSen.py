import pandas as pd
import numpy as np
import argparse


# Load your CSV
parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["rest", "locomotion"])
parser.add_argument("--pc_type", required=True, choices=["sensitizers", "intermediates", "depressors"])
args = parser.parse_args()

if args.condition == "rest":
    df = pd.read_csv(f"good_fits_rest_{args.pc_type}.csv")
    # Scale to average amplitude during stim
    if args.pc_type == "sensitizers": 
        scale = 0.9375792
    elif args.pc_type == "intermediates":
        scale = 1.0897695
    elif args.pc_type == "depressors":
        scale = 0.88019663
elif args.condition == "locomotion":
    df = pd.read_csv(f"good_fits_locomotion_{args.pc_type}.csv")
    # Scale to average amplitude during stim
    if args.pc_type == "sensitizers": 
        scale = 1.1043589
    elif args.pc_type == "intermediates":
        scale = 1.163496
    elif args.pc_type == "depressors":
        scale = 0.83324456
else:
    raise ValueError(f"Unknown condition: {args.condition}")

row = df.drop(columns=["ID"]).mean()

# Extract the weights in the desired order
weights = np.array([
    row["w_d_0"],  # PC -> PC
    row["w_d_1"],  # FF -> PC
    row["w_d_2"],  # SS -> PC
    row["w_d_3"],  # FB -> PC
    row["w_d_4"],  # PV -> PC
    row["w_d_5"],  # SST -> PC
], dtype=np.float32) / scale

##Transfer weights to a table##
# Table size
n_rows = 6
n_cols = 1
table = np.full((n_rows, n_cols), np.nan)

positions = [
    (0, 0), # PC -> PC w_d_0
    (3, 0), # FF -> PC w_d_1
    (4, 0), # SS -> PC w_d_2
    (5, 0), # FB -> PC w_d_3
    (2, 0), # PV -> PC w_d_4
    (1, 0), # SST -> PC w_d_5
]
for w, (r, c) in zip(weights, positions):
    table[r, c] = w

celltype_pre = ["PC", "SST", "PV", "FF", "SS", "FB"]
celltype_post = [f"{args.pc_type}"]
df_save = pd.DataFrame(table, index=celltype_pre, columns = celltype_post)

## Saving ##
df_save.to_csv(f"ConnWeights_{args.condition}_{args.pc_type}.csv", index=True)

