import pandas as pd
import glob
import os

# Step 1: Find all your CSV files
folder_path = os.path.join(".", "reports")  # change this to your actual folder path folder "results_filtered"
csv_files = glob.glob(os.path.join(folder_path, "opto_ID_*.csv"))

# Step 2: Collect transformed rows
rows = []

for file in csv_files:
    df = pd.read_csv(file)


    # Pivot: One row per file, columns from 'Parameter', values from 'Value'
    df_wide = (
    df.pivot(index="ID", columns=["Parameter", "Type"], values="Value")
      .reset_index()
    )
    
    # Flatten the multi-level column names
    df_wide.columns = [
        "_".join(map(str, col)).strip() if isinstance(col, tuple) else col
        for col in df_wide.columns
    ]

    rows.append(df_wide)

# Step 3: Combine all rows
combined_df = pd.concat(rows, ignore_index=True)

# Step 4: Define filtering conditions
redchi_thre = 0.16
opto_threup = 4
opto_threlow = 1.5

cond_redchi = ((combined_df["redchi_PV_Chr"] < redchi_thre) & 
    (combined_df["redchi_PV_Arch"] < redchi_thre) &
    (combined_df["redchi_SST_Arch"] < redchi_thre) &
    (combined_df["redchi_SST_Chr"] < redchi_thre)
)
    

cond_opto = ((combined_df["opto_param_PV_Chr"] > opto_threlow) & (combined_df["opto_param_PV_Chr"] < opto_threup) & 
             (combined_df["opto_param_PV_Arch"]  > opto_threlow) & (combined_df["opto_param_PV_Arch"]  < opto_threup) &
             (combined_df["opto_param_SST_Arch"] > opto_threlow) & (combined_df["opto_param_SST_Arch"] < opto_threup) &
             (combined_df["opto_param_SST_Chr"]  > opto_threlow) & (combined_df["opto_param_SST_Chr"]  < opto_threup)
)
# Step 5: Apply filters
filteredopto_df = combined_df[cond_redchi & cond_opto]

all_fits = pd.read_csv("good_fits_local_locomotion.csv")

IDs_Good = filteredopto_df["ID_"]

filtered_df = all_fits[all_fits["ID"].isin(IDs_Good)]

filtered_df = filtered_df.sort_values("ID")
filtered_df = filtered_df.reset_index(drop=True)
filtered_df["ID"] = filtered_df.index + 1


# Step 6: Save to CSV
filtered_df.to_csv("good_fits_opto_locomotion.csv", index=False)