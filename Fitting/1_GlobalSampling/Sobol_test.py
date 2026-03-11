from scipy.stats.qmc import Sobol, scale
import numpy as np
import pandas as pd
import os

# Set number of dimensions and samples
dim = 20
seednum = 42
n_samples = 131072  # Initial samples 2^17
output_file = "sobol_samples.csv"

# Load existing samples if available
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    last_id = df_existing["ID"].max()
else:
    df_existing = pd.DataFrame()
    last_id = 0

# Initialize Sobol sampler
sobol = Sobol(d=dim, scramble=True, seed=seednum)
if last_id > 0:
    sobol.fast_forward(last_id)

# Generate Sobol samples in [0, 1]^18
samples_unit = sobol.random(n=n_samples)

# Define bounds (PC-PC Max 0.3, Other weights Max 2.5, k Min 0.01 Max 0.1)
lower_bounds = [0.00001] + [0.00001]*(dim-2) + [0.01]   # Avoid exact 0
upper_bounds = [0.3] + [2.5]*(dim-2) + [0.1]

# Scale to actual parameter ranges
samples_scaled = scale(samples_unit, lower_bounds, upper_bounds)

# Combine and save
df_new = pd.DataFrame(samples_scaled, columns=[f"param_{i+1}" for i in range(dim-1)] + ["k"])
df_new["ID"] = np.arange(last_id + 1, last_id + n_samples + 1)
df_full = pd.concat([df_existing, df_new], ignore_index=True)
df_full.to_csv(output_file, index=False)

print(f"Extended dataset to {df_full.shape[0]} total samples.")