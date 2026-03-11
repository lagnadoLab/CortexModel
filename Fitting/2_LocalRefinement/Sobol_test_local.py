from scipy.stats.qmc import Sobol, scale
import numpy as np
import pandas as pd
import os
import argparse

# Set number of dimensions and samples
dim = 20
seednum = 42
n_samples = 32768  # Initial samples 2^17

# Load existing good solutions

parser = argparse.ArgumentParser()
parser.add_argument("--condition", required=True, choices=["rest", "locomotion"])
args = parser.parse_args()

filename = f"good_fits_{args.condition}.csv"
output_file = f"sobol_local_{args.condition}.csv"

df = pd.read_csv(filename)

# Extract mean and sd of weights in the desired order

means = df.drop(columns=["ID"]).mean()

STDVs = df.drop(columns=["ID"]).std()

meanweights = np.array(
    [means[f"w_{i}"] for i in range(19)] + [means["k"]],
    dtype=np.float32)

sdweights = np.array(
    [STDVs[f"w_{i}"] for i in range(19)] + [STDVs["k"]],
    dtype=np.float32)

# Initialize Sobol sampler
sobol = Sobol(d=dim, scramble=True, seed=seednum)

# Generate Sobol samples in [0, 1]^18
samples_unit = sobol.random(n=n_samples)

# Define bounds (Mean+- 2 STDV
meanweights = np.array(meanweights)
sdweights = np.array(sdweights)

lower_bounds = meanweights - 2*sdweights
upper_bounds = meanweights + 2*sdweights

# Clip values betweem 0.00001 and 2.5
lower_bounds = np.clip(lower_bounds, 0.00001, 2.5)
upper_bounds = np.clip(upper_bounds, 0.00001, 2.5)

# Convert to Python lists
lower_bounds = lower_bounds.tolist()
upper_bounds = upper_bounds.tolist()

# Scale to actual parameter ranges
samples_scaled = scale(samples_unit, lower_bounds, upper_bounds)

# Combine and save
df_new = pd.DataFrame(samples_scaled, columns=[f"param_{i+1}" for i in range(dim-1)] + ["k"])
df_new["ID"] = np.arange(1, n_samples + 1)
df_new.to_csv(output_file, index=False)