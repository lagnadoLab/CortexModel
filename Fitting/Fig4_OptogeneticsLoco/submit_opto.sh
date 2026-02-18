#!/bin/bash
#SBATCH --job-name=optoAvg
#SBATCH --output=logs_opto/param_%A_%a.out
#SBATCH --error=logs_opto/param_%A_%a.err
#SBATCH --array=1-1%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G # try to increase memory
#SBATCH --time=00:30:00 # up to 8 hours
#SBATCH --partition=short

source ~/myenv/bin/activate

OFFSET=0
# Use SLURM_ARRAY_TASK_ID directly as the integer parameter
VAL=$((SLURM_ARRAY_TASK_ID + OFFSET))        #80000

# Create results and logs directory if it doesn’t exist
mkdir -p reports
mkdir -p logs_opto

# Run the Python script with the integer parameter
echo "Running with PARAM_VAL=$VAL"
PARAM_VAL=$VAL python Opto_run_loco.py

