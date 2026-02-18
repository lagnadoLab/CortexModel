#!/bin/bash
#SBATCH --job-name=int_param_sweep
#SBATCH --partition=short
#SBATCH --output=logs_rest/param_%A_%a.out
#SBATCH --error=logs_rest/param_%A_%a.err
#SBATCH --array=1-20000%500
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G # try to increase memory
#SBATCH --time=02:00:00 # up to 8 hours

source ~/myenv/bin/activate

OFFSET=0
# Use SLURM_ARRAY_TASK_ID directly as the integer parameter
VAL=$((SLURM_ARRAY_TASK_ID + OFFSET))        #80000

# Create results and logs directory if it doesn’t exist
mkdir -p results_rest
mkdir -p logs_rest

# Run the Python script with the integer parameter
echo "Running with PARAM_VAL=$VAL"
PARAM_VAL=$VAL python model_run_rest.py

