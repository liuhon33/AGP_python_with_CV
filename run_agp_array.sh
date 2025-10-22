#!/bin/bash
#SBATCH --job-name=AGP_Array
#SBATCH --output=AGP_%A_%a.out  # %A = main job ID, %a = task ID
#SBATCH --error=AGP_%A_%a.err   # Separate error file

# --- Resource Requests ---
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # 8 CPUs for each job
#SBATCH --time=06:00:00        # 4 hours (adjust as needed)

# --- Job Array ---
# This line creates 4 jobs, numbered 1, 2, 3, and 4
#SBATCH --array=1-4

# --- Account and Notifications ---
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=your.email@domain.com # <-- IMPORTANT: Change this

# --- Setup ---
echo "--- Loading environment for job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID ---"
cd $SCRATCH/parallel/AGPMicrobiomeHostPredictions
module load StdEnv/2023
module load python/3.11
# Removed 'module load cuda/12.6' as scikit-learn doesn't use it
source my_sklearn_env/bin/activate
echo "--- Environment loaded ---"

# --- Run ---
# $SLURM_ARRAY_TASK_ID will be 1, 2, 3, or 4.
# We pass this ID as an argument to the Python script.
echo "Starting array task $SLURM_ARRAY_TASK_ID"
srun python Code/cohort_classifer.py $SLURM_ARRAY_TASK_ID
echo "--- Array task $SLURM_ARRAY_TASK_ID finished ---"