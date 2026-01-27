#!/bin/bash
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks=1                 # 1 task
#SBATCH --cpus-per-task=8          
#SBATCH --time=12:00:00            # hh:mm:ss
#SBATCH --output=AGP_regression_%A_%a.out  # %A = main job ID, %a = task ID
#SBATCH --error=AGP_regression_%A_%a.err   # Separate error file
#SBATCH --mail-type=FAIL

cd $SCRATCH/parallel/AGPMicrobiomeHostPredictions
module load python/3.11
module load StdEnv/2023
source ./my_sklearn_env/bin/activate
python regression_update.py