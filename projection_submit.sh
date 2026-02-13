#!/bin/bash
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks=1                 # 1 task
#SBATCH --cpus-per-task=24          
#SBATCH --time=12:00:00            # hh:mm:ss
#SBATCH --output=project%A_%a.out  # %A = main job ID, %a = task ID
#SBATCH --error=project%A_%a.err   # Separate error file
#SBATCH --mail-type=FAIL

cd $SCRATCH/parallel/AGPMicrobiomeHostPredictions
module load python/3.11
module load StdEnv/2023
source ./my_sklearn_env/bin/activate
python projection.py