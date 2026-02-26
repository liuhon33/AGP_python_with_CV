#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=AGP_RF_PCA_%A.out
#SBATCH --error=AGP_RF_PCA_%A.err
#SBATCH --mail-type=FAIL

# Navigate to your project directory
cd $SCRATCH/parallel/AGPMicrobiomeHostPredictions

# Load modules (ensure these match your cluster's available versions)
module load python/3.11
module load StdEnv/2023

# Activate your environment (ensure 'jupyter' and 'nbconvert' are installed here)
source ./my_sklearn_env/bin/activate

# Execute the notebook
# This creates 'RF_PCA_out.ipynb' with all cells populated
jupyter nbconvert --to notebook --execute RF_PCA.ipynb --output RF_PCA_out.ipynb