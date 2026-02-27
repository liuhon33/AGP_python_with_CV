#!/bin/bash
#SBATCH --nodes=1                  # 1 node
#SBATCH --ntasks=1                 # 1 task
#SBATCH --cpus-per-task=48
#SBATCH --time=6:00:00
#SBATCH --job-name=jupyter_notebook
#SBATCH --output=jupyter_notebook_%j.out
#SBATCH --error=jupyter_notebook_%j.err

# 1. Set the runtime directory via Environment Variable (highly reliable)
# $SLURM_TMPDIR is the best place, but /tmp/ also works if that fails
export JUPYTER_RUNTIME_DIR=$SLURM_TMPDIR/jupyter_runtime
mkdir -p $JUPYTER_RUNTIME_DIR
export JUPYTER_DATA_DIR=$SLURM_TMPDIR/jupyter_data
mkdir -p $JUPYTER_DATA_DIR

# 2. Get tunneling info (keep your existing port/node logic)
port=$(shuf -i8000-9999 -n1)
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=trillium.alliancecan.ca

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}
# Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}
SSH login: $user
SSH port: 22
#Use a Browser on your local machine to go to:
https://localhost:${port}  (prefix w/ http:// instead if the browser complains that secured connection cannot be established)
" > jupyter_notebook_$SLURM_JOBID.login_info

# 3. Start jupyter (removed the unrecognized flag)
source ~/.bash_profile
cd /scratch/liuhon33/parallel/AGPMicrobiomeHostPredictions
module load r/4.3.1
source my_sklearn_env/bin/activate

# Redirect all Jupyter metadata to scratch or temp storage
export JUPYTER_CONFIG_DIR="/scratch/liuhon33/.jupyter"
export JUPYTER_DATA_DIR="$SLURM_TMPDIR/jupyter_data"
export IPYTHONDIR="$SLURM_TMPDIR/ipython"

# Create the directories to ensure they exist
mkdir -p $JUPYTER_CONFIG_DIR
mkdir -p $JUPYTER_DATA_DIR
mkdir -p $IPYTHONDIR

jupyter lab --no-browser --port=${port} --ip=${node}