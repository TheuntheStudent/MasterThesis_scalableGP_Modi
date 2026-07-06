#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=19:59:00
 
module load 2024
 
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

unset LD_LIBRARY_PATH
source $HOME/kozax_env/bin/activate

python -u Kozax_Final_Experiment_Brax.py