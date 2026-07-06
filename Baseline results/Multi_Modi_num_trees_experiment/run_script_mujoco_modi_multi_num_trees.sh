#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=9:59:59
 
module load 2024
 
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

unset LD_LIBRARY_PATH
source $HOME/modiko_mujoco/bin/activate

python -u Modi_Extra_Experiment_Mujoco_Multi_num_trees.py