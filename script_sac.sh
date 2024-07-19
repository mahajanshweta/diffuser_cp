#!/bin/bash -l

#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --signal=USR2
#SBATCH --job-name=walkerreplay
#SBATCH --output=/scratch/users/k20108107/diffuser_cp/output/diffuser.out.%j
#SBATCH --time=24:00:00
#SBATCH --mem=8000MB
#SBATCH --chdir=/scratch/users/k20108107/diffuser_cp/

module load cuda/11.8.0-gcc-11.4.0
module load cudnn/8.7.0.84-11.8-gcc-11.4.0
module load anaconda3/2021.05-gcc-13.2.0
source /users/${USER}/.bashrc
source activate diffuser
CUDA_VISIBLE_DEVICES=0 python train_sac.py walker2d-medium-replay-v2