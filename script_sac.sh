#!/bin/bash -l

#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --signal=USR2
#SBATCH --job-name=sac
#SBATCH --output=/scratch/users/k20108107/diffuser_cp/output/diffuser.out.%j
#SBATCH --time=02:00:00
#SBATCH --mem=8000MB
#SBATCH --chdir=/scratch/users/k20108107/diffuser_cp/

module load anaconda3/2021.05-gcc-13.2.0
source /users/${USER}/.bashrc
source activate diffuser
CUDA_VISIBLE_DEVICES=0 python train_sac.py
