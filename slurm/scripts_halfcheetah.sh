#!/bin/bash -l

#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --signal=USR2
#SBATCH --job-name=halfcheetah
#SBATCH --output=/scratch/users/k20108107/diffuser_cp/output/diffuser.out.%j
#SBATCH --time=35:00:00
#SBATCH --mem=24000MB
#SBATCH --chdir=/scratch/users/k20108107/diffuser_cp/

module load anaconda3/2021.05-gcc-13.2.0
source /users/${USER}/.bashrc
source activate diffuser
CUDA_VISIBLE_DEVICES=0 python plan_guided.py --dataset halfcheetah-medium-v2 --logbase logs/pretrained
CUDA_VISIBLE_DEVICES=0 python plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained
CUDA_VISIBLE_DEVICES=0 python plan_guided.py --dataset halfcheetah-medium-replay-v2 --logbase logs/pretrained

