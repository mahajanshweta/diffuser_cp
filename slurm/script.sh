#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --job-name=diffuser
#SBATCH --output=/scratch/users/k20108107/diffuser_cp/diffuser.out.%j
#SBATCH --time=0-3:00:00
#SBATCH --mem=6000MB
#SBATCH --chdir=/scratch/users/k20108107/diffuser_cp/

source diffuser/bin/activate
python plan_guided.py --dataset halfcheetah-medium-v2 --logbase logs/pretrained