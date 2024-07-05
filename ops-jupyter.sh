#!/bin/bash -l

#SBATCH --job-name=torch-jupyter
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --signal=USR2
#SBATCH --time=02:00:00

module load anaconda3/2021.05-gcc-13.2.0

# get unused socket per https://unix.stackexchange.com/a/132524
readonly DETAIL=$(python -c 'import datetime; print(datetime.datetime.now())')
readonly IPADDRESS=$(hostname -I | tr ' ' '\n' | grep '10.211.4.')
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cat 1>&2 <<END
1. SSH tunnel from your workstation using the following command:

   ssh -NL 8888:${HOSTNAME}:${PORT} ${USER}@hpc.create.kcl.ac.uk

   and point your web browser to http://localhost:8888/lab?token=<add the token from the jupyter output below>

Time started: ${DETAIL}

When done using the notebook, terminate the job by
issuing the following command on the login node:

      scancel -f ${SLURM_JOB_ID}

END

source /users/${USER}/.bashrc
source activate diffuser
# source torch-venv/bin/activate
jupyter-lab --port=${PORT} --ip=${IPADDRESS} --no-browser

printf 'notebook exited' 1>&2