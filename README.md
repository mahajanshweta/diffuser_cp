# Conformal Prediction with Diffusion-based planning


Our code is based on [Planning with Diffusion for Flexible Behavior Synthesis](https://diffusion-planning.github.io/).

The [main branch](https://github.com/mahajanshweta/diffuser_cp/tree/main) contains code for training diffusion models and planning via value-function guided sampling on the D4RL locomotion environments and perfoming conformal prediction.


## Quickstart

Load a pretrained diffusion model and sample from it in your browser with [diffuser-sample.ipynb].


## Installation

```
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

## Using pretrained models

### Downloading weights

Download pretrained diffusion models and value functions with:
```
./download_pretrained.sh
```

This command downloads and extracts a tarfile to `logs/pretrained`. The models are organized according to the following structure:
```
└── logs/pretrained
    ├── ${environment_1}
    │   ├── diffusion
    │   │   └── ${experiment_name}
    │   │       ├── state_${epoch}.pt
    │   │       ├── sample-${epoch}-*.png
    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl
    │   ├── values
    │   │   └── ${experiment_name}
    │   │       ├── state_${epoch}.pt
    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl
    │   └── plans
    │       └── defaults
    │           ├── 0
    │           ├── 1
    │           ├── ...
    │           └── 149
    │
    ├── ${environment_2}
    │   └── ...
```

The `state_${epoch}.pt` files contain the network weights and the `config.pkl` files contain the instantation arguments for the relevant classes.
The png files contain samples from different points during training of the diffusion model.
Within the `plans` subfolders, there are the results of 150 evaluation trials for each environment using the default hyperparameters.


### Planning

To plan with guided sampling, run:
```
python plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained
```

The `--logbase` flag points the experiment loaders to the folder containing the pretrained models.
You can override planning hyperparameters with flags, such as `--batch_size 8`, but the default
hyperparameters are a good starting point.

## Training from scratch

1. Train a diffusion model with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

The default hyperparameters are listed in [locomotion:diffusion](config/locomotion.py#L22-L65).
You can override any of them with flags, eg, `--n_diffusion_steps 100`.

2. Train a value function with:
```
python scripts/train_values.py --dataset halfcheetah-medium-expert-v2
```
See [locomotion:values](config/locomotion.py#L67-L108) for the corresponding default hyperparameters.


3. Plan using your newly-trained models with the same command as in the pretrained planning section, simply replacing the logbase to point to your new models:
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs
```
See [locomotion:plans](config/locomotion.py#L110-L149) for the corresponding default hyperparameters.

**Deferred f-strings.** Note that some planning script arguments, such as `--n_diffusion_steps` or `--discount`,
do not actually change any logic during planning, but simply load a different model using a deferred f-string.
For example, the following flags:
```
---horizon 32 --n_diffusion_steps 20 --discount 0.997
--value_loadpath 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}'
```
will resolve to a value checkpoint path of `values/defaults_H32_T20_d0.997`. It is possible to change the horizon of the diffusion model after training, but not for the value function.



## Running on HPC

The slurm jobs scripts are located in the [slurm](slurm) folder which has scripts for trainig the diffuser and sac and planning with it. We store the outputs from these jobs in 'data' folder. 


## Conformal Prediction and Conformalized Quantile Regression

1. To run conformal prediction use the following command, the results are stored in [resultsCP.txt](resultsCP.txt) :

```
python CP.py hopper-medium-v2

```

2. To run CQR use the following command, the results are stored in [resultsCQR.txt](resultsCQR.txt) :

```
python CQR.py hopper-medium-v2

```

## Plotting 

To plot the results of effect of calibration set size on coverage and interval width, use the jupyter notebook [plotting.ipynb](plotting.ipynb)