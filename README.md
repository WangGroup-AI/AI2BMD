<p align="center">
  <img src="./code/logo.png" alt="ViSNet-PIMA logo" width="100%">
</p>

<h1 align="center">
  Enhancing long-range interaction modeling for ab initio biomolecular calculation with ViSNet-PIMA
</h1>

<p align="center">
  <a><img src="https://img.shields.io/badge/License-MIT-green"></a>
  <a><img src="https://img.shields.io/github/last-commit/WangGroup-AI/AI2BMD"></a>
  <a><img src="https://img.shields.io/badge/Python-3.10-red"></a>
</p> 

## Overview

ViSNet-PIMA (short for “**ViSNet** with **P**hysics-**I**nformed **M**ultipole **A**ggregator”) combines multipole expansion theory with ViSNet to accurately model both short-range and long-range molecular interactions, outperforming state-of-the-art MLFFs in energy and force predictions.

## 🌟 Quick Start

### **We have provided a jupyter notebook named `run_all.ipynb`. Users can click the `Run All` button to reproduce all the results.**

![Button](./code/button.png)

For reproduction, users can also open a terminal and run:

```bash
bash ./run_all.sh
```

This command sequentially runs all evaluation tasks and displays the consolidated results.

These experimental are also saved in the `results` folder. Meanwhile, the generated simulation trajectories are saved in `code/AI2BMD-PIMA_simulation/Logs-1_rep_chig.c0/SimulationResults/*`.

## Environments

### 1. Clone the repository

Clone this repository and enter its root directory.

### 2. Create the Conda environment

The complete environment used by the unified reproduction workflow is specified in [`environment.yml`](./code/ViSNet-PIMA_test_on_Chig_and_MD22/environment.yml). From the repository root, run:

```bash
conda env create --file code/ViSNet-PIMA-final/environment.yml
conda activate ViSNet
```

The environment uses Python 3.10, PyTorch 2.3.1 with CUDA 11.8, and the corresponding PyTorch Geometric extensions.

### 3. Verify CUDA availability

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

For GPU inference, `CUDA available` should report `True` and the expected GPU should be displayed.

## Reproduction Workflow

Run the complete workflow with the following command:

```bash
./code/run
```

The master script executes the following stages in order:

| Order | Evaluation | Entry point | Primary output |
|:---:|---|---|---|
| 1 | AIMD-Chig | [`inference_Chig.sh`](./code/ViSNet-PIMA_test_on_Chig_and_MD22/inference_Chig.sh) | `results/log_Chig/inference_results.pt` |
| 2 | MD22 | [`inference_MD22.sh`](./code/ViSNet-PIMA_test_on_Chig_and_MD22/inference_MD22.sh) | `results/log_md22_*/inference_results.pt` |
| 3 | Trp-cage learning curve and ablation study | [`inference_Trp-cage.sh`](./code/ViSNet-PIMA_train_and_test_on_Trp/inference_Trp-cage.sh) | `results/log_learning_curve_and_ablation_studys/*/inference_results.pt` |
| 4 | NCI Atlas | [`inference_NCIA.sh`](./code/ViSNet-PIMA_test_on_NCIA/inference_NCIA.sh) | `results/log_NCIA/ncia_binding_visnet-pima.csv` |

All terminal output produced by the unified workflow is also saved to:

```text
results/output
```

### MD22 evaluation order

The seven MD22 subsets are evaluated in the following order, matching `inference_MD22.sh`:

1. Ac-Ala3-NHMe
2. DHA
3. Stachyose
4. AT-AT
5. AT-AT-CG-CG
6. Buckyball catcher
7. Double-walled nanotube

### Trp-cage evaluation order

The learning-curve and ablation models are evaluated in the following order, matching `inference_Trp-cage.sh`:

1. 20% finetuned with pretraining
2. 40% finetuned with pretraining
3. 60% finetuned with pretraining
4. 80% finetuned with pretraining
5. 100% finetuned with pretraining
6. 100% finetuned without pretraining

## Running Individual Evaluation Stages

Each stage can be run independently from the repository root while preserving the same order and configuration used by the unified workflow.

### 1. AIMD-Chig

```bash
bash code/ViSNet-PIMA-final/inference_Chig.sh
```

### 2. MD22

```bash
bash code/ViSNet-PIMA-final/inference_MD22.sh
```

### 3. Trp-cage learning curve and ablation study

```bash
bash code/AI2BMD-PIMA/inference_Trp-cage.sh
```

### 4. NCI Atlas

```bash
bash code/ViSNet-PIMA_test_on_NCIA/inference_NCIA.sh
```

### 5. Consolidated result display

After all required result files have been generated, display the consolidated tables with:

```bash
python code/show_results.py
```

The summary contains the AIMD-Chig results, all seven MD22 subsets, the six Trp-cage learning-curve and ablation results, and the NCI Atlas interaction-energy results.

### 6. Simulation and calculation with AI<sup>2</sup>BMD-PIMA

We provide scripts for running molecular dynamics simulations and evaluating multi-frame PDB trajectories with AI<sup>2</sup>BMD-PIMA. Run the following commands from the repository root:

```bash
bash code/AI2BMD-PIMA_simulation/run.sh
bash code/AI2BMD-PIMA_simulation/infer_multiframe_pdb.sh
```

The animation below shows a representative molecular dynamics trajectory generated with the AI<sup>2</sup>BMD-PIMA simulation workflow.

<p align="center">
  <img src="./code/trajectory.gif" alt="AI2BMD-PIMA molecular dynamics trajectory" width="800">
</p>


## Training

### AIMD-Chig

To train ViSNet-PIMA on AIMD-Chig, run:

```bash
CUDA_VISIBLE_DEVICES=0 python code/ViSNet-PIMA-final/train.py \
  --conf code/ViSNet-PIMA-final/examples_MD22_AIMD-Chig/ViSNet-Chignolin.yml \
  --dataset-root data/AIMD-Chig \
  --log-dir results/log_Chig
```

Example training configuration files for the other datasets are provided in [`examples_MD22_AIMD-Chig`](./code/ViSNet-PIMA_test_on_Chig_and_MD22/examples_MD22_AIMD-Chig/) and can be used in the same way.

### MD22

To train ViSNet-PIMA on MD22, run:

```bash
CUDA_VISIBLE_DEVICES=0 python code/ViSNet-PIMA-final/train.py \
  --conf code/ViSNet-PIMA-final/examples_MD22_AIMD-Chig/ViSNet-MD22-Ac_Ala3_NHMe.yml \
  --dataset-root data/md22-dataset \
  --log-dir results/log_MD22
```

## License

This project is licensed under the terms of the MIT license.
