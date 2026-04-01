# AI<sup>2</sup>BMD: AI-powered *ab initio* biomolecular dynamics simulation

## Contents

- [Feature Updates](#feature-updates)
- [Get Started](#get-started)
- [System Requirements](#system-requirements)
- [Advanced Setup](#advanced-setup)
- [Citation](#citation)

## Feature Updates
- **Add Geometry Optimization function**

## Get Started

The source code of AI<sup>2</sup>BMD is hosted in this repository.

We can run a molecular dynamics simulation as follows.

```shell
# the Chignolin protein is already in the example dir
# launch the program, with all simulation parameters set to default values
apptainer run --nv ai2bmd.sif
. /opt/env
cd ~/AI2BMD
python ./src/main.py --prot-file ./examples/chig.pdb --task simulation --base-dir ~ --sim-steps 1000 --temp-k 300 --timestep 1 --preeq-steps 0 --record-per-steps 1 
```

Here we use a very simple protein `Chignolin` as an example.
The program will run a simulation with the default parameters.

The results will be placed in a new directory `Logs-chig`.
The directory contains the simulation trajectory file:

- chig-traj.traj: The full trajectory file in ASE binary format.

Note: Currently, AI<sup>2</sup>BMD supports MD simulations for proteins with neutral terminal caps (ACE and NME), single chain and standard amino acids.


## System Requirements

### Hardware Requirements

The AI<sup>2</sup>BMD program runs on x86-64 GNU/Linux systems.
We recommend a machine with the following specs:

- **CPU**: 8+ cores
- **Memory**: 32+ GB
- **GPU**: CUDA-enabled GPU with 8+ GB memory

The program has been tested on the following GPUs:
- RTX 5090

## Advanced Setup
### Environment
The runtime libraries and requirents are packed into a apptainer sif file for convenience and practicality. Before launching the apptainer, you need to install the apptainer package.


### Protein File Preparation

The input file for AI<sup>2</sup>BMD should be `.pdb` format.
If hydrogen atoms are missing in the `.pdb` file, hydrogens should be added.
Then, the protein should be capped with ACE (acetyl) at the N-terminus and NME (N-methyl) at the C-terminus.  These steps can be efficiently done using the PyMOL software with the following commands as a reference.

```python
from pymol import cmd
pymol.finish_launching()
cmd.load("your_protein.pdb","molecule")
cmd.h_add("molecule") # Adding hydrogen

cmd.wizard("mutagenesis")
cmd.get_wizard().set_n_cap("acet")
selection = "/%s//%s/%s" % (molecule, chain, resi) #selection of N-term
cmd.get_wizard().do_select(selection)
cmd.get_wizard().apply()

cmd.get_wizard().set_c_cap("nmet")
selection = "/%s//%s/%s" % (molecule, chain, resi) #selection of N-term
cmd.get_wizard().do_select(selection)
cmd.get_wizard().apply()

cmd.set_wizard()
```

Next, you can use AmberTools' `pdb4amber` utility to adjust atom names in the `.pdb` file, specifically ensuring compatibility for ACE and NME as required by `ai2bmd`. The atom names for ACE and NME should conform to the following:

- ACE: C, O, CH3, H1, H2, H3
- NME: N, C, H, H1, H2, H3

```
pdb4amber -i your_protein.pdb -o processed_your_protein.pdb
```

In addition, please verify that there are no `TER` separators in the protein chain. Additionally, the residue numbering should start from 1 without gaps.


After completing the above steps, your `.pdb` file should resemble the following format:

```
ATOM      1  H1  ACE     1      10.845   8.614   5.964  1.00  0.00           H
ATOM      2  CH3 ACE     1      10.143   9.373   5.620  1.00  0.00           C
ATOM      3  H2  ACE     1       9.425   9.446   6.437  1.00  0.00           H
ATOM      4  H3  ACE     1       9.643   9.085   4.695  1.00  0.00           H
ATOM      5  C   ACE     1      10.805  10.740   5.408  1.00  0.00           C
ATOM      6  O   ACE     1      10.682  11.417   4.442  1.00  0.00           O
...
ATOM    170  N   NME    12       9.499   8.258  10.367  1.00  0.00           N
ATOM    171  H   NME    12       9.393   8.028  11.345  1.00  0.00           H
ATOM    172  C   NME    12       8.845   7.223   9.569  1.00  0.00           C
ATOM    173  H1  NME    12       7.842   6.990   9.925  1.00  0.00           H
ATOM    174  H2  NME    12       8.798   7.589   8.543  1.00  0.00           H
ATOM    175  H3  NME    12       9.418   6.305   9.435  1.00  0.00           H
END

```

You can also take the protein files in `examples` folder as reference. Note, currently, the machine learning potential doesn't support the protein with disulfide bonds well. We will update it soon.

### Preprocess
During the preprocess, the solvated sytem is built and encounted energy minimization and alternative pre-equilibrium stages. Currently, AI<sup>2</sup>MD provides two methods for the preprocess via the argument `preprocess_method`.

If you choose the `FF19SB` method, the system will go through solvation, energy minimization, heating and several pre-equilibrium stages. To accelerate the preprocess by multiple CPU cores and GPUs, you should get AMBER software packages and modify the corresponding commands in `src/AIMD/preprocess.py`.

If you choose the `AMOEBA` method, the system will go through solvation and energy minimization stages. We highly recommend to perform pre-equilibrium simulations to let the simulation system fully relaxed.

If you choose the `None` method, the system will skip the preprocess stage.

### Simulation
AI<sup>2</sup>BMD provides two modes for performing the production simulations via the argument `mode`. The default mode of `fragment` represents protein is fragmented into dipeptides and then calculated by the machine learning potential in every simulation step.

AI<sup>2</sup>BMD also supports to train the machine learning potential by yourselves and perform simulations without fragmentation. The `visnet` mode represents the potential energy and atomic forces of the protein are directly calculated by the ViSNet model as a whole molecule without fragmentation. When using this mode, you need to train ViSNet model with the data of the molecules by yourself, upload the model to `src/ViSNet` and give the corresponding value to the argument `ckpt-type`. In this way, you can use AI<sup>2</sup>BMD simulation program to simulate any kinds of molecules beyond proteins. 

### Geometry Optimization
AI<sup>2</sup>BMD supports geometry optimization by setting the argument "task" to "optimization". This feature is based on ASE framework, and use the visnet model to predict the energy and force. The default GO method is LBFGS and BFGSLinsearch.

To perform the whole AI<sup>2</sup>BMD program including the preprocess, please use the following commands as reference.


```shell
apptainer run --nv ai2bmd.sif
. /opt/env
cd ~/AI2BMD
python ./src/main.py --prot-file ./examples/chig.pdb --task simulation --base-dir ~ --sim-steps 1000 --temp-k 300 --timestep 1 --preeq-steps 0 --record-per-steps 1 
#                 '---------------- required argument ---------------' '-------------------------------------optional arguments------------------------------------'
#
# Notable optional arguments:
#
# [Task directory mapping options]
#   --base-dir path/to/base-dir    Directory for running task (default: current directory)
#   --log-dir  path/to/log-dir     Directory for logs, results (default: base-dir/Logs-protein-name)
#   --src-dir  path/to/src-dir     Mount src-dir in place of src/ from this repository (default: not used)
#
# [Task parameter options]
#   --sim-steps nnn                Task steps
#   --temp-k nnn                   Simulation temperature in Kelvin
#   --timestep nnn                 Time-step (fs) for simulation
#   --preeq-steps nnn              Pre-equilibration simulation steps for each constraint
#   --max-cyc nnn                  Maximum energy minimization cycles in preprocessing
#   --preprocess-method [method]   The method for preprocess
#   --mode [mode]                  Use fragmentation or not during the task
#   --record-per-steps nnn         The frequency to save trajectory
#
# [Performance tweaks]
#   --device-strategy [strategy]   The compute device allocation strategy
#       excess-compute                 Reserves last GPU for non-bonded/solvent computation
#       small-molecule                 Maximize resources for model inference
#       large-molecule                 Improve performance for large molecules
#   --chunk-size nnn               Number of atoms in each batch (reduces memory consumption)
```

### Post-analysis
The format of the simulation trajectory is `.traj` of ASE format. To convert it to `.dcd` format for visualization, you can install MDAnalysis first and take `src/utils/traj2dcd.py` as reference with the following commands:

```shell
python traj2dcd.py --input xxx.pdb --output xxx.dcd --pdb xxx.pdb --num-atoms nnn --stride nnn

# arguments
# --input         The name of the input trajectory file
# --output        The name of the output trajectory file
# --pdb           The reference pdb file corresponding to the input trajectory
# --num-atoms     The number of atoms for protein or the whole solvated system
# --stride        The frequency to output the trajectory
```

## Citation
(#: co-first author; *: corresponding author)

Cui, T., Wang, Z., Wang, T.* (2026). Enhancing non-local interaction modeling for ab initio biomolecular calculations and simulations with ViSNet-PIMA. BioRxiv. (BioRxiv)

Cui, T.#, Zhou, Y.#, Wang, T.* (2025). Recent advances in artificial intelligence–driven biomolecular dynamics simulations based on machine learning force fields. Curr. Opin. Struct. Biol., 95: 103191. 

Wang, T.#*, He, X.#, Li, M.#, Li, Y.#, Bi, R., Wang, Y., Cheng, C., Shen, X., Meng, J., Zhang, H., Liu, H., Wang, Z., Li, S., Shao, B.*, Liu, T. Y. (2024). Ab initio characterization of protein molecular dynamics with AI2BMD. Nature, 635: 1019–1027. 

Wang, Y.#, Wang, T.#*, Li, S.#, He, X., Li, M., Wang, Z., Zheng, N., Shao, B.*, Liu, T. Y. (2024). Enhancing geometric representations for molecules with equivariant vector-scalar interactive message passing. Nat. Commun., 15(1): 313.


