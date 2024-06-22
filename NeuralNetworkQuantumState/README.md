

# Neural Network Quantum States algorithm for computing ground states of quantum manybody problems


We propose a high-performance and scalable neural network quantum state (NNQS) method for $\textit{ab initio}$ electronic structure calculations. The NNQS leverages a PyTorch implementation of a transformer-based architecture (wave function ansatz) for inference and backward propagation. 
Additionally, we implement a data-centric parallelization scheme for the variational Monte Carlo (VMC) algorithm using MPI.jl and accelerate the local energy calculation kernel on the GPU using CUDA C/C++.
Julia serves as the main control language, with PyCall.jl utilized to call PyTorch and ccall used to invoke CUDA C/C++.


## Install

### Some needed libs
- python >= 3.8
    - numpy==1.23.5
    - openfermion==1.5.1
    - pyscf==2.1.1
    - PyYAML==6.0
    - scipy==1.10.1
    - torch==1.13.1+cu116

### Install steps

1. Install conda example:
    ```shell
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
    ```

2. Install python and libs
    ```shell
    # install python >= 3.8
    conda create -n py38 python=3.8
    # activate conda environment
    conda activate py38
    # pip3 install requirements
    pip install -r requirements.txt
    # if your cuda11.7, then you can run
    pip3 install torch
    ```

## How to Run

### Detail steps (Pure CPU version)
```shell
# enter project root directory
cd NeuralNetworkQuantumState
# set python load path
source setup.sh

# Config calculate local energy implementation version 
# if you choose CPP_CPU calculate local energy, run `make cpu`
# if you want multi-thread speedup, using openmp by `make cpu-openmp`
# if you want GPU speedup, run `make gpu`
cd NeuralNetworkQuantumState/local_energy
make clean && make cpu # for cpu backend
make clean && make gpu # for gpu backend

cd NeuralNetworkQuantumState/test
# Submit your job
# eg. `./submit.sh ../molecules/lih/qubit_op.data configs/config-lih.yaml lih-test`
# This will output into file `out.lih-test`
# Your can modify the submit.sh acoording your compute platform
./submit.sh <hamiltonian_path> <config_file_name> <out_file_name_suffix>
```

### Parallel Version (MPI)
Using sampling, local energy calculation and grad update parallel, just set the `use_uniq_sampling_parallel` in the YAML configuration file (e.g. `config-h4.yaml`):

```yaml
use_uniq_sampling_parallel: true
```
## Code structure
```shell
NeuralNetworkQuantumState/
├── README.md
├── local_energy # calculate local energy lib
│   ├── Makefile
│   ├── backend
│   │   ├── cpu
│   │   │   └── calculate_local_energy_cpu.cpp
│   │   └── cuda
│   │       ├── calculate_local_energy.cu
│   │       └── hashTable.cuh
│   ├── env.sh # environment
│   ├── include
│   │   ├── calculate_local_energy.cuh
│   │   ├── calculate_local_energy_cpu.h
│   │   ├── hamiltonian
│   │   │   └── hamiltonian.h
│   │   ├── switch_backend.h
│   │   └── utils
│   │       ├── numpy.h
│   │       └── timer.h
│   ├── interface
│   │   ├── julia
│   │   │   ├── calculate_local_energy_wrapper.jl
│   │   │   └── eloc.jl
│   │   └── python
│   │       ├── eloc.py
│   │       ├── pybind_calculate_local_energy.cpp
│   │       └── setup.py
│   ├── test # unit test
│   │   ├── __init__.py
│   │   ├── test_calculate_local_energy.cpp
│   │   ├── test_eloc.jl
│   │   ├── test_eloc.py
│   │   └── utils
│   └── testcases
├── molecules
│   ├── gen.sh
│   ├── parse.sh
│   ├── pyscf_helper.py
│   ├── reference_energy.txt # molecules reference energy
│   ├── template
│   │   ├── gen_ham.py # generate hamiltonian
│   │   └── gen_ham_loc.py
│   └── utils.py
├── requirements.txt # python packages
├── setup.sh # set environments
├── src
│   ├── NeuralNetworkQuantumState.py # main interface
│   ├── networks # pure NN model
│   │   ├── mingpt
│   │   │   ├── model.py
│   │   │   └── utils.py
│   │   └── networks.py
│   ├── pretrain.py
│   ├── sampler.py # MCMC sampler
│   ├── utils
│   │   ├── GPUMemTrack.py
│   │   ├── cisd.py # generate cisd or fci states of molecules
│   │   ├── complex_helper.py
│   │   ├── config.py
│   │   └── utils.py
│   └── wavefunctions
│       ├── DecoderWaveFunction.py # complex WF
│       ├── DecoderWaveFunctionReal.py # real WF
└── test
    ├── configs # config examples
    │   ├── config-h2.yaml
    │   ├── config-h2o.yaml
    │   └── config-lih.yaml
    ├── job_script # job script utils
    │   ├── gen.sh
    │   ├── job-gpu-tp.sh
    │   └── showjob.sh
    ├── nnqs.py # VMC training loop
    └── submit.sh # job submit script
```
## Configuration
```yaml
device: 'cuda' # specify accelaration platform: {'cpu' | 'cuda'}
local_energy_version: "CPP_GPU" # local energy speedup: {"CPP_GPU" | "CPP_CPU"}

n_samples: 1000000 # total samples (batch size)
n_epoches: 1000000 # total train epoches
save_model: 1 # whether save model checkpoints
save_per_epoches: 500 # how long save the model once
load_model: 0 # load model from checkpoint_path
checkpoint_path: "checkpoints/h10_sto6g_1.84-d64L6h4-1.64L0-iter6000-rank0.pt"

do_ham: 0 # Hamiltonian calculation

model_fp: fp32 # model parameter precision

transfer_learning: false # only valid for load_model, `true` train from the 0-th, otherwise from checkpoints continue
seed: 333 # global seed
pos_independ: false
use_sr: false
use_clip_grad: false
# 'qk1' ('complex') / 'qk2' / 'qk3' ('real')
psi_type: 'qk1' # equal to psi_type: 'complex'
only_infer: false
search_init_best_seed: false
weak_scaling: false

use_uniq_sampling_parallel: false # parallel sampling
incr_per_epoch: 50
min_partition_samples: 64
partition_algo: 'number'

max_transfer: 0

use_samples_recursive: true
n_samples_min: 1e5
n_samples_max: 1e12
n_unq_samples_min: 6e3
n_unq_samples_max: 5e4
n_samples_scale_factor: 1.3

drop_samples: false
drop_samples_eps: 1e-8

use_amp_infer_batch: true
amp_infer_batch_size: 8192

use_kv_cache: false
use_grad_accumulation: false
grad_accumulation_width: 8192

sampling_algo: "bfs"
sampling_dfs_width: 128
sampling_dfs_uniq_samples_min: 2048

# "none"/"restricted"/"unrestricted"
electron_conservation_type: "restricted"
qubit_order: -1 # set qubit order
log_level: 'INFO'
log_step: 1 # log ouput per log_step iteration
system: 'LiH' # molecular name
n_elecs: 4 # number of electrons
elec_cons_method: 0

# decoder model
model:
    d_model: 64
    n_layers: 6
    n_heads: 4
    p_dropout: 0.0

use_phase_embedding: false

# phase model for QiankunNet2 (Optional)
phase_hidden_features: 256
phase_num_blocks: 1

# phase model for QiankunNet1 (Optional)
phase_hidden_size: [512,512,512,512]

# optimizer
optim:
    name: 'AdamW' # optimizer name
    lr: 1.0 # learning rate
    betas: [0.9, 0.99]
    eps: 1e-9
    weight_decay: 0.00
    open_lr_scheduler: true
    warmup_step: 4000
```
If your system alpha and beta electrons are not the same, you should specify them as follow:
```yaml
# Example: O2 molecule
n_alpha_electrons: 9 # spin-up elec
n_beta_electrons: 7 # spin-down elec
```
### GPU Support
We support pytorch model infer and backward in GPU and local energy calculation speedup using GPU independently. You can set keywords in the YAML configuration file (e.g. `config-h4.yaml`):

```shell
# for torch using GPU 
device: cuda # CUDA

# for local energy calculation using GPU
# and run `make clean && make gpu` to generate `libeloc.so` in local_energy
```

## Generate input data (Hamiltonian)

Speedup PySCF to generate input hamiltonian:
`pip install opt-einsum`
