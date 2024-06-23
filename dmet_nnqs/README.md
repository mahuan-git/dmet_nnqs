# NNQS solver for dmet algorithm.

## libdmet [https://github.com/gkclab/libdmet_preview](https://github.com/gkclab/libdmet_preview)
git clone the code and add the directory to PYTHONPATH

## NeuralNetworkQuantumState
follow the README.md file in the nnqs code. Here we offer a brief version.
### needed python lib:
- python >= 3.8
    - numpy==1.23.5
    - openfermion==1.5.1
    - pyscf==2.1.1
    - PyYAML==6.0
    - scipy==1.10.1
    - torch==1.13.1+cu116

### how to run nnqs
compile from source code:\
cpu version:\
```shell
make cpu
```
gpu version:\
first go to local_energy folder, and modify the Makefile based on your system. Modify CUDA_ROOT and CUDA_GENCODE may also be modified.\
Then make with
```shell
make gpu
```
In the given nnqs code, we do not offer source code of the GPU version local energy calculation. But a compile cpu lib is given. So in the config file should use following settings:\
device: 'cpu'\
local_energy_version: "CPP_CPU"\
Add path to PATHs:
```shell
NNQS_PATH=/path/to/NeuralNetworkQuantumState
export PYTHONPATH=$NNQS_PATH:$NNQS_PATH/local_energy/:$PYTHONPATH
export LD_LIBRARY_PATH=$NNQS_PATH/local_energy/:$LD_LIBRARY_PATH
```

### dmet_nnqs
As the libdmet package and the NeuralNetworkQuantumState package is installed, the dmet_nnqs code does not require other packages.\
The dmet_nnqs package can be installed simply by add path to PTHONPATH 
```shell
export PYTHONPATH=/path/t0/dmet_nnqs/$PYTHONPATH
```

### inportant settings
To define a QiankunNet solver for dmet_nnqs method can be done as following 
```Python
from dmet_nnqs.nnqs import NNQS
from dmet_nnqs.config import MyConfig
config = MyConfig("config.yaml")
solver = NNQS(config = config,mol_name = mol_name,frag_idx = I ,restricted=restricted,calc_rdm2=False)
```
where config.yaml is a yaml file which contain settings for QiankunNet solver.\
The config.yaml file should look like this:
```yaml
# torch device
#device: 'cpu'
device: 'cuda'
# local energy calculation device
#local_energy_version: "CPP_CPU"
local_energy_version: "CPP_GPU"
#local_energy_version: "JL_CPU"

#n_samples: 1000000000000
n_samples: 12288000
n_epoches: 200000
save_model: 0
save_per_epoches: 5000
load_model: 0
checkpoint_path: "checkpoints/li2o-nomask-iter1000-rank0.pt"

# 0->extract ham; 1->load ham; 2->save ham;
do_ham: 0

#model_fp: fp64
model_fp: fp32

transfer_learning: true
#transfer_learning: false
seed: 333
pos_independ: false
use_sr: false
use_clip_grad: false
#psi_type: 'real'
psi_type: 'complex'
only_infer: false
search_init_best_seed: false
weak_scaling: false

# Open parallel of software
#use_uniq_sampling_parallel: true
use_uniq_sampling_parallel: false
incr_per_epoch: 50
min_partition_samples: 64
#partition_algo: 'weight'
partition_algo: 'number'

# if > 0, using max input dimension for phase
max_transfer: 0
#max_transfer: 128

use_samples_recursive: true
n_samples_min: 1e8
n_samples_max: 1e12
n_unq_samples_min: 1e3
n_unq_samples_max: 1e6
n_samples_scale_factor: 1.3

sampling_algo: "bfs"
#sampling_algo: "dfs"
sampling_dfs_width: 128
sampling_dfs_uniq_samples_min: 2048

drop_samples: false
drop_samples_eps: 1e-8

use_amp_infer_batch: true
amp_infer_batch_size: 8192

use_kv_cache: false
use_grad_accumulation: false
grad_accumulation_width: 40960
#grad_accumulation_width: 8192

# "none"/"restricted"/"unrestricted"
electron_conservation_type: "restricted"
qubit_order: -1
#qubit_order: 0
log_level: 'INFO'
#log_level: 'DEBUG'
log_step: 10
system: SYSTEM
n_elecs: NELEC
# default will be n_elecs/2
n_alpha_electrons: NELEC_A # spin-up elec
n_beta_electrons: NELEC_B # spin-down elec
elec_cons_method: 0
std_dev_tol: 2e-6
result_filter_size: 100
model:
    d_model: 32
    n_layers: 4
    n_heads: 4
    p_dropout: 0.0

#use_phase_embedding: true
use_phase_embedding: false

optim:
    name: 'AdamW'
      #lr: 0.0005
    lr: 1.0
    betas: [0.9, 0.99]
    eps: 1e-9
    weight_decay: 0.001
    open_lr_scheduler: true
    #open_lr_scheduler: false
    warmup_step: 4000
    #warmup_step: 10000

```
Some inportant setting that may be crucial for QiankunNet convergence:
```yaml
load_model: 0   # 0 for not load model and train from scratch. 1 for load existing model.
checkpoint_path: "checkpoints/li2o-nomask-iter1000-rank0.pt"  # path of the model to load.
n_samples_min: 1e14  # minimum number of samples, if encounter a CUDA OUT of Memory error, can try make n_sample_min smaller.
n_samples_max: 1e16  # maximum number of samples.
log_step: 10   # every n steps record a QiankunNet energy.
```
```yaml
std_dev_tol: 2e-6
result_filter_size: 100
```
These two setting defines the convergence criteria of the QiankunNet solver. The QiankunNet solver will restore last N results in a list (N is controled by the result_filter_size tag), and calculate the standard deviation in the result list. If the standard deviation is smaller than a given tolerance (controled by std_dev_tol tag), the QiankunNet solver gives the result with lowest energy in the result list. Increase result_filter_size would increase stability of the QiankunNet solver while taking more time.
```yaml
model:
    d_model: 32
    n_layers: 4
    n_heads: 4
    p_dropout: 0.0
```
Defines the neural network used by QiankunNet solver. If you found the expressibility of the neural network is not enough, increase d_model and n_layers to increase expressibility.
```yaml
optim:
    name: 'AdamW'
      #lr: 1e-5
    lr: 1.0   # learning rate 
    betas: [0.9, 0.99]
    eps: 1e-9
    weight_decay: 0.001
    open_lr_scheduler: true   # If true, the learning rate would be scheduled, the learning rate will first increase, then decrease to a smaller value gradually.
    #open_lr_scheduler: false
    warmup_step: 4000
```
If train a Neural Network from scratch, set lr a large value and open_le_scheduler = true. If load a well trained model, set lr a small value.\
 \
Next we introduce how to define a transfer learning strategy:
```python


strong_cfg ={
            'load_model': 1,
            'log_step' : 1,
            'std_dev_tol': 2e-6,
            'result_filter_size' : 100,
            'optim': {'name': 'AdamW',
                'lr': 1e-5,
                'betas': [0.9, 0.99],
                'eps': '1e-9',
                'weight_decay': 0.0,
                'open_lr_scheduler': False,
                'warmup_step': 100}
weak_cfg ={
            'load_model': 1,
            'log_step' : 5,
            'std_dev_tol': 2e-6,
            'result_filter_size' : 100,
            'optim': {'name': 'AdamW',
                'lr': 1e-2,
                'betas': [0.9, 0.99],
                'eps': '1e-9',
                'weight_decay': 0.001,
                'open_lr_scheduler': True,
                'warmup_step': 100}
            }
```


