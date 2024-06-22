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
first go to local_energy folder, and modify the Makefile based on your system. Modify CUDA_ROOT and CUDA_GENCODE may also be modified.\
Then make with
```shell
make gpu
```
and add path to PATHs
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
