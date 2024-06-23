LIBNAME=libeloc.so

# configure your NVCC and CXX path
#VERSION=11.8
#CUDA_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/${VERSION} # HPC_SDK
CUDA_ROOT=/opt/cuda/11.7.1_515.65.01/
CXX=/usr/bin/g++ # C++ compiler
NVCC=${CUDA_ROOT}/bin/nvcc
LIB_DIR=${CUDA_ROOT}/lib64/
# support Nvidia A100 and H100
CUDA_GENCODE=-gencode arch=compute_80,code=sm_80
NVCCOPTIONS=--std=c++17 --expt-relaxed-constexpr --expt-extended-lambda ${CUDA_GENCODE} -g
CXXOPTIONS=--std=c++17
#RELEASED=-D BITARR_HASH_OPT -D RELEASED # for training
DFLAGS=-DBITARR_HASH_OPT
RELEASED=-DRELEASED # for uint test
DEBUG_OPTIONS=-DDEBUG_LOCAL_ENERGY
DEBUG_OPTIONS=
PY_ENV=__PSI_DTYPE=PSI_DTYPE_FP32
#INCLUDES=-I./
INCLUDES=-I./ -I./include/

# Source and object files
CPU_SRC=backend/cpu/calculate_local_energy_cpu.cpp
GPU_SRC=backend/cuda/calculate_local_energy.cu
TEST_SRC=test/test_calculate_local_energy.cpp
#TEST_CPU_SRC=calculate_local_energy_cpu.cpp test_calculate_local_energy.cpp

# Object files
#TEST_GPU_OBJECTS=$(GPU_SRC:.cu=.o)
TEST_GPU_OBJECTS+=$(TEST_SRC:.cpp=.o)

#TEST_CPU_OBJECTS=$(CPU_SRC:.cpp=.o)
TEST_CPU_OBJECTS+=$(TEST_SRC:.cpp=.o)

# Test targets
TARGET=eloc_test
TEST_TARGETS_CPU=test-cpu
TEST_TARGETS_GPU=test test-bitarr test-bithash

.PHONY: all cpu gpu $(TEST_TARGETS) clean

all: gpu

# Test
test: DFLAGS=-DBACKEND_GPU
test-bitarr: DFLAGS=-DBITARR_OPT -DBACKEND_GPU
test-bithash: DFLAGS=-DBITARR_HASH_OPT -DBACKEND_GPU
test-cpu: DFLAGS=-DBACKEND_CPU

check_pybind11:
	python -c "import pybind11" 2>/dev/null || (echo "Installing pybind11..." && pip install pybind11)

# Dynamic library rules
cpu: $(CPU_SRC) check_pybind11
	$(CXX) ${INCLUDES} $(CXXOPTIONS) ${DFLAGS} -fPIC -shared -O3 ${CPU_SRC} -o $(LIBNAME)
	__BACKEND=BACKEND_CPU ${PY_ENV} python interface/python/setup.py build_ext --inplace

gpu: $(GPU_SRC) check_pybind11
	$(NVCC) ${INCLUDES} $(NVCCOPTIONS) ${DFLAGS} ${RELEASED} $(DEBUG_OPTIONS) $(INCLUDES) -Xcompiler "-fPIC -shared" ${GPU_SRC} -O3 -lineinfo -o $(LIBNAME) -L$(LIB_DIR)
	__BACKEND=BACKEND_GPU ${PY_ENV} python interface/python/setup.py build_ext --inplace

# fp64 version
gpu_fp64 cpu_fp64: DFLAGS+=-DPSI_DTYPE_FP64
gpu_fp64 cpu_fp64: PY_ENV=__PSI_DTYPE=PSI_DTYPE_FP64
cpu_fp64: cpu
gpu_fp64: gpu

# Test rules

$(TEST_TARGETS_CPU): cpu $(TEST_CPU_OBJECTS)
	$(CXX) ${INCLUDES} $(CXXOPTIONS) $(DFLAGS) $(TEST_CPU_OBJECTS) -L. -leloc -o $(TARGET)

$(TEST_TARGETS_GPU): gpu $(TEST_GPU_OBJECTS)
	$(NVCC) ${INCLUDES} $(NVCCOPTIONS) $(DFLAGS) $(TEST_GPU_OBJECTS) -L. -leloc -o $(TARGET)

%.o: %.cu
	$(NVCC) ${INCLUDES} $(NVCCOPTIONS) $(DFLAGS) $(DEBUG_OPTIONS) -c $< -o $@

%.o: %.cpp
	$(CXX) ${INCLUDES} $(CXXOPTIONS) $(DFLAGS) $(DEBUG_OPTIONS) -c $< -o $@

# test interface
test_python_interface:
	python test/test_eloc.py

test_julia_interface:
	julia test/test_eloc.jl

clean:
	find . -name "*.o" -exec rm {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f *.so log $(TARGET) build/ __pycache__ -rf
