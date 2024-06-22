#ifndef __CALCULATE_LOCAL_ENERGY_H__
#define __CALCULATE_LOCAL_ENERGY_H__

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <utility>

//#define BACKEND_CUDA

//#ifdef BITARR_HASH_OPT
//#include "hashTable.cuh"
//#endif

//#define DEBUG_LOCAL_ENERGY
#include "utils/timer.h"

// alias data type
typedef int8_t int8;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;
typedef float float32;
typedef double float64;

// Precision control
#if defined(BITARR_OPT) || defined(BITARR_HASH_OPT)
typedef uint32_t dtype;           // hamitonian indices / states
#else
typedef int8 dtype;           // hamitonian indices / states
#endif
typedef float64 coeff_dtype;  // pauli term coeff
const int32 MAX_NQUBITS = 64; // max qubits
const int64 id_stride = 64;   // for BigInt id, 64 qubit using uint64 represent

#ifdef PSI_DTYPE_FP64
typedef float64 psi_dtype;
#else
typedef float32 psi_dtype;
#endif

//#if !defined(BITARR_HASH_OPT)
//// typedef float64 psi_dtype;
//typedef float32 psi_dtype;
//#endif

#define ID_UNIT_SIZE 64 // uint64

// Float point absolute
#define FABS(x) (((x) < 0.) ? (-(x)) : (x))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// Export C interface for julia
extern "C" {
    //int32 init_hamiltonian(std::string ham_file);
    int32 init_hamiltonian(char *ham_file);
    void free_hamiltonian();
    void set_indices_ham_int_opt(
        const int32 n_qubits,
        const int64 K,
        const int64 NK,
        const int64 *idxs,
        const coeff_dtype *coeffs,
        //#ifdef BITARR_OPT
        #if defined(BITARR_OPT) || defined(BITARR_HASH_OPT)
        const int8 *pauli_mat12,
        const int8 *pauli_mat23);
        #else
        const dtype *pauli_mat12,
        const dtype *pauli_mat23);
        #endif

    void calculate_local_energy(
        const int64 batch_size,
        const int64 *_states,
        const int64 ist, // assume [ist, ied) and ist start from 0
        const int64 ied,
        const int64 *k_idxs,
        // const int64 *ks,
        const uint64 *ks,
        const psi_dtype *vs,
        const int64 rank,
        const float64 eps,
        psi_dtype *res_eloc_batch);

    void calculate_local_energy_sampling_parallel(
        const int64 all_batch_size,
        const int64 batch_size,
        const int64 *_states,
        const int64 ist,
        const int64 ied,
        const int64 ks_disp_idx,
        const uint64 *ks,
        const psi_dtype *vs,
        const int64 rank,
        const float64 eps,
        psi_dtype *res_eloc_batch);

    void calculate_local_energy_sampling_parallel_bigInt(
        const int64 all_batch_size,
        const int64 batch_size,
        const int64 *_states,
        const int64 ist,
        const int64 ied,
        const int64 ks_disp_idx,
        const uint64 *ks,
        const int64 id_width,
        const psi_dtype *vs,
        const int64 rank,
        const float64 eps,
        psi_dtype *res_eloc_batch);

    void set_gpu_to_mpi(int rank);
    void get_gpu_id(int rank, int print_verbose);
}
#endif // __CALCULATE_LOCAL_ENERGY_H__
