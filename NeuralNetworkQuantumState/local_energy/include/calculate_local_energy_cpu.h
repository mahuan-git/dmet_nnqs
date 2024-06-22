#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

//#define BACKEND_CPU // compile tag indicate CPU backend

// alias data type
typedef int8_t int8;
typedef int32_t int32;
typedef int64_t int64;
typedef uint64_t uint64;
typedef float float32;
typedef double float64;

// Precision control
typedef int8 dtype;           // hamitonian indices / states
typedef float64 coeff_dtype;  // pauli term coeff
const int32 MAX_NQUBITS = 64; // max qubits
const int64 id_stride = 64;      // for BigInt id, 64 qubit using uint64 represent

#ifdef PSI_DTYPE_FP64
typedef float64 psi_dtype;
#else
typedef float32 psi_dtype;
#endif

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
        const dtype *pauli_mat12,
        const dtype *pauli_mat23);

    void calculate_local_energy(
        const int64 batch_size,
        const int64 *_states,
        const int64 ist, // assume [ist, ied) and ist start from 0
        const int64 ied,
        const int64 *k_idxs,
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

    // just for function validation
    void calculate_local_energy_diff(
        const int64 batch_size,
        const int64 *_states,
        const int64 ist,
        const int64 ied,
        const int64 rank,
        const float64 eps,
        int64 *res_states,
        coeff_dtype *res_coeffs,
        int *n_res);
}

