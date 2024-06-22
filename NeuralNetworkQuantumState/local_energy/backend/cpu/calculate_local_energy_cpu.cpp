#include "calculate_local_energy_cpu.h"
#include "hamiltonian/hamiltonian.h"
#include <typeinfo>

#define PRINT_TYPE(x) ((sizeof(x) == 4) ? "Float32" : "Float64")

// Global persistent data
static int32 g_n_qubits = -1;
static int64 g_K = -1;
static int64 g_NK = -1;
static int64 *g_idxs = NULL;
static float64 *g_coeffs = NULL;
static dtype *g_pauli_mat12 = NULL;
static dtype *g_pauli_mat23 = NULL;

// Float point absolute
#define FABS(x) (((x) < 0.) ? (-(x)) : (x))
#define MIN(x,y) ((x) < (y)) ? (x) : (y)

void _state2id_huge(const dtype *state, const int64 N, const int64 id_width, const int64 stride, const uint64 *tbl_pow2, uint64 *res_id) {
    memset(res_id, 0, sizeof(uint64) * id_width);
    int max_len = N / stride + (N % stride != 0);
    // int max_len = N / stride;
    // printf("c max_len: %d\n", max_len);

    for (int i = 0; i < max_len; i++) {
        int st = i*stride, ed = MIN((i+1)*stride, N);
        uint64 id = 0;
        for (int j = st, k=0; j < ed; j++, k++) {
            id += tbl_pow2[k] * state[j];
        }
        // printf("c id: %lu\n", id);
        res_id[i] = id;
    }
}

void _state2id_huge_fuse(const dtype *state_ii, const dtype *pauli_mat12, const int64 N, const int64 id_width, const int64 stride, const uint64 *tbl_pow2, uint64 *res_id) {
    memset(res_id, 0, sizeof(uint64) * id_width);
    int max_len = N / stride + (N % stride != 0);
    // int max_len = N / stride;
    // printf("c max_len: %d\n", max_len);
    for (int i = 0; i < max_len; i++) {
        int st = i*stride, ed = MIN((i+1)*stride, N);
        uint64 id = 0;
        for (int j = st, k=0; j < ed; j++, k++) {
            // id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            id += (state_ii[j] ^ pauli_mat12[j])*tbl_pow2[k];
            // id += tbl_pow2[k] * state[j];
        }
        // printf("c id: %lu\n", id);
        res_id[i] = id;
    }
}

int _compare_id(const uint64 *s1, const uint64 *s2, const int64 len) {
    for (int i = len-1; i >= 0; i--) {
        if (s1[i] > s2[i]) {
            return 1;
        } else if (s1[i] < s2[i]) {
            return -1;
        }
    }
    return 0;
}

// Copy Julia data into CPP avoid gc free memory
// ATTENTION: This must called at the first time
void set_indices_ham_int_opt(
    const int32 n_qubits,
    const int64 K,
    const int64 NK,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23)
{
    g_n_qubits = n_qubits;
    g_K = K;
    g_NK = NK;

    const size_t size_g_idxs = sizeof(int64) * (NK + 1);
    const size_t size_g_coeffs = sizeof(coeff_dtype) * K;
    const size_t size_g_pauli_mat12 = sizeof(dtype) * (n_qubits * NK);
    const size_t size_g_pauli_mat23 = sizeof(dtype) * (n_qubits * K);

    g_idxs = (int64 *)malloc(size_g_idxs);
    g_coeffs = (coeff_dtype *)malloc(size_g_coeffs);
    g_pauli_mat12 = (dtype *)malloc(size_g_pauli_mat12);
    g_pauli_mat23 = (dtype *)malloc(size_g_pauli_mat23);

    memcpy(g_idxs, idxs, size_g_idxs);
    memcpy(g_coeffs, coeffs, size_g_coeffs);
    memcpy(g_pauli_mat12, pauli_mat12, size_g_pauli_mat12);
    memcpy(g_pauli_mat23, pauli_mat23, size_g_pauli_mat23);
    printf("----set_indices_ham_int_opt in CPP_CPU psi_dtype: %s-----\n", PRINT_TYPE(psi_dtype));
}

/**
 * Calculate local energy by fusing Hxx' and summation.
 * Args:
 *     n_qubits: number of qubit
 *     idxs: index of pauli_mat23 block
 *     coeffs: pauli term coefficients
 *     pauli_mat12: extract info of pauli operator 1 and 2 only for new states calculation
 *     pauli_mat23: extract info of pauli operator 2 and 3 only for new coeffs calculation
 *     batch_size: samples number which ready to Hxx'
 *     state_batch: samples
 *     ks: map samples into id::Int Notion: n_qubits <= 64!
 *     vs: samples -> psis (ks -> vs)
 *     eps: dropout coeff < eps
 * Returns:
 *     res_eloc_batch: save the local energy result with complex value,
 *                     res_eloc_batch(1/2,:) represent real/imag
 * */
void calculate_local_energy_kernel(
    const int32 n_qubits,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const dtype *state_batch,
    const uint64 *ks,
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    const int64 NK = g_NK;
    const int64 K = g_K;

    // replace branch to calculate state -> id
    uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    // loop all samples
    for (int ii = 0; ii < batch_size_cur_rank; ii++) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        int64 i_base = 0;
        for (int sid = 0; sid < NK; sid++) {
            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                int _sum = 0;
                for (int ik = 0; ik < N; ik++) {
                    _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                }
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                i_base += N;
            }

            // filter value < eps
            if (FABS(coef) < eps) {
                continue;
            }

            // map state -> id
            int64 j_base = sid * N;
            uint64 id = 0;
            for (int ik = 0; ik < N; ik++) {
                id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            }

            #if 0
            // linear find id among the sampled samples
            for (int _ist = 0; _ist < batch_size; _ist++) {
                if (ks[_ist] == id) {
                    e_loc_real += coef * vs[_ist * 2];
                    e_loc_imag += coef * vs[_ist * 2 + 1];
                    break;
                }
            }
            #else
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size, _imd = 0;
            while (_ist < _ied) {
                _imd = (_ist + _ied) / 2;
                if (ks[_imd] == id) {
                    // e_loc += coef * vs[_imid]
                    e_loc_real += coef * vs[_imd * 2];
                    e_loc_imag += coef * vs[_imd * 2 + 1];
                    break;
                }

                if (ks[_imd] < id) {
                    _ist = _imd + 1;
                } else {
                    _ied = _imd;
                }
                // int res = ks[_imd] < id;
                // _ist = (res == 1) ? _imd + 1 : _ist;
                // _ied = (res == 1) ? _ied : _imd;
            }
            #endif
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        // res_eloc_batch[ii*2+1] = (a*d - b*c) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

// just for function validation
void calculate_local_energy_kernel_diff(
    const int32 n_qubits,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const dtype *state_batch,
    const float64 eps,
    int64 *res_states,
    coeff_dtype *res_coeffs,
    int *n_res)
{
    const int32 N = n_qubits;
    const int64 NK = g_NK;
    const int64 K = g_K;

    // replace branch to calculate state -> id
    uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    // loop all samples
    for (int ii = 0; ii < batch_size_cur_rank; ii++) {
        int res_cnt = 0;
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        int64 i_base = 0;
        for (int sid = 0; sid < NK; sid++) {
            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                int _sum = 0;
                for (int ik = 0; ik < N; ik++) {
                    _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                }
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                i_base += N;
            }

            // filter value < eps
            if (FABS(coef) < eps) {
                continue;
            }

            res_coeffs[res_cnt] = coef;
            // map state -> id
            int64 j_base = sid * N;
            uint64 id = 0;
            for (int ik = 0; ik < N; ik++) {
                // id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
                res_states[res_cnt*N+ik] = state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik];
            }
            res_cnt++;
        }
        *n_res = res_cnt;
    }
}


void calculate_local_energy_kernel_bigInt(
    const int32 n_qubits,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const dtype *state_batch,
    const uint64 *ks,
    const int64 id_width,
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    const int64 NK = g_NK;
    const int64 K = g_K;
    // printf("C: N: %d, NK: %d K: %d\n", N, NK, K);
    // replace branch to calculate state -> id
    uint64 tbl_pow2[id_stride];
    tbl_pow2[0] = 1;
    for (int i = 1; i < id_stride; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    // loop all samples
    #pragma omp parallel for
    for (int ii = 0; ii < batch_size_cur_rank; ii++) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        // int64 i_base = 0;
        uint64 big_id[id_width];
        for (int sid = 0; sid < NK; sid++) {
            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            // printf("i: %d st: %d ed: %d\n", sid, st, ed);
            for (int i = st; i < ed; i++) {
                int _sum = 0;
                for (int ik = 0; ik < N; ik++) {
                    // _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                    _sum += state_batch[ii*N+ik] & pauli_mat23[i*N+ik];
                }
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                // i_base += N;
            }

            // filter value < eps
            if (FABS(coef) < eps) {
                continue;
            }

            // map state -> id
            int64 j_base = sid * N;
            // int64 id = 0;
            // for (int ik = 0; ik < N; ik++) {
            //     id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            // }
            _state2id_huge_fuse(&state_batch[ii*N], &pauli_mat12[j_base], N, id_width, id_stride, tbl_pow2, big_id);

            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size, _imd = 0;
            while (_ist < _ied) {
                _imd = (_ist + _ied) / 2;
                int res = _compare_id(&ks[_imd*id_width], big_id, id_width);
                if (res == 0) {
                    // e_loc += coef * vs[_imid]
                    e_loc_real += coef * vs[_imd * 2];
                    e_loc_imag += coef * vs[_imd * 2 + 1];
                    break;
                }

                if (res == -1) {
                    _ist = _imd + 1;
                } else {
                    _ied = _imd;
                }
            }
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        // res_eloc_batch[ii*2+1] = (a*d - b*c) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

/**
 * Calculate local energy by fusing Hxx' and summation interface (for Julia).
 * Current rank only calculate _states[ist, ied) and ist start from 0
 * Args:
 *     batch_size: total samples number
 *     _state: samples
 *     k_idxs: ordered samples origin index used for permutation
 *     ks: map samples into id::Int Notion: n_qubits <= 64!
 *     vs: samples -> psis (ks -> vs)
 *     rank: MPI rank
 *     eps: dropout coeff < eps
 * Returns:
 *     res_eloc_batch: save the local energy result with complex value,
 *                     res_eloc_batch(1/2,:) represent real/imag
 **/
void calculate_local_energy(
    const int64 batch_size,
    const int64 *_states,
    const int64 ist,
    const int64 ied,
    const int64 *k_idxs, // unused now
    const uint64 *ks,
    const psi_dtype *vs,
    const int64 rank,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    // transform _states{int64} into states{dtype} and map {+1,-1} to {+1,0}
    // assume states id is ordered after unique sampling, for using binary find
    //const int64 target_value = -1;
    dtype *states = (dtype *)malloc(sizeof(dtype) * batch_size * N);
    memset(states, 0, sizeof(dtype) * batch_size * N); // init 0
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            //if (_states[i*N+j] != target_value) {
            if (_states[i*N+j] == 1) {
                states[i*N+j] = 1;
            }
        }
    }

    calculate_local_energy_kernel(
        n_qubits,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        batch_size,
        batch_size_cur_rank,
        ist,
        &states[ist*N],
        ks,
        vs,
        eps,
        res_eloc_batch);

    free(states);
}

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
    int *n_res)
{
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    // transform _states{int64} into states{dtype} and map {+1,-1} to {+1,0}
    // assume states id is ordered after unique sampling, for using binary find
    //const int64 target_value = -1;
    dtype *states = (dtype *)malloc(sizeof(dtype) * batch_size * N);
    memset(states, 0, sizeof(dtype) * batch_size * N); // init 0
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            //if (_states[i*N+j] != target_value) {
            if (_states[i*N+j] == 1) {
                states[i*N+j] = 1;
            }
        }
    }

    calculate_local_energy_kernel_diff(
        n_qubits,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        batch_size,
        batch_size_cur_rank,
        ist,
        &states[ist*N],
        eps,
        res_states,
        res_coeffs,
        n_res);

    free(states);
}

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
    psi_dtype *res_eloc_batch)
{
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    // transform _states{int64} into states{dtype} and map {+1,-1} to {+1,0}
    // assume states id is ordered after unique sampling, for using binary find
    //const int64 target_value = -1;
    dtype *states = (dtype *)malloc(sizeof(dtype) * batch_size * N);
    memset(states, 0, sizeof(dtype) * batch_size * N); // init 0
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            //if (_states[i*N+j] != target_value) {
            if (_states[i*N+j] == 1) {
                states[i*N+j] = 1;
            }
        }
    }

    calculate_local_energy_kernel(
        n_qubits,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        all_batch_size,
        batch_size_cur_rank,
        ks_disp_idx,
        &states[ist*N],
        ks,
        vs,
        eps,
        res_eloc_batch);

    free(states);
}

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
    psi_dtype *res_eloc_batch)
{
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    // transform _states{int64} into states{dtype} and map {+1,-1} to {+1,0}
    // assume states id is ordered after unique sampling, for using binary find
    //const int64 target_value = -1;
    dtype *states = (dtype *)malloc(sizeof(dtype) * batch_size * N);
    memset(states, 0, sizeof(dtype) * batch_size * N); // init 0
    // printf("batch_size: %d all_batch_size: %d, N: %d ist: %d\n", batch_size, all_batch_size, N, ist);

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            //if (_states[i*N+j] != target_value) {
            if (_states[i*N+j] == 1) {
                states[i*N+j] = 1;
            }
        }
    }

    // printf("states addr: %p, %p\n", states, &states[ist*N]);
    // for (int i = 0; i < batch_size; i++) {
    //     printf("i: %d ", i);
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", states[i*N+j]);
    //     }
    //     puts("\n");
    // }

    calculate_local_energy_kernel_bigInt(
        n_qubits,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        all_batch_size,
        batch_size_cur_rank,
        ks_disp_idx,
        &states[ist*N],
        ks,
        id_width,
        vs,
        eps,
        res_eloc_batch);

    free(states);
}

int32_t init_hamiltonian(std::string ham_file) {
    if (g_n_qubits == -1) {
        Ham::get_preprocess_ham(ham_file);
    }
    return g_n_qubits;
}

int32_t init_hamiltonian(char *ham_file) {
    std::string ham_file_str = std::string(ham_file);
    return init_hamiltonian(ham_file_str);
}

void free_hamiltonian() {
    if (g_n_qubits != -1) {
        g_n_qubits = -1;
        g_NK = -1;

        free(g_idxs);
        free(g_coeffs);
        free(g_pauli_mat12);
        free(g_pauli_mat23);
    }
}

