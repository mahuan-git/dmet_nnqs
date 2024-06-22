#include "calculate_local_energy.cuh"
#include "hamiltonian/hamiltonian.h"
#ifdef BITARR_HASH_OPT
#include "hashTable.cuh"
#endif    
#include <cassert>
#include <string>
        
#ifdef NPY_SAVE_DATA
    std::string molecule_name = "lih";
    std::string ham_file = molecule_name + ".ham";
    std::string indata_file = molecule_name + ".indata";                                                                                                                                                              
    #include "utils/numpy.h"
#endif

#define PRINT_TYPE(x) ((sizeof(x) == 4) ? "Float32" : "Float64")

// Global persistent data
static int32 g_n_qubits = -1;
static int64 g_NK = -1;
static int64 *g_idxs = NULL;
// static float64 *g_coeffs = NULL;
static coeff_dtype *g_coeffs = NULL;
static dtype *g_pauli_mat12 = NULL;
static dtype *g_pauli_mat23 = NULL;

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
        res_id[i] = id;
    }
}

__device__ void _state2id_huge_fuse(const dtype *state_ii, const dtype *pauli_mat12, const int64 N, const int64 id_width, const int64 stride, const uint64 *tbl_pow2, uint64 *res_id) {
    memset(res_id, 0, sizeof(uint64) * id_width);
    int max_len = N / stride + (N % stride != 0);
    // int max_len = N / stride;
    // printf("c max_len: %d\n", max_len);
    for (int i = 0; i < max_len; i++) {
        int st = i*stride, ed = MIN((i+1)*stride, N);
        uint64 id = 0;
        for (int j = st, k=0; j < ed; j++, k++) {
            id += (state_ii[j] ^ pauli_mat12[j])*tbl_pow2[k];
            // id += tbl_pow2[k] * state[j];
        }
        res_id[i] = id;
    }
}

template<int len=1>
//__device__ int _compare_id(const uint64 *s1, const uint64 *s2, const int64 len) {
__device__ int _compare_id(const uint64 *s1, const uint64 *s2) {
#pragma unroll
    for (int i = len-1; i >= 0; i--) {
        if (s1[i] > s2[i]) {
            return 1;
        } else if (s1[i] < s2[i]) {
            return -1;
        }
    }
    return 0;
}

// binary find id among the sampled samples
// idx = binary_find(ks, big_id), [ist, ied) start from 0
// ret_res = 0: find the big_id, and save result in psi_real and psi_imag
template<int idWidth=1>
//__device__ void binary_find_bigInt(const int32 ist, const int32 ied, const uint64 *ks, const psi_dtype *vs, int64 id_width, uint64 *big_id, psi_dtype *psi_real, psi_dtype *psi_imag, int32 *ret_res) {
__device__ void binary_find_bigInt(const int32 ist, const int32 ied, const uint64 *ks, const psi_dtype *vs, uint64 *big_id, psi_dtype *psi_real, psi_dtype *psi_imag, int32 *ret_res) {
    int32 _ist = ist, _ied = ied;
    int32 _imd = 0, res = 0xffff;
    while (_ist < _ied) {
        _imd = (_ist + _ied) / 2;
        //res = _compare_id(&ks[_imd*id_width], big_id, id_width);
        res = _compare_id<idWidth>(&ks[_imd*idWidth], big_id);
        if (res == 0) {
            // e_loc += coef * vs[_imid]
            *psi_real = vs[_imd * 2];
            *psi_imag = vs[_imd * 2 + 1];
            break;
        }

        if (res == -1) {
            _ist = _imd + 1;
        } else {
            _ied = _imd;
        }
    }
    *ret_res = res;
}

// one MPI <-> one GPU
void set_gpu_to_mpi(int rank) {
    int gpu_cnts = 0;
    cudaGetDeviceCount(&gpu_cnts);
    // cudaGetDevice(&cnt);
    int local_gpu_id = rank % gpu_cnts;
    cudaSetDevice(local_gpu_id);
}

// get bind GPU id of rank
void get_gpu_id(int rank, int print_verbose) {
    int device_id = -1;
    cudaGetDevice(&device_id);
    char pciBusId[256] = {0};
    cudaDeviceGetPCIBusId(pciBusId, 255, device_id);
    if (print_verbose == 1) {
        printf("rank %d bind into local gpu: %d (%s)\n", rank, device_id, pciBusId);
    }
}

#if defined(BITARR_OPT) || defined(BITARR_HASH_OPT)
// convert T_ARR type arr into bit array: arr[len] -> bit_arr[num_uint32]
// assume bit_arr is init zeros
template<typename T_ARR>
void convert2bitarray(const T_ARR *arr, int len, const int num_uint32, uint32_t *bit_arr) {
    for(int j = 0; j < num_uint32; ++j) {
        for(size_t i = j*32; i < len && i < (j + 1)*32; ++i) {
            // map 0/-1 -> 0; 1 -> 1
            if (arr[i] == 1) bit_arr[j] |= (arr[i] << (i - j*32));
        }
    }
}

// convert T_ARR type arr into bit array; arr[nrow][ncol] -> bit_arr[nrow][num_uint32]
template<typename T_ARR>
std::pair<int, uint32_t*> convert2bitarray_batch(const T_ARR *arr, int nrow, int ncol) {
    const int num_uint32 = std::ceil(ncol / 32.0);
    const int len_bit_arr = num_uint32 * nrow;
    uint32_t *bit_arr = (uint32_t*)malloc(sizeof(uint32_t) * len_bit_arr);
    memset(bit_arr, 0, sizeof(uint32_t) * len_bit_arr); // init zeros
    for (int i = 0; i < nrow; i++) {
        convert2bitarray(&arr[i*ncol], ncol, num_uint32, &bit_arr[i*num_uint32]);
    }
    return std::make_pair(num_uint32, bit_arr);
}

void print_bit(const uint32_t *bitRepresentation, int len) {
    for(int l = 0; l < len; ++l) {
        auto part = bitRepresentation[l];
        for(int i = 0; i < 32; ++i) {
            printf("%d%s", (part & (1 << i) ? 1 : 0), (i == 31) ? "\n" : ", ");
        }
    }
}

template<typename T>
void print_mat(const T *arr, int nrow, int ncol, std::string mess="") {
    if (mess != "") {
        printf("==%s==\n", mess.c_str());
    }

    for(int i = 0; i < nrow; ++i) {
        for(int j = 0; j < ncol; ++j) {
            printf("%d%s", (arr[i*ncol+j] == 1 ? 1 : 0), (j == ncol-1) ? "\n" : ", ");
        }
    }
}

template<typename T>
void print_mat_bit(const T *arr, int nrow, int ncol, std::string mess="") {
    if (mess != "") {
        printf("==%s==\n", mess.c_str());
    }

    for(int i = 0; i < nrow; ++i) {
        print_bit(&arr[i*ncol], ncol);
    }
}

// Copy Julia data into CPP avoid gc free memory
// ATTENTION: This must called at the first time
void set_indices_ham_int_opt(
    const int32 n_qubits,
    const int64 K,
    const int64 NK,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const int8 *pauli_mat12,
    const int8 *pauli_mat23)
    // const dtype *pauli_mat12,
    // const dtype *pauli_mat23)
{
    g_n_qubits = n_qubits;
    // g_K = K;
    g_NK = NK;

    const size_t size_g_idxs = sizeof(int64) * (NK + 1);
    const size_t size_g_coeffs = sizeof(coeff_dtype) * K;

    // print_mat(pauli_mat12, NK, g_n_qubits, "puali_mat12");

    auto ret1 = convert2bitarray_batch(pauli_mat12, NK, g_n_qubits);
    auto ret2 = convert2bitarray_batch(pauli_mat23, K, g_n_qubits);
    auto num_uint32 = ret1.first;
    auto pauli_mat12_bitarr = ret1.second;
    auto pauli_mat23_bitarr = ret2.second;

    // print_mat_bit(pauli_mat12_bitarr, NK, num_uint32, "puali_mat12_bitarr");
    // print_mat(pauli_mat23, K, g_n_qubits, "puali_mat23");
    // print_mat_bit(pauli_mat23_bitarr, K, num_uint32, "puali_mat23_bitarr");

    const size_t size_g_pauli_mat12 = sizeof(dtype) * (num_uint32 * NK);
    const size_t size_g_pauli_mat23 = sizeof(dtype) * (num_uint32 * K);

    cudaMalloc(&g_idxs, size_g_idxs);
    cudaMalloc(&g_coeffs, size_g_coeffs);
    cudaMalloc(&g_pauli_mat12, size_g_pauli_mat12);
    cudaMalloc(&g_pauli_mat23, size_g_pauli_mat23);
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(g_idxs, idxs, size_g_idxs, cudaMemcpyHostToDevice);
    cudaMemcpy(g_coeffs, coeffs, size_g_coeffs, cudaMemcpyHostToDevice);
    cudaMemcpy(g_pauli_mat12, pauli_mat12_bitarr, size_g_pauli_mat12, cudaMemcpyHostToDevice);
    cudaMemcpy(g_pauli_mat23, pauli_mat23_bitarr, size_g_pauli_mat23, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    float32 real_size = (size_g_idxs + size_g_coeffs + size_g_pauli_mat12 + size_g_pauli_mat23) / 1024.0 / 1024.0;
    printf("[libeloc] set_indices_ham_int_opt BitOpt in CPP_GPU with psi_dtype: %s memory occupied: %.4fMB\n", PRINT_TYPE(psi_dtype), real_size);
    fflush(stdout);
}
#else
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
    // g_K = K;
    g_NK = NK;

    const size_t size_g_idxs = sizeof(int64) * (NK + 1);
    const size_t size_g_coeffs = sizeof(coeff_dtype) * K;
    const size_t size_g_pauli_mat12 = sizeof(dtype) * (n_qubits * NK);
    const size_t size_g_pauli_mat23 = sizeof(dtype) * (n_qubits * K);

#ifdef NPY_SAVE_DATA
    npy_cpp::NumpyCpp np;
    np.insert("n_qubits", &n_qubits, sizeof(n_qubits));
    np.insert("K", &K, sizeof(K));
    np.insert("NK", &NK, sizeof(NK));
    np.insert("idxs", idxs, size_g_idxs);
    np.insert("coeffs", coeffs, size_g_coeffs);
    np.insert("pauli_mat12", pauli_mat12, size_g_pauli_mat12);
    np.insert("pauli_mat23", pauli_mat23, size_g_pauli_mat23);
    np.savez(ham_file);

    np.loadz(ham_file);
    npy_cpp::check(np["n_qubits"], &n_qubits, sizeof(n_qubits), "n_qubits");
    npy_cpp::check(np["K"], &K, sizeof(K), "K");
    npy_cpp::check(np["NK"], &NK, sizeof(NK), "NK");
    npy_cpp::check(np["idxs"], idxs, size_g_idxs, "idxs");
    npy_cpp::check(np["coeffs"], coeffs, size_g_coeffs, "coeffs");
    npy_cpp::check(np["pauli_mat12"], pauli_mat12, size_g_pauli_mat12, "pauli_mat12");
    npy_cpp::check(np["pauli_mat23"], pauli_mat23, size_g_pauli_mat23, "pauli_mat23");
#endif

    cudaMalloc(&g_idxs, size_g_idxs);
    cudaMalloc(&g_coeffs, size_g_coeffs);
    cudaMalloc(&g_pauli_mat12, size_g_pauli_mat12);
    cudaMalloc(&g_pauli_mat23, size_g_pauli_mat23);
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(g_idxs, idxs, size_g_idxs, cudaMemcpyHostToDevice);
    cudaMemcpy(g_coeffs, coeffs, size_g_coeffs, cudaMemcpyHostToDevice);
    cudaMemcpy(g_pauli_mat12, pauli_mat12, size_g_pauli_mat12, cudaMemcpyHostToDevice);
    cudaMemcpy(g_pauli_mat23, pauli_mat23, size_g_pauli_mat23, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    float32 real_size = (size_g_idxs + size_g_coeffs + size_g_pauli_mat12 + size_g_pauli_mat23) / 1024.0 / 1024.0;
    printf("[libeloc] set_indices_ham_int_opt in CPP_GPU with psi_dtype: %s memory occupied: %.4fMB\n", PRINT_TYPE(psi_dtype), real_size);
    fflush(stdout);
}
#endif

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
__global__ void calculate_local_energy_kernel(
    const int32 n_qubits,
    const int64 NK,
    const int64 *idxs,
    const coeff_dtype *coeffs,
    const dtype *pauli_mat12,
    const dtype *pauli_mat23,
    const int64 batch_size,
    const int64 batch_size_cur_rank,
    const int64 ist,
    const dtype *state_batch,
    const uint64_t *ks,
    const psi_dtype *vs,
    const float64 eps,
    psi_dtype *res_eloc_batch)
{
    const int32 N = n_qubits;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    // replace branch to calculate state -> id
    __shared__ uint64 tbl_pow2[MAX_NQUBITS];
    tbl_pow2[0] = 1;
    for (int i = 1; i < N; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }
    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
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
                // if (ii == 0 && index==0) printf("st:%d ed:%d; i: %d _sum: %d\n", st, ed, i, _sum);
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

template<int MAXN=32>
__global__ void calculate_local_energy_bitarr_kernel(
    const int32 num_uint32,
    const int64 NK,
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
    const int32 N = num_uint32;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        int64 i_base = 0;
        for (int sid = 0; sid < NK; sid++) {
            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                // int _sum = 0;
                // for (int ik = 0; ik < N; ik++) {
                //     _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                // }
                int _sum = __popc(state_batch[ii*N] & pauli_mat23[i_base]);
                if (MAXN == 64) {
                    _sum += __popc(state_batch[ii*N+1] & pauli_mat23[i_base+1]);
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
            // uint64 id = 0;
            // for (int ik = 0; ik < N; ik++) {
            //     id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            // }
            uint64 id = state_batch[ii*N] ^ pauli_mat12[sid*N];
            if (MAXN == 64) {
                id = ((uint64)(state_batch[ii*N+1] ^ pauli_mat12[sid*N+1]) << 32) | id;
            }
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
            }
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

template<typename T>
__device__ T warp_reduce_sum(T value) {
    #pragma unroll 5
    for (int j = 16; j >= 1; j /= 2)
        value += __shfl_xor_sync(0xffffffff, value, j, 32);
    return value;
}

template<int MAXN=32>
__global__ void calculate_local_energy_kernel_bigInt(
    const int32 n_qubits,
    const int64 NK,
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
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    // replace branch to calculate state -> id
    __shared__ uint64 tbl_pow2[id_stride];
    tbl_pow2[0] = 1;
    for (int i = 1; i < id_stride; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    // TODO
    // uint64 big_id[id_width];
    const int ID_WIDTH = (MAXN + ID_UNIT_SIZE - 1) / ID_UNIT_SIZE;
    uint64 big_id[ID_WIDTH];

    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
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
            // printf("ii: %d coef: %f\n", ii, coef);
            // map state -> id
            int64 j_base = sid * N;
            _state2id_huge_fuse(&state_batch[ii*N], &pauli_mat12[j_base], N, id_width, id_stride, tbl_pow2, big_id);

            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size, res = 0xffff;
            psi_dtype psi_real = 0., psi_imag = 0.;
            //binary_find_bigInt(_ist, _ied, ks, vs, id_width, big_id, &psi_real, &psi_imag, &res);
            binary_find_bigInt<ID_WIDTH>(_ist, _ied, ks, vs, big_id, &psi_real, &psi_imag, &res);
            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

// find coupled state first.
// if don't exist we just calculate next coupled state and drop the coef calculation
template<int MAXN=32>
__global__ void calculate_local_energy_kernel_bigInt_V1(
    const int32 n_qubits,
    const int64 NK,
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
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    // replace branch to calculate state -> id
    __shared__ uint64 tbl_pow2[id_stride];
    tbl_pow2[0] = 1;
    for (int i = 1; i < id_stride; i++) {
        tbl_pow2[i] = tbl_pow2[i-1] * 2;
    }

    // TODO
    const int ID_WIDTH = (MAXN + ID_UNIT_SIZE - 1) / ID_UNIT_SIZE;
    uint64 big_id[ID_WIDTH];

    // loop all samples
    // for (int ii = 0; ii < batch_size_cur_rank; ii++) {
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        // int64 i_base = 0;
        for (int sid = 0; sid < NK; sid++) {
            psi_dtype psi_real = 0., psi_imag = 0.;
            // map state -> id
            int64 j_base = sid * N;
            int res = 0xffff;
            // int64 id = 0;
            // for (int ik = 0; ik < N; ik++) {
            //     id += (state_batch[ii*N+ik] ^ pauli_mat12[j_base+ik])*tbl_pow2[ik];
            // }
            _state2id_huge_fuse(&state_batch[ii*N], &pauli_mat12[j_base], N, id_width, id_stride, tbl_pow2, big_id);
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size;
            //binary_find_bigInt(_ist, _ied, ks, vs, id_width, big_id, &psi_real, &psi_imag, &res);
            binary_find_bigInt<ID_WIDTH>(_ist, _ied, ks, vs, big_id, &psi_real, &psi_imag, &res);
            //printf("index: %d big_id[0]: %llu res: %d\n", index, big_id[0], res);

            // don't find this coupled state in current samples
            if (res != 0) {
                continue;
            }

            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                int _sum = 0;
                for (int ik = 0; ik < N; ik++) {
                    // _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                    _sum += state_batch[ii*N+ik] & pauli_mat23[i*N+ik];
                }
                //if (ii == 0 && index==0) printf("st:%d ed:%d; i: %d _sum: %d\n", st, ed, i, _sum);
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                // i_base += N;
            }

            //if (FABS(coef) < eps) continue;

            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

// find coupled state first.
// if don't exist we just calculate next coupled state and drop the coef calculation
template<int MAXN=64>
__global__ void calculate_local_energy_kernel_bigInt_V1_bitarr(
    const int32 num_uint32,
    const int64 NK,
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
    const int32 N = num_uint32;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    //uint64 big_id[ID_WIDTH];
    const int ID_WIDTH = (MAXN + ID_UNIT_SIZE - 1) / ID_UNIT_SIZE;
    uint64 big_id[ID_WIDTH];

    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        for (int sid = 0; sid < NK; sid++) {
            psi_dtype psi_real = 0., psi_imag = 0.;
            // map state -> id
            // int64 j_base = sid * N;
            int res = 0xffff;
            #pragma unroll
            for (int kk = 0; kk < MAXN/32; kk++) {
                if(kk % 2 == 0) {
                    big_id[kk/2] = state_batch[ii*N+kk] ^ pauli_mat12[sid*N+kk];
                    // if (ii == 0 && sid == 0) printf("kk%2=0 big_id[0]: %llu\n", big_id[kk/2]);

                } else {
                    big_id[kk/2] = ((uint64)(state_batch[ii*N+kk] ^ pauli_mat12[sid*N+kk]) << 32) | big_id[kk/2];
                    // if (ii == 0 && sid == 0) printf("kk%2 big_id[0]: %llu\n", big_id[kk/2]);
                }
            }
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size;
            binary_find_bigInt<ID_WIDTH>(_ist, _ied, ks, vs, big_id, &psi_real, &psi_imag, &res);
            // if (ii == 0 && sid == 0) printf("index: %d big_id[0]: %llu res: %d\n", index, big_id[0], res);

            // don't find this coupled state in current samples
            if (res != 0) {
                continue;
            }

            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                // int _sum = 0;
                // for (int ik = 0; ik < N; ik++) {
                //     // _sum += state_batch[ii*N+ik] & pauli_mat23[i_base+ik];
                //     _sum += state_batch[ii*N+ik] & pauli_mat23[i*N+ik];
                // }
                int _sum = 0;
                #pragma unroll
                for (int kk = 0; kk < MAXN/32; kk++) {
                    _sum += __popc(state_batch[ii*N+kk] & pauli_mat23[i*N+kk]);
                }
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
                // i_base += N;
            }
            // if (FABS(coef) < eps) continue;
            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;
        }

        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }
}

template<int MAXN=64>
__global__ void calculate_local_energy_kernel_bigInt_bitarr(
    const int32 num_uint32,
    const int64 NK,
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
    const int32 N = num_uint32;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    const int ID_WIDTH = (MAXN + ID_UNIT_SIZE - 1) / ID_UNIT_SIZE;
    uint64 big_id[ID_WIDTH];

    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        for (int sid = 0; sid < NK; sid++) {
            psi_dtype psi_real = 0., psi_imag = 0.;

            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                // for (int ik = 0; ik < N; ik++) {
                //     _sum += state_batch[ii*N+ik] & pauli_mat23[i*N+ik];
                // }
                int _sum = 0;
                #pragma unroll
                for (int kk = 0; kk < MAXN/32; kk++) {
                    _sum += __popc(state_batch[ii*N+kk] & pauli_mat23[i*N+kk]);
                }
                // if (ii == 0 && index==0) printf("st:%d ed:%d; i: %d _sum: %d\n", st, ed, i, _sum);
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
            }
            if (FABS(coef) < eps) continue;

            // map state -> id
            int res = 0xffff;
            #pragma unroll
            for (int kk = 0; kk < MAXN/32; kk++) {
                if(kk % 2 == 0) {
                    big_id[kk/2] = state_batch[ii*N+kk] ^ pauli_mat12[sid*N+kk];
                } else {
                    big_id[kk/2] = ((uint64)(state_batch[ii*N+kk] ^ pauli_mat12[sid*N+kk]) << 32) | big_id[kk/2];
                }
            }
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            int32 _ist = 0, _ied = batch_size;
            //binary_find_bigInt(_ist, _ied, ks, vs, id_width, big_id, &psi_real, &psi_imag, &res);
            binary_find_bigInt<ID_WIDTH>(_ist, _ied, ks, vs, big_id, &psi_real, &psi_imag, &res);
            // printf("index: %d big_id[0]: %llu res: %d\n", index, big_id[0], res);

            // don't find this coupled state in current samples
            if (res != 0) {
                continue;
            }

            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;

            // printf("ii: %d coef: %f\n", ii, coef);
            // printf("ii=%d e_loc_real=%f\n", ii, e_loc_real);
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

#if defined(RELEASED)
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
#ifdef DEBUG_LOCAL_ENERGY
    printf("RELEASED BigInt wrapper\n");
#endif
    assert(g_n_qubits != -1);
    int64 all_batch_size = batch_size, id_width = (g_n_qubits-1)/64+1, ks_disp_idx = 0;
    calculate_local_energy_sampling_parallel_bigInt(
        all_batch_size,
        batch_size,
        _states,
        ist,
        ied,
        ks_disp_idx,
        ks,
        id_width,
        vs,
        rank,
        eps,
        res_eloc_batch);
}
#elif defined(BITARR_OPT)
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
#ifdef DEBUG_LOCAL_ENERGY
    printf("BITARR_OPT\n");
#endif
    Timer timer[4];
    timer[3].start();
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    timer[0].start();
    auto ret = convert2bitarray_batch(_states, batch_size, g_n_qubits);
    timer[0].stop("convert2bitarray_batch");
    const int num_uint32 = ret.first;
    uint32_t *states = ret.second;
    const size_t size_states = sizeof(uint32_t) * batch_size * num_uint32;
    const int32 N = num_uint32;
    // print_mat(_states, batch_size, g_n_qubits, "states");
    // print_mat_bit(states, batch_size, num_uint32, "states_bitarr");

    // timer[1].start();
    const size_t size_ks = sizeof(uint64_t) * batch_size;
    const size_t size_vs = sizeof(psi_dtype) * batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64_t *d_ks = NULL;
    dtype *d_states = NULL;
    psi_dtype *d_vs = NULL;
    psi_dtype *d_res_eloc_batch = NULL;

    cudaMalloc(&d_ks, size_ks);
    cudaMalloc(&d_vs, size_vs);
    cudaMalloc(&d_states, size_states);
    cudaMalloc(&d_res_eloc_batch, size_res_eloc_batch);
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_ks, ks, size_ks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, size_vs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, size_states, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    // timer[1].stop("cuda_malloc_memcpy");

    timer[2].start();
    // int nthreads = 256;
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
    if (n_qubits <= 32) {
        calculate_local_energy_bitarr_kernel<32><<<nblocks, nthreads>>>(
            num_uint32,
            g_NK,
            idxs,
            coeffs,
            pauli_mat12,
            pauli_mat23,
            batch_size,
            batch_size_cur_rank,
            ist,
            &d_states[ist*N],
            d_ks,
            d_vs,
            eps,
            d_res_eloc_batch);
    } else if (n_qubits <= 64) {
        calculate_local_energy_bitarr_kernel<64><<<nblocks, nthreads>>>(
            num_uint32,
            g_NK,
            idxs,
            coeffs,
            pauli_mat12,
            pauli_mat23,
            batch_size,
            batch_size_cur_rank,
            ist,
            &d_states[ist*N],
            d_ks,
            d_vs,
            eps,
            d_res_eloc_batch);
    } else {
        printf("Error: only support n_qubits <= 64\n");
    }

    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");
    timer[2].stop("local_energy_kernel");

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
    timer[3].stop("Cuda calculate_local_energy");
}
#else
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
    Timer timer[4];
    timer[3].start();
#ifdef DEBUG_LOCAL_ENERGY
    printf("ORG\n");
#endif
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    timer[0].start();
    // transform _states{int64} into states{dtype} and map {+1,-1} to {+1,0}
    // assume states id is ordered after unique sampling, for using binary find
    //const int64 target_value = -1;
    const size_t size_states = sizeof(dtype) * batch_size * N;
    dtype *states = NULL, *d_states = NULL;
    states = (dtype *)malloc(size_states);
    memset(states, 0, size_states); // init 0
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            //if (_states[i*N+j] != target_value) {
            if (_states[i*N+j] == 1) {
                states[i*N+j] = 1;
            }
        }
    }
    timer[0].stop("convert201_batch");

    const size_t size_ks = sizeof(uint64) * batch_size;
    const size_t size_vs = sizeof(psi_dtype) * batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64 *d_ks = NULL;
    psi_dtype *d_vs = NULL;
    psi_dtype *d_res_eloc_batch = NULL;

    cudaMalloc(&d_ks, size_ks);
    cudaMalloc(&d_vs, size_vs);
    cudaMalloc(&d_states, size_states);
    cudaMalloc(&d_res_eloc_batch, size_res_eloc_batch);
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_ks, ks, size_ks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, size_vs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, size_states, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");

    timer[2].start();
    // int nthreads = 256;
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
    if (nblocks * nthreads < batch_size_cur_rank) {
        puts("ERROR: nblocks * nthreads < batch_size_cur_rank");
        return;
    }
    calculate_local_energy_kernel<<<nblocks, nthreads>>>(

    // const dim3 nthreads(32, 32);
    // const int nblocks = batch_size_cur_rank;
    //calculate_local_energy_kernel_V1<<<nblocks, nthreads>>>(
        n_qubits,
        g_NK,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        batch_size,
        batch_size_cur_rank,
        ist,
        &d_states[ist*N],
        d_ks,
        d_vs,
        eps,
        d_res_eloc_batch);
    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");
    timer[2].stop("local_energy_kernel");

#ifdef NPY_SAVE_DATA
    npy_cpp::NumpyCpp np;
    np.insert("batch_size", &batch_size, sizeof(batch_size));
    np.insert("_states", _states, sizeof(int64)*batch_size*N);
    np.insert("ks", ks, size_ks);
    np.insert("vs", vs, size_vs);
    np.insert("res_eloc_batch", res_eloc_batch, size_res_eloc_batch);
    np.insert("ist", &ist, sizeof(ist));
    np.insert("ied", &ied, sizeof(ied));
    np.insert("rank", &rank, sizeof(rank));
    np.insert("eps", &eps, sizeof(eps));
    np.savez(indata_file);
     
    np.loadz(indata_file);
    npy_cpp::check(np["batch_size"], &batch_size, sizeof(batch_size), "batch_size");
    npy_cpp::check(np["_states"], _states, sizeof(int64)*batch_size*N, "_states");
    npy_cpp::check(np["ks"], ks, size_ks, "ks");
    npy_cpp::check(np["vs"], vs, size_vs, "vs");
    npy_cpp::check(np["res_eloc_batch"], res_eloc_batch, size_res_eloc_batch, "res_eloc_batch");
    npy_cpp::check(np["ist"], &ist, sizeof(ist), "ist");
    npy_cpp::check(np["ied"], &ied, sizeof(ied), "ied");
    npy_cpp::check(np["rank"], &rank, sizeof(rank), "rank");
    npy_cpp::check(np["eps"], &eps, sizeof(eps), "eps");
#endif

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
    timer[3].stop("Cuda calculate_local_energy");
}
#endif

void calculate_local_energy_sampling_parallel(
    const int64 all_batch_size,
    const int64 batch_size,
    const int64 *_states,
    const int64 ist,
    const int64 ied,
    const int64 ks_disp_idx,
    const uint64_t *ks,
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
    const size_t size_states = sizeof(dtype) * batch_size * N;
    dtype *states = NULL, *d_states = NULL;
    states = (dtype *)malloc(size_states);
    memset(states, 0, size_states); // init 0
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            //if (_states[i*N+j] != target_value) {
            if (_states[i*N+j] == 1) {
                states[i*N+j] = 1;
            }
        }
    }

    const size_t size_ks = sizeof(uint64_t) * all_batch_size;
    const size_t size_vs = sizeof(psi_dtype) * all_batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64_t *d_ks = NULL;
    psi_dtype *d_vs = NULL;
    psi_dtype *d_res_eloc_batch = NULL;

    cudaMalloc(&d_ks, size_ks);
    cudaMalloc(&d_vs, size_vs);
    cudaMalloc(&d_states, size_states);
    cudaMalloc(&d_res_eloc_batch, size_res_eloc_batch);
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_ks, ks, size_ks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, size_vs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, size_states, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    
    // int nthreads = 256;
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
    calculate_local_energy_kernel<<<nblocks, nthreads>>>(

    // const dim3 nthreads(32, 32);
    // const int nblocks = batch_size_cur_rank;
    // calculate_local_energy_kernel_V1<<<nblocks, nthreads>>>(
        n_qubits,
        g_NK,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        all_batch_size,
        batch_size_cur_rank,
        ks_disp_idx,
        &d_states[ist*N],
        d_ks,
        d_vs,
        eps,
        d_res_eloc_batch);
    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
}

#if defined(BITARR_HASH_OPT)
// find coupled state first.
// if don't exist we just calculate next coupled state and drop the coef calculation
template<int MAXN=64, size_t KN=16>
__global__ void calculate_local_energy_kernel_bigInt_V1_bitarr_hash(
    const int32 num_uint32,
    const int64 NK,
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
    psi_dtype *res_eloc_batch,
    myHashTable<KN> ht)
{
    const int32 N = num_uint32;
    const int32 index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32 stride = gridDim.x * blockDim.x;

    //uint64 big_id[ID_WIDTH];
    const int ID_WIDTH = (MAXN + ID_UNIT_SIZE - 1) / ID_UNIT_SIZE;
    uint64 big_id[ID_WIDTH];
    //int myfound = 0;
    //int mymissed = 0;

    // loop all samples
    for (int ii = index; ii < batch_size_cur_rank; ii+=stride) {
        psi_dtype e_loc_real = 0, e_loc_imag = 0;
        for (int sid = 0; sid < NK; sid++) {
            psi_dtype psi_real = 0., psi_imag = 0.;
            // map state -> id
            #pragma unroll
            for (int kk = 0; kk < MAXN/32; kk++) {
                if(kk % 2 == 0) {
                    big_id[kk/2] = state_batch[ii*N+kk] ^ pauli_mat12[sid*N+kk];
                } else {
                    big_id[kk/2] = ((uint64)(state_batch[ii*N+kk] ^ pauli_mat12[sid*N+kk]) << 32) | big_id[kk/2];
                }
            }
            // binary find id among the sampled samples
            // idx = binary_find(ks, id), [_ist, _ied) start from 0
            //int32 _ist = 0, _ied = batch_size;
            
            //binary_find_bigInt(_ist, _ied, ks, vs, id_width, big_id, &psi_real, &psi_imag, &res);
            //KeyT<KN> key(big_id[0], big_id[1]);
            KeyT<KN> key(big_id);

            int64_t off = ht.search_key(key);

            if (off != -1) {
                psi_real = ht.values[off].data[0];
                psi_imag = ht.values[off].data[1];
            } else {
                continue;
            }

            coeff_dtype coef = 0.0;

            int st = idxs[sid], ed = idxs[sid+1];
            for (int i = st; i < ed; i++) {
                int _sum = 0;
                #pragma unroll
                for (int kk = 0; kk < MAXN/32; kk++) {
                    _sum += __popc(state_batch[ii*N+kk] & pauli_mat23[i*N+kk]);
                }
                // coef += ((-1)^(_sum)) * coeffs[i]; // pow_n?
                const psi_dtype _sgn = (_sum % 2 == 0) ? 1 : -1;
                coef += _sgn * coeffs[i];
            }
            // if (FABS(coef) < eps) continue;
            e_loc_real += coef * psi_real;
            e_loc_imag += coef * psi_imag;
        }
        // store the result number as return
        // (a+bi)/(c+di) = (ac+bd)/(c^2+d^2) + (bc-ad)/(c^2+d^2)i
        const psi_dtype a = e_loc_real, b = e_loc_imag;
        const psi_dtype c = vs[(ist+ii)*2], d = vs[(ist+ii)*2+1];
        const psi_dtype c2_d2 = c*c + d*d;
        res_eloc_batch[ii*2  ] = (a*c + b*d) / c2_d2;
        res_eloc_batch[ii*2+1] = -(a*d - b*c) / c2_d2;
    }   
}
#endif

#ifdef BITARR_OPT
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
#ifdef DEBUG_LOCAL_ENERGY
    printf("BITARR_OPT BigInt\n");
#endif
    Timer timer[4];
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    timer[0].start();
    auto ret = convert2bitarray_batch(_states, batch_size, g_n_qubits);
    timer[0].stop("convert2bitarray_batch");
    const int num_uint32 = ret.first;
    uint32 *states = ret.second;
    const size_t size_states = sizeof(uint32) * batch_size * num_uint32;

    const size_t size_ks = sizeof(uint64) * all_batch_size * id_width;
    const size_t size_vs = sizeof(psi_dtype) * all_batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64 *d_ks = NULL;
    dtype *d_states = NULL;
    psi_dtype *d_vs = NULL;
    psi_dtype *d_res_eloc_batch = NULL;

    cudaMalloc(&d_ks, size_ks);
    cudaMalloc(&d_vs, size_vs);
    cudaMalloc(&d_states, size_states);
    cudaMalloc(&d_res_eloc_batch, size_res_eloc_batch);
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_ks, ks, size_ks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, size_vs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, size_states, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    
    timer[2].start();
    // int nthreads = 256;
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
#define CALCULATE_KERNEL_BIGINT_V1_BITOPT(_id) calculate_local_energy_kernel_bigInt_V1_bitarr<((_id)+1)*32>
    void (*pFunc[])(
        const int32 num_uint32,
        const int64 NK,
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
        psi_dtype *res_eloc_batch) = 
    {
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(0),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(1),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(2),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(3),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(4),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(5),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(6),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(7),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(8),
        CALCULATE_KERNEL_BIGINT_V1_BITOPT(9), // (9+1)*32 = 320 qubits
        // ... You can continue for as many cases as needed

        //calculate_local_energy_kernel_bigInt_bitarr<32>,
        //calculate_local_energy_kernel_bigInt_bitarr<64>,
        //calculate_local_energy_kernel_bigInt_bitarr<96>,
        //calculate_local_energy_kernel_bigInt_bitarr<128>
    };
    int pFuncId = (n_qubits - 1) / 32; // n_qubits >= 1
    //printf("pFuncId: %d\n", pFuncId);
    pFunc[pFuncId]<<<nblocks, nthreads>>>(
        num_uint32,
        g_NK,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        all_batch_size,
        batch_size_cur_rank,
        ks_disp_idx,
        &d_states[ist*N],
        d_ks,
        id_width,
        d_vs,
        eps,
        d_res_eloc_batch);

    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");
    timer[2].stop("bigIntBitarray: local_energy_kernel");

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
}
#elif defined(BITARR_HASH_OPT)
#define funcWrapper(pFunc, maxn, kn) \
    pFunc<(maxn),(kn)><<<nblocks, nthreads>>>( \
        num_uint32, \
        g_NK, \
        idxs, \
        coeffs, \
        pauli_mat12, \
        pauli_mat23, \
        all_batch_size, \
        batch_size_cur_rank, \
        ks_disp_idx, \
        &d_states[ist*N], \
        d_ks, \
        id_width, \
        d_vs, \
        eps, \
        d_res_eloc_batch, \
        ht); \
    cudaCheckErrors("kernel launch failure"); \
    cudaDeviceSynchronize(); \
    freeHashTable(ht);

template<size_t N>
myHashTable<N> buildHashTableWrapper(uint64 *d_ks, psi_dtype *d_vs, int64 all_batch_size) {
    float avg2cacheline = 0.3;
    float avg2bsize = 0.55;

    int cacheline_size = 128/sizeof(KeyT<N>);
    int avg_size = cacheline_size*avg2cacheline;
    int bucket_size = avg_size/avg2bsize;
    int bucket_num = (all_batch_size + avg_size - 1)/avg_size;

#ifdef DEBUG_LOCAL_ENERGY
    printf("sizeof(KeyT<N>): %d cs: %d avg_size: %d bs: %d bn: %d\n", sizeof(KeyT<N>), cacheline_size, avg_size, bucket_size, bucket_num);
#endif
    myHashTable<N> ht;

    int rebuild_num = 0;
    while(!buildHashTable(ht, (KeyT<N> *)d_ks, (ValueT *)d_vs, bucket_num, bucket_size, all_batch_size)) {
        bucket_size = 1.4*bucket_size+1.0; // avoid dead loop when bs=1
#ifdef DEBUG_LOCAL_ENERGY
        rebuild_num++;
        avg2bsize = (float)avg_size/bucket_size;
        printf("Build hash table failed! avg_size=%d avg2bsize=%f bucket_size=%d\n", avg_size, avg2bsize, bucket_size);
#endif
    }

#ifdef DEBUG_LOCAL_ENERGY
    float fill_rate = (float)all_batch_size / (bucket_size * bucket_num);
    printf("bucket_num: %d rebuild_num: %d bucket_size: %d fill_rate: %.2f\n", bucket_num, rebuild_num, bucket_size, fill_rate);
#endif

    return ht;
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
#ifdef DEBUG_LOCAL_ENERGY
    printf("BITARR_HASH_OPT BigInt\n");
#endif
    Timer timer[4];
    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    timer[0].start();
    auto ret = convert2bitarray_batch(_states, batch_size, g_n_qubits);
    timer[0].stop("convert2bitarray_batch");
    const int num_uint32 = ret.first;
    uint32 *states = ret.second;
    const size_t size_states = sizeof(uint32) * batch_size * num_uint32;

    const size_t size_ks = sizeof(uint64) * all_batch_size * id_width;
    const size_t size_vs = sizeof(psi_dtype) * all_batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64 *d_ks = NULL;
    dtype *d_states = NULL;
    psi_dtype *d_vs = NULL;
    psi_dtype *d_res_eloc_batch = NULL;

    cudaMalloc(&d_ks, size_ks);
    cudaMalloc(&d_vs, size_vs);
    cudaMalloc(&d_states, size_states);
    cudaMalloc(&d_res_eloc_batch, size_res_eloc_batch);
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_ks, ks, size_ks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, size_vs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, size_states, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    
    timer[2].start();
    // int nthreads = 256;
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
    // calculate_local_energy_kernel_bigInt<<<nblocks, nthreads>>>(

    /*long *found, *missed;
    cudaMalloc((void **)&found, sizeof(long)*nblocks*4);
    cudaMalloc((void **)&missed, sizeof(long)*nblocks*4);
    cudaMemset(found, 0, sizeof(long)*nblocks*4);
    cudaMemset(found, 0, sizeof(long)*nblocks*4);*/

    int pFuncId = (n_qubits - 1) / 32; // n_qubits >= 1

#define GEN_WRAPPER(_pFuncId) \
    auto ht = buildHashTableWrapper<((_pFuncId)/2+1)*8>(d_ks, d_vs, all_batch_size); \
    funcWrapper(calculate_local_energy_kernel_bigInt_V1_bitarr_hash, ((_pFuncId+1)*32), ((_pFuncId)/2+1)*8);
#define GEN_WRAPPER_CASE(ID) case ID: {GEN_WRAPPER(ID); break;}

#ifdef DEBUG_LOCAL_ENERGY
    printf("batchsize is %d, g_nk is %d, id_width is %d\n", batch_size_cur_rank, g_NK, id_width);
    printf("pFuncId: %d\n", pFuncId);
#endif
    // if (pFuncId == 0) {
    //     auto ht = buildHashTableWrapper<8>(d_ks, d_vs, all_batch_size);
    //     funcWrapper(calculate_local_energy_kernel_bigInt_V1_bitarr_hash, 32, 8);
    // } else if (pFuncId == 1) {
    //     myHashTable<8> ht = buildHashTableWrapper<8>(d_ks, d_vs, all_batch_size);
    //     funcWrapper(calculate_local_energy_kernel_bigInt_V1_bitarr_hash, 64, 8);
    // } else if (pFuncId == 2) {
    //     auto ht = buildHashTableWrapper<16>(d_ks, d_vs, all_batch_size);
    //     funcWrapper(calculate_local_energy_kernel_bigInt_V1_bitarr_hash, 96, 16);
    // } else if (pFuncId == 3) {
    //     auto ht = buildHashTableWrapper<16>(d_ks, d_vs, all_batch_size);
    //     funcWrapper(calculate_local_energy_kernel_bigInt_V1_bitarr_hash, 128, 16);
    // } else if (pFuncId == 4) {
    //     auto ht = buildHashTableWrapper<24>(d_ks, d_vs, all_batch_size);
    //     funcWrapper(calculate_local_energy_kernel_bigInt_V1_bitarr_hash, 160, 24);
    // } else {
    //     printf("Error n_qubits= %d is unsupported!\n", n_qubits);
    // }
    switch (pFuncId) {
        GEN_WRAPPER_CASE(0);
        GEN_WRAPPER_CASE(1);
        GEN_WRAPPER_CASE(2);
        GEN_WRAPPER_CASE(3);
        GEN_WRAPPER_CASE(4);
        GEN_WRAPPER_CASE(5);
        GEN_WRAPPER_CASE(6);
        GEN_WRAPPER_CASE(7); // (7+1) * 32 = 256 qubits
        GEN_WRAPPER_CASE(8); // (8+1) * 32 = 288 qubits
        GEN_WRAPPER_CASE(9); // (9+1) * 32 = 320 qubits
        // ... You can continue for as many cases as needed
        default:
            printf("Error n_qubits= %d is unsupported! Must <=320 qubits now!\n", n_qubits);
            break;
    };

    /*long found_total = thrust::reduce(thrust::device, found, found + nblocks*4);
    long missed_total = thrust::reduce(thrust::device, missed, missed + nblocks*4);*/
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");
    timer[2].stop("bigIntBitarray: local_energy_kernel");
    //printf("total missed is %ld, total found is %ld\n", missed_total, found_total);

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
    //cudaFree(found);
    //cudaFree(missed);
}
#else
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
#ifdef DEBUG_LOCAL_ENERGY
    printf("ORG BigInt\n");
#endif
    Timer timer[4];

    const int32 n_qubits = g_n_qubits;
    const int64 *idxs = g_idxs;
    const coeff_dtype *coeffs = g_coeffs;
    const dtype *pauli_mat12 = g_pauli_mat12;
    const dtype *pauli_mat23 = g_pauli_mat23;

    const int64 batch_size_cur_rank = ied - ist;
    const int32 N = g_n_qubits;

    timer[0].start();
    // transform _states{int64} into states{dtype} and map {+1,-1} to {+1,0}
    // assume states id is ordered after unique sampling, for using binary find
    //const int64 target_value = -1;
    const size_t size_states = sizeof(dtype) * batch_size * N;
    dtype *states = NULL, *d_states = NULL;
    states = (dtype *)malloc(size_states);
    memset(states, 0, size_states); // init 0
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < N; j++) {
            //if (_states[i*N+j] != target_value) {
            if (_states[i*N+j] == 1) {
                states[i*N+j] = 1;
            }
        }
    }
    timer[0].stop("convert_states");

    const size_t size_ks = sizeof(uint64) * all_batch_size * id_width;
    const size_t size_vs = sizeof(psi_dtype) * all_batch_size * 2;
    const size_t size_res_eloc_batch = sizeof(psi_dtype) * batch_size_cur_rank * 2;
    uint64 *d_ks = NULL;
    psi_dtype *d_vs = NULL;
    psi_dtype *d_res_eloc_batch = NULL;

    cudaMalloc(&d_ks, size_ks);
    cudaMalloc(&d_vs, size_vs);
    cudaMalloc(&d_states, size_states);
    cudaMalloc(&d_res_eloc_batch, size_res_eloc_batch);
    cudaCheckErrors("cudaMalloc failure");
    cudaMemcpy(d_ks, ks, size_ks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vs, vs, size_vs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_states, states, size_states, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    
    timer[1].start();
    const int nthreads = 128;
    const int nblocks = batch_size_cur_rank / nthreads + ((batch_size_cur_rank%nthreads) != 0);
#define CALCULATE_KERNEL_BIGINT_V1(_id) calculate_local_energy_kernel_bigInt_V1<((_id)+1)*32>
    void (*pFunc[])(
        const int32 num_uint32,
        const int64 NK,
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
        psi_dtype *res_eloc_batch) = 
    {
        CALCULATE_KERNEL_BIGINT_V1(0),
        CALCULATE_KERNEL_BIGINT_V1(1),
        CALCULATE_KERNEL_BIGINT_V1(2),
        CALCULATE_KERNEL_BIGINT_V1(3),
        CALCULATE_KERNEL_BIGINT_V1(4),
        CALCULATE_KERNEL_BIGINT_V1(5),
        CALCULATE_KERNEL_BIGINT_V1(6),
        CALCULATE_KERNEL_BIGINT_V1(7),
        CALCULATE_KERNEL_BIGINT_V1(8),
        CALCULATE_KERNEL_BIGINT_V1(9), // (9+1)*32 = 320 qubits
        // ... You can continue for as many cases as needed

        //calculate_local_energy_kernel_bigInt<32>,
        //calculate_local_energy_kernel_bigInt<64>,
        //calculate_local_energy_kernel_bigInt<96>,
        //calculate_local_energy_kernel_bigInt<128>
    };
    int pFuncId = (n_qubits - 1) / 32; // n_qubits >= 1
#ifdef DEBUG_LOCAL_ENERGY
    printf("pFuncId: %d\n", pFuncId);
#endif
    pFunc[pFuncId]<<<nblocks, nthreads>>>(
        n_qubits,
        g_NK,
        idxs,
        coeffs,
        pauli_mat12,
        pauli_mat23,
        all_batch_size,
        batch_size_cur_rank,
        ks_disp_idx,
        &d_states[ist*N],
        d_ks,
        id_width,
        d_vs,
        eps,
        d_res_eloc_batch);

    cudaCheckErrors("kernel launch failure");
    cudaDeviceSynchronize();
    cudaMemcpy(res_eloc_batch, d_res_eloc_batch, size_res_eloc_batch, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy failure");
    timer[1].stop("bigIntOrg: local_energy_kernel");

#ifdef NPY_SAVE_DATA
    npy_cpp::NumpyCpp np;
    np.insert("all_batch_size", &all_batch_size, sizeof(all_batch_size));
    np.insert("batch_size", &batch_size, sizeof(batch_size));
    np.insert("_states", _states, sizeof(int64)*batch_size*N);
    np.insert("ks", ks, size_ks);
    np.insert("vs", vs, size_vs);
    np.insert("res_eloc_batch", res_eloc_batch, size_res_eloc_batch);
    np.insert("ist", &ist, sizeof(ist));
    np.insert("ied", &ied, sizeof(ied));
    np.insert("id_width", &id_width, sizeof(id_width));
    np.insert("ks_disp_idx", &ks_disp_idx, sizeof(ks_disp_idx));
    np.insert("rank", &rank, sizeof(rank));
    np.insert("eps", &eps, sizeof(eps));
    np.savez(indata_file);

    np.loadz(indata_file);
    npy_cpp::check(np["batch_size"], &batch_size, sizeof(batch_size), "batch_size");
    npy_cpp::check(np["_states"], _states, sizeof(int64)*batch_size*N, "_states");
    npy_cpp::check(np["ks"], ks, size_ks, "ks");
    npy_cpp::check(np["vs"], vs, size_vs, "vs");
    npy_cpp::check(np["res_eloc_batch"], res_eloc_batch, size_res_eloc_batch, "res_eloc_batch");
    npy_cpp::check(np["ist"], &ist, sizeof(ist), "ist");
    npy_cpp::check(np["ied"], &ied, sizeof(ied), "ied");
    npy_cpp::check(np["rank"], &rank, sizeof(rank), "rank");
    npy_cpp::check(np["all_batch_size"], &all_batch_size, sizeof(all_batch_size), "all_batch_size");
    npy_cpp::check(np["id_width"], &id_width, sizeof(id_width), "id_width");
    npy_cpp::check(np["ks_disp_idx"], &ks_disp_idx, sizeof(ks_disp_idx), "ks_disp_idx");
    npy_cpp::check(np["eps"], &eps, sizeof(eps), "eps");
#endif

    free(states);
    cudaFree(d_states);
    cudaFree(d_res_eloc_batch);
    cudaFree(d_ks);
    cudaFree(d_vs);
}
#endif

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

        cudaFree(g_idxs);
        cudaFree(g_coeffs);
        cudaFree(g_pauli_mat12);
        cudaFree(g_pauli_mat23);
    }
}

