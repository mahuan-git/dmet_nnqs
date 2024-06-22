#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "switch_backend.h"

namespace py = pybind11;

PYBIND11_MODULE(calculate_local_energy, m) {
    m.def("init_hamiltonian", &init_hamiltonian, "Initialize hamiltonian from file");

    m.def("free_hamiltonian", &free_hamiltonian, "Free hamiltonian resources");

    m.def("set_indices_ham_int_opt", [](const int32 n_qubits,
                                        const int64 K,
                                        const int64 NK,
                                        py::array_t<int64> idxs,
                                        py::array_t<coeff_dtype> coeffs,
                                        py::array_t<int8> pauli_mat12,
                                        py::array_t<int8> pauli_mat23) {
        set_indices_ham_int_opt(n_qubits, K, NK, idxs.data(), coeffs.data(), pauli_mat12.data(), pauli_mat23.data());
    }, "Set indices for Hamiltonian integer optimization");

    m.def("calculate_local_energy", [](const int64 batch_size,
                                       py::array_t<int64> _states,
                                       const int64 ist,
                                       const int64 ied,
                                       py::array_t<int64> k_idxs,
                                       py::array_t<uint64> ks,
                                       py::array_t<psi_dtype> vs,
                                       const int64 rank,
                                       const float64 eps,
                                       py::array_t<psi_dtype> res_eloc_batch) {
        calculate_local_energy(batch_size, _states.data(), ist, ied, k_idxs.data(), ks.data(), vs.data(), rank, eps, res_eloc_batch.mutable_data());
    }, "Calculate local energy");

    m.def("calculate_local_energy_sampling_parallel", [](const int64 all_batch_size,
                                                         const int64 batch_size,
                                                         py::array_t<int64> _states,
                                                         const int64 ist,
                                                         const int64 ied,
                                                         const int64 ks_disp_idx,
                                                         py::array_t<uint64> ks,
                                                         py::array_t<psi_dtype> vs,
                                                         const int64 rank,
                                                         const float64 eps,
                                                         py::array_t<psi_dtype> res_eloc_batch) {
        calculate_local_energy_sampling_parallel(all_batch_size, batch_size, _states.data(), ist, ied, ks_disp_idx, ks.data(), vs.data(), rank, eps, res_eloc_batch.mutable_data());
    }, "Calculate local energy with sampling in parallel");

    m.def("calculate_local_energy_sampling_parallel_bigInt", [](const int64 all_batch_size,
                                                               const int64 batch_size,
                                                               py::array_t<int64> _states,
                                                               const int64 ist,
                                                               const int64 ied,
                                                               const int64 ks_disp_idx,
                                                               py::array_t<uint64> ks,
                                                               const int64 id_width,
                                                               py::array_t<psi_dtype> vs,
                                                               const int64 rank,
                                                               const float64 eps,
                                                               py::array_t<psi_dtype> res_eloc_batch) {
        calculate_local_energy_sampling_parallel_bigInt(all_batch_size, batch_size, _states.data(), ist, ied, ks_disp_idx, ks.data(), id_width, vs.data(), rank, eps, res_eloc_batch.mutable_data());
    }, "Calculate local energy with sampling in parallel using BigInt");

#ifdef BACKEND_GPU
    m.def("set_gpu_to_mpi", &set_gpu_to_mpi, "Set GPU to MPI");

    m.def("get_gpu_id", [](int rank, int print_verbose) {
        get_gpu_id(rank, print_verbose);
    }, "Get GPU ID");
#endif
}

