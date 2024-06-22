#!/usr/bin/env python
# encoding: utf-8

import math
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import calculate_local_energy as eloc_cpp


# Constants
__STRIDE = 64
__EPS = 1e-12

__POWER2_LOOKUP = 2**np.arange(64, dtype=np.uint64)
def __state2id_huge_batch(state: np.ndarray, ret_id_width=False) -> np.ndarray:
    """
    Mapping state ({0|1}^N) into uint64 id list.
    """
    id_width = math.ceil(state.shape[1] / __STRIDE)
    ret_id = np.zeros((state.shape[0], id_width), dtype=np.uint64)
    for i in range(id_width):
        _state_mask = state[:, i*__STRIDE : (i+1)*__STRIDE].clip(0,1).astype(np.uint64)
        L = _state_mask.shape[1]
        ret_id[:,i] = _state_mask @ __POWER2_LOOKUP[:L]
    # print(f"state: {state}")
    # print(f"ret_id: {ret_id}")
    if ret_id_width:
        return ret_id, id_width
    return ret_id

def __ensure_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Ensure the input array is a NumPy array. Convert if it's a PyTorch tensor.
    """
    if not isinstance(array, (np.ndarray, torch.Tensor)):
        raise TypeError("Input must be either a NumPy array or a PyTorch tensor.")

    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array

def calculate_local_energy(
    states: Union[np.ndarray, torch.Tensor],
    psis: Union[np.ndarray, torch.Tensor],
    is_need_sort: bool = False) -> np.ndarray:
    """
    Calculate local energy for given states.
    states, psis, weights must be np.ndarray or torch.Tensor
    return: local energies: np.ndarray
    """
    states = __ensure_numpy(states)
    psis = __ensure_numpy(psis)

    assert states.shape[0] == psis.shape[0]

    ptype_real = type(psis[0].real)
    num_unique = len(psis)

    ks = __state2id_huge_batch(states)
    vs = np.stack((np.real(psis), np.imag(psis)), axis=1).astype(ptype_real)
    if is_need_sort:
        idxs = np.argsort(ks, axis=0).reshape(-1)
        ks = ks[idxs, :]
        vs = vs[idxs, :]
        states = states[idxs, :]
        psis = psis[idxs]

    #print(f'ptype_real: {ptype_real} num_unique: {num_unique} states: {states.shape}')
    energy_space = np.zeros(num_unique*2, dtype=ptype_real)
    ks_disp_idx = 0
    ist, ied = 0, num_unique
    k_idxs = np.zeros(1, dtype=np.int64)

    eloc_cpp.calculate_local_energy(num_unique, states.reshape(-1), ist, ied, k_idxs, ks.reshape(-1), vs.reshape(-1), ks_disp_idx, __EPS, energy_space)
    #print(f"ks: {ks}\nvs: {vs}")

    energy_space = energy_space.reshape(num_unique, 2)
    local_energies = np.zeros(num_unique, dtype=psis.dtype)
    local_energies[:] = energy_space[:, 0] + energy_space[:, 1]*1j

    if is_need_sort:
        idxs_rev = np.argsort(idxs)
        local_energies = local_energies[idxs_rev]

    return local_energies

def log_rank(rank, mess, prompt=""):
    print(f"[rank {rank}] {prompt} mess: {mess}")

def npy_to_torch_dtype(np_dtype):
    dtype_mapping = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
    }
    return dtype_mapping.get(np_dtype, None)

def calculate_local_energy_parallel(
    states_local: Union[np.ndarray, torch.Tensor],
    psis_local: Union[np.ndarray, torch.Tensor],
    device: str = 'cpu',
    global_rank: int = 0,
    world_size: int = 1,
    is_need_sort: bool = False) -> np.ndarray:
    """
    Calculate local energy for given states.
    states_local, psis_local, weights must be np.ndarray or torch.Tensor
    return: local energies: np.ndarray
    """
    states_local = __ensure_numpy(states_local)
    psis_local = __ensure_numpy(psis_local)

    assert states_local.shape[0] == psis_local.shape[0]

    ptype_real = type(psis_local[0].real)
    ptype_real_torch = npy_to_torch_dtype(ptype_real)
    num_unique_local = len(psis_local)

    ks_local, id_width = __state2id_huge_batch(states_local, ret_id_width=True)
    vs_local = np.stack((np.real(psis_local), np.imag(psis_local)), axis=1).astype(ptype_real)
    if is_need_sort:
        idxs = np.argsort(ks_local, axis=0).reshape(-1)
        ks_local = ks_local[idxs, :]
        vs_local = vs_local[idxs, :]
        states_local = states_local[idxs, :]
        psis_local = psis_local[idxs]

    # store results
    energy_space = np.zeros(num_unique_local*2, dtype=ptype_real)

    # Collect data: concat all unique samples's psis for local energy calculation
    num_unique_local_tensor = torch.zeros(world_size, dtype=type(num_unique_local), device=device)
    dist.all_gather_into_tensor(num_unique_local_tensor, torch.tensor(num_unique_local, device=device))
    num_unique_global = num_unique_local_tensor.sum().item()

    # simulate MPI all_gatherv
    vs_global = [torch.zeros(num_unique.item()*2, dtype=ptype_real_torch, device=device) for num_unique in num_unique_local_tensor] # float32 / float64
    ks_global = [torch.zeros(num_unique.item(), dtype=torch.int64, device=device) for num_unique in num_unique_local_tensor] # torch unsupport uint64
    vs_local, ks_local = torch.from_numpy(vs_local).to(device), torch.from_numpy(ks_local.astype(np.int64)).to(device)
    dist.all_gather(vs_global, vs_local)
    dist.all_gather(ks_global, ks_local)
    vs_global, ks_global = __ensure_numpy(torch.cat(vs_global)), __ensure_numpy(torch.cat(ks_global)).astype(np.uint64) # solve torch unsupport uint64

    # calculate current rank ks's disp in the ks_global
    ks_disp_idx: int = sum(num_unique_local_tensor[:global_rank])
    ist, ied = 0, num_unique_local

    # eloc_cpp.calculate_local_energy_sampling_parallel_bigInt(all_batch_size, batch_size, _states.data(), ist, ied, ks_disp_idx, ks.data(), id_width, vs.data(), rank, eps, res_eloc_batch.mutable_data())
    eloc_cpp.calculate_local_energy_sampling_parallel_bigInt(num_unique_global, num_unique_local, states_local.reshape(-1), ist, ied, ks_disp_idx, ks_global.reshape(-1), id_width, vs_global.reshape(-1), global_rank, __EPS, energy_space)

    energy_space = energy_space.reshape(num_unique_local, 2)
    local_energies = np.zeros(num_unique_local, dtype=psis_local.dtype)
    local_energies[:] = energy_space[:, 0] + energy_space[:, 1]*1j

    if is_need_sort:
        idxs_rev = np.argsort(idxs)
        local_energies = local_energies[idxs_rev]

    return local_energies

def energy(
    states: Union[np.ndarray, torch.Tensor],
    psis: Union[np.ndarray, torch.Tensor],
    weights: Union[np.ndarray, torch.Tensor],
    is_need_sort: bool = False):

    assert states.shape[0] == psis.shape[0] == weights.shape[0]
    weights = __ensure_numpy(weights)
    local_energies = calculate_local_energy(states, psis, is_need_sort=is_need_sort)
    weights = weights / weights.sum() # norm
    eloc_expectation = np.dot(weights, local_energies)
    # print(f"local_energies: {local_energies}")
    # print(f"weights: {weights}")

    return eloc_expectation

def init_hamiltonian(ham_file: str):
    n_qubits = eloc_cpp.init_hamiltonian(ham_file)
    return n_qubits

def free_hamiltonian():
    eloc_cpp.free_hamiltonian()
