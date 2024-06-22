#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy as np
import torch
import interface.python.eloc as eloc
import test.utils.check_eloc as check_eloc

def strToIntList(arrStr):
    l = [int(i) for i in arrStr]
    return l

def test_rand_single(n_orb=4, seed=123, psis_dtype=torch.complex64, is_need_sort=False, err_eps=1e-4):
    eloc_ref, op_name, fci_name = check_eloc.gen_random_ham(n_orb=n_orb, eps=1e-12, seed=seed, is_save_ham=True)
    print(f'eloc_ref: {eloc_ref[0]} op_name: {op_name} fci_name: {fci_name}', flush=True)
    eloc.init_hamiltonian(op_name)
    ci_data = np.load(fci_name)
    ci_probs, ci_states = ci_data['ci_probs'], ci_data['ci_states']
    ci_probs, states = torch.from_numpy(ci_probs), torch.from_numpy(ci_states)
    if not (type(ci_probs) is torch.complex64 or type(ci_probs) is torch.complex128):
        ci_probs = ci_probs + 0j

    ci_probs = ci_probs.type(psis_dtype)
    counts = ci_probs.real ** 2
    eloc_expectation = eloc.energy(states, ci_probs, counts, True)
    eloc.free_hamiltonian()
    err = abs(eloc_expectation - eloc_ref)
    assert err < err_eps, f"err: {err} seed: {seed} n_orb: {n_orb}"

def test_rand():
    n_qubits_l, n_qubits_r = 1, 8
    n_rand = 10
    rands = np.abs(np.random.randn(n_rand)*100).astype(np.int)
    for i in range(n_qubits_l, n_qubits_r):
        for seed in rands:
            test_rand_single(n_orb=i+1, seed=seed)

def test_mol(ham_path: str, np_data: str, psis_dtype=torch.complex64, is_need_sort=False, err_eps=1e-4):
    n_qubits = eloc.init_hamiltonian(ham_path)
    ci_data = np.load(np_data, allow_pickle=True)
    # ci_probs, ci_states = ci_data['ci_probs'], ci_data['ci_states']
    ci_probs, ci_states, e_fci = ci_data['ci_probs'], ci_data['ci_states'], ci_data['e_fci']
    #ci_states[0] = np.array(strToIntList('11111111111100000000'))
    #ci_probs[0] = 1
    #ci_probs, ci_states = ci_probs[0:1], ci_states[0:1]
    #ci_probs = ci_probs.real
    #f_idxs = (ci_probs**2 > 1e-12)
    #ci_probs, ci_states = ci_probs[f_idxs], ci_states[f_idxs]
    ci_probs, states = torch.from_numpy(ci_probs), torch.from_numpy(ci_states)
    if not (type(ci_probs) is torch.complex64 or type(ci_probs) is torch.complex128):
        ci_probs = ci_probs + 0j

    ci_probs = ci_probs.type(psis_dtype)
    counts = ci_probs.real ** 2
    eloc_expectation = eloc.energy(states, ci_probs, counts, is_need_sort)
    err = np.abs(e_fci - eloc_expectation.real)
    assert err < err_eps, f"[test] CHECK FAIL == e_fci: {e_fci} eloc_expectation: {eloc_expectation.real} err: {err}"
    print(f"[test] PASS == n_qubits: {n_qubits} eloc_expectation: {eloc_expectation.real} ABS(err): {err}")
    eloc.free_hamiltonian()

if __name__ == "__main__":
    # test_rand()
    # exit(0)
    np_data = "H2-fci.npz"
    ham_path = "H2-qubit_op.data"
    np_data = "LiH-fci.npz"
    ham_path = "LiH-qubit_op.data"
    ham_path = "qubits8-seed111-qubit_op.data"
    np_data = "qubits8-seed111-fci.npz"
    if len(sys.argv) == 3:
        ham_path, np_data = sys.argv[1], sys.argv[2]
    # test_mol(ham_path, np_data, psis_dtype=torch.complex128, is_need_sort=True)
    test_mol(ham_path, np_data, psis_dtype=torch.complex64, is_need_sort=True)
