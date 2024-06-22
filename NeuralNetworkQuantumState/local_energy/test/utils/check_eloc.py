#!/usr/bin/env python
# encoding: utf-8

import sys
import openfermion
import scipy.sparse, scipy.sparse.linalg
import numpy as np

import test.utils.pyscf_helper as pyscf_helper
import test.utils.utils as utils

def gen_random_ham(n_orb=4, eps=1e-12, seed=111, is_save_ham=True):
    np.random.seed(seed)
    one_body_mo = np.random.rand(n_orb**2).reshape([n_orb]*2)
    two_body_mo = np.random.rand(n_orb**4).reshape([n_orb]*4)
    constant = np.random.rand()
    ham_1e, ham_2e = pyscf_helper.get_hamiltonian_ferm_op_from_mo_ints(one_body_mo, two_body_mo)
    ham = ham_1e + ham_2e + constant
    # Force ham to be hermitian
    ham = ham + openfermion.hermitian_conjugated(ham)

    n_qubits = openfermion.count_qubits(ham)
    ham_sparse_mat = openfermion.get_sparse_operator(ham)
    e_fci, v_fci = scipy.sparse.linalg.eigsh(ham_sparse_mat, k=1, which="SA")

    # filter
    v_fci[(abs(v_fci) < eps)] = 0.0
    cnt = (abs(v_fci) > 0.0).sum()
    # print(f"e_fci: {e_fci.shape} v: {v_fci.shape} cnt: {cnt}")
    print("FCI energy: ", v_fci.T.conj().dot(ham_sparse_mat.dot(v_fci)), 'v_fci:', v_fci.shape, 'eps:', eps)

    fname = f"qubits{n_qubits}-seed{seed}"
    op_name = ""
    if is_save_ham:
        qubit_op = openfermion.jordan_wigner(ham)
        op_name = f"{fname}-qubit_op.data"
        utils.save_binary_qubit_op(qubit_op, filename=op_name)
        print("Qubit Hamiltonian saved to %s." % (op_name))

    # if n_orb <= 4:
    #     ham_dense_mat = ham_sparse_mat.toarray()
    #     e_, v_ = np.linalg.eigh(ham_dense_mat)

    fci_name = save_fci_states_coefs(n_qubits, v_fci.reshape(-1), e_fci, fname=fname)
    return e_fci, op_name, fci_name

def save_fci_states_coefs(n_qubits, coeffs, e_fci, fname):
    dump_states = []
    bitstr_scipy = []
    coefs_scipy = []
    for i in range(2**n_qubits):
        coefs_i = coeffs[i]
        if abs(coefs_i)>1e-12:
            bitstr_i = bin(i)[2:]
            bitstr_i = "0" * (n_qubits - len(bitstr_i)) + bitstr_i
            bitstr_scipy.append(bitstr_i)
            coefs_scipy.append(coefs_i)
            dump_states.append(strToIntList(bitstr_i))
        else:
            #coefs_scipy.append(0+0j)
            pass

    bitstr_scipy = np.array(bitstr_scipy)
    coefs_scipy = np.array(coefs_scipy)
    #coefs_scipy = np.exp(1.j * -np.angle(coefs_scipy[abs(coefs_scipy).argmax()])) * coefs_scipy
    fname = f"{fname}-fci.npz"
    np.savez(fname, ci_states=np.array(dump_states), ci_probs=coefs_scipy, e_fci=e_fci[0])
    print(f'save into {fname}')
    return fname

def gen_molecule_ham(geometry, eps=1e-12, mol_name=None, is_save_ham=True):
    res = pyscf_helper.init_scf(geometry)
    #res = pyscf_helper.init_scf(geometry, use_symmetry=True)
    ham = res[5]

    n_qubits = openfermion.count_qubits(ham)
    ham_sparse_mat = openfermion.get_sparse_operator(ham)
    #np.savez("c2-ham-sparse_mat.npz", ham=ham_sparse_mat)
    e_fci, v_fci = scipy.sparse.linalg.eigsh(ham_sparse_mat, k=1, which="SA")
    v_fci = np.exp(1.j * -np.angle(v_fci[abs(v_fci).argmax()])) * v_fci

    print("FCI energy: ", v_fci.T.conj().dot(ham_sparse_mat.dot(v_fci)), 'v_fci:', v_fci.shape, 'eps:', eps)

    if is_save_ham:
        qubit_op = openfermion.jordan_wigner(ham)
        fname = f"{mol_name}-qubit_op.data"
        utils.save_binary_qubit_op(qubit_op, filename=fname)
        print("Qubit Hamiltonian saved to %s." % (fname))
    if mol_name is not None:
        save_fci_states_coefs(n_qubits, v_fci.reshape(-1), e_fci, fname=f"{mol_name}")
    return n_qubits, ham_sparse_mat, v_fci.reshape(-1)

def intListToStr(arrList):
    str_a = ''.join(str(x) for x in arrList)
    return str_a

def strToIntList(arrStr):
    l = [int(i) for i in arrStr]
    return l

def diff_coeff(v_fci, ci_states, my_fci):
    my_fci_pool = np.zeros_like(v_fci)
    for ci_state, fci in zip(ci_states, my_fci):
        fci_str = intListToStr(ci_state)
        #fci_str = "".join(reversed(fci_str))
        idx = int('0b'+fci_str, 2)
        my_fci_pool[idx] = fci
        err = abs(fci - v_fci[idx])
        print(f"idx: {idx} state: {ci_state} {fci_str} fci: {fci} v_fci: {v_fci[idx]} err: {err}")
    errs = abs(my_fci_pool - v_fci)
    some_errs = errs[(errs > 1e-10)]
    for err in some_errs:
        print(err)
    print(f"some_errs: {some_errs.shape}")

def single_state_eloc(bitstr, coeff=1.0):
    """
        Hartree-Fock state corresponds to bitstr="1100", coeff=1.0
        FCI correspons to bitstr="1100", coeff=-0.97672448-0.12359735j
        plus bitstr="11" (or 0011), coeff=0.17392127+0.02200847j,
        which corresponds to v_test has two non-zeros values.
    """
    if type(bitstr) is not str:
        bitstr = intListToStr(bitstr)

    v_test = np.zeros(2**n_qubits, dtype=np.complex128).reshape([-1, 1])
    bitstr = "0" * (n_qubits - len(bitstr)) + bitstr
    index_for_v_test = int("0b" + bitstr, 2)
    v_test[index_for_v_test, 0] = coeff
    e_test = v_test.T.conj().dot(ham_sparse_mat.dot(v_test))
    # print(f'state: {bitstr} coeff: {coeff} eloc: {e_test[0][0]}')
    return e_test[0][0]

def get_idx(bitstr):
    if type(bitstr) is not str:
        bitstr = intListToStr(bitstr)
    bitstr = "0" * (n_qubits - len(bitstr)) + bitstr
    index_for_v_test = int("0b" + bitstr, 2)
    return index_for_v_test

def single_Hxy_eloc(bitstrx, bitstry, coeffx=1.0, coeffy=1.0):
    v_test_x = np.zeros(2**n_qubits, dtype=np.complex128).reshape([-1, 1])
    v_test_y = np.zeros(2**n_qubits, dtype=np.complex128).reshape([-1, 1])
    v_test_x[get_idx(bitstrx), 0] = coeffx
    v_test_y[get_idx(bitstry), 0] = coeffy
    e_xx = v_test_x.T.conj().dot(ham_sparse_mat.dot(v_test_x))[0][0]
    e_xy = v_test_x.T.conj().dot(ham_sparse_mat.dot(v_test_y))[0][0]
    e_yy = v_test_y.T.conj().dot(ham_sparse_mat.dot(v_test_y))[0][0]
    e_yx = v_test_y.T.conj().dot(ham_sparse_mat.dot(v_test_x))[0][0]
    eloc = e_xx + e_xy + e_yy + e_yx
    return eloc, e_xx, e_xy, e_yy, e_yx

def multi_state_eloc(states, coeffs, n_qubits, ham_sparse_mat):
    """
    calculate expectation energy using <states, coeffs> and ham_sparse_mat
    """
    v_test = np.zeros(2**n_qubits, dtype=np.complex128).reshape([-1, 1])
    for bitstr, coeff in zip(states, coeffs):
        if type(bitstr) is not str:
            bitstr = intListToStr(bitstr)

        bitstr = "0" * (n_qubits - len(bitstr)) + bitstr
        #bitstr = "".join(reversed(bitstr))
        index_for_v_test = int("0b" + bitstr, 2)
        v_test[index_for_v_test, 0] = coeff

    e_test = v_test.T.conj().dot(ham_sparse_mat.dot(v_test))
    return e_test[0][0]

if __name__ == "__main__":
    test_type, mol_name = "mol_test", "H2"
    if len(sys.argv) >= 2:
        test_type = sys.argv[1]

    if len(sys.argv) >= 3:
        mol_name  = sys.argv[2]

    if test_type == "rand_test":
        # Rand test
        eloc, op_name, fci_name = gen_random_ham(n_orb=4, eps=1e-12, seed=111, is_save_ham=True)
    elif test_type == "mol_test":
        # Chemistry molecule test
        mol_dict = {
            "NH3" : [('N', (0, 0, 0)), ('H', (-0.4417, 0.2906, 0.8711)), ('H', (0.7256, 0.6896, -0.1907)), ('H', (0.4875, -0.8701, 0.2089))], # NH3
            "CH4" : [('C', (0, 0, 0)), ('H', (0.5541, 0.7996, 0.4965)), ('H', (0.6833, -0.8134, -0.2536)), ('H', (-0.7782, -0.3735, 0.6692)), ('H', (-0.4593, 0.3874, -0.9121))], # CH4
            "CH2" : [('C', (2.5369, -0.155, 0)), ('H', (3.0739, 0.155, 0)), ('H', (2, 0.155, 0))], # CH2 S=1
            "O2" : [('O', (-0.616, 0, 0)), ('O', (0.616, 0, 0))], # O2 S=1
            "LiH" : [('Li', (3, 0, 0)), ('H', (2, 0, 0))], # LiH
            "H2O" : [('O', (0, 0, 0)), ('H', (0.2774, 0.8929, 0.2544)), ('H', (0.6068, -0.2383, -0.7169))], # H2O
            "C2" : [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])], # C2
            "H2": [('H', (2, 0, 0)), ('H', (3, 0, 0))], # H2
        }
        geometry = mol_dict[mol_name]
        n_qubits, ham_sparse_mat, v_fci = gen_molecule_ham(geometry, mol_name=mol_name)
        print(f"geometry: {geometry}\nn_qubits: {n_qubits}")
    else:
        ValueError(f"Unsupport test_type: {test_type}")
