import itertools
from typing import TypeVar

import numpy
import openfermion


def _Pij(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = openfermion.FermionOperator(
        ((ja, 0), (ib, 0)),
        1.0
    )
    term2 = openfermion.FermionOperator(
        ((ia, 0), (jb, 0)),
        1.0
    )
    return numpy.sqrt(0.5) * (term1 + term2)


def _Pij_dagger(i: int, j: int):
    return openfermion.hermitian_conjugated(_Pij(i, j))


def _Qij_plus(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term = openfermion.FermionOperator(
        ((ja, 0), (ia, 0)),
        1.0
    )
    return term


def _Qij_minus(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term = openfermion.FermionOperator(
        ((jb, 0), (ib, 0)),
        1.0
    )
    return term


def _Qij_0(i: int, j: int):
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = openfermion.FermionOperator(
        ((ja, 0), (ib, 0)),
        1.0
    )
    term2 = openfermion.FermionOperator(
        ((ia, 0), (jb, 0)),
        1.0
    )
    return numpy.sqrt(0.5) * (term1 - term2)


def _Qij_vec(i: int, j: int):
    return [_Qij_plus(i, j), _Qij_minus(i, j), _Qij_0(i, j)]


def _Qij_vec_dagger(i: int, j: int):
    return [openfermion.hermitian_conjugated(i) for i in _Qij_vec(i, j)]


def _Qij_vec_inner(a: int, b: int, i: int, j: int):
    vec_dagger = _Qij_vec_dagger(a, b)
    vec = _Qij_vec(i, j)
    return sum([vec[i] * vec_dagger[i] for i in range(len(vec))])


def spin_adapted_T1(i, j):
    """
    Spin-adapted single excitation operators.

    Args:
        i (int): index of the spatial orbital which the
            creation operator will act on.
        j (int): index of the spatial orbital which the
            annihilation operator will act on.

    Returns:
        tpq_list (list): Spin-adapted single excitation operators.

    Reference:
        Scuseria, G. E. et al., J. Chem. Phys. 89, 7382 (1988)
    """
    ia = i * 2 + 0
    ib = i * 2 + 1
    ja = j * 2 + 0
    jb = j * 2 + 1
    term1 = openfermion.FermionOperator(((ia, 1), (ja, 0)), 1.0)
    term2 = openfermion.FermionOperator(((ib, 1), (jb, 0)), 1.0)
    tpq_list = [term1 + term2]
    return tpq_list


def spin_adapted_T2(creation_list, annihilation_list):
    """
    Spin-adapted double excitation operators.

    Args:
        creation_list (list): list of spatial orbital indices which the
            creation operator will act on.
        annihilation_list (list): list of spatial orbital indices which the
            annihilation operator will act on.

    Returns:
        tpqrs_list (list): Spin-adapted double excitation operators.

    Reference:
        Igor O. Sokolov et al., J. Chem. Phys. 152, 124107 (2020)
        Ireneusz W. Bulik et al., J. Chem. Theory Comput. 11, 3171−3179 (2015)
        Scuseria, G. E. et al., J. Chem. Phys. 89, 7382 (1988)
    """
    p = creation_list[0]
    r = annihilation_list[0]
    q = creation_list[1]
    s = annihilation_list[1]
    tpqrs1 = _Pij_dagger(p, q) * _Pij(r, s)
    tpqrs2 = _Qij_vec_inner(p, q, r, s)
    tpqrs_list = [tpqrs1, tpqrs2]
    return tpqrs_list


def generate_molecule_uccsd(n_orb, n_orb_occ, anti_hermitian=True):
    """
    Generate UCCSD ansatz operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.

    """

    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb_occ)]
    vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]

    T1_singles = []
    T2_doubles = []
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(len(occ_indices)):
            q = occ_indices[q_idx]

            tpq_list = spin_adapted_T1(p, q)
            for idx in range(len(tpq_list)):
                tpq = tpq_list[idx]
                if anti_hermitian:
                    tpq = tpq - openfermion.hermitian_conjugated(tpq)
                tpq = openfermion.normal_ordered(tpq)
                if (tpq.many_body_order() > 0):
                    T1_singles.append(tpq)

    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(p_idx, len(vir_indices)):
            q = vir_indices[q_idx]
            for r_idx in range(len(occ_indices)):
                r = occ_indices[r_idx]
                for s_idx in range(r_idx, len(occ_indices)):
                    s = occ_indices[s_idx]

                    tpqrs_list = spin_adapted_T2([p, q], [r, s])
                    for idx in range(len(tpqrs_list)):
                        tpqrs = tpqrs_list[idx]
                        if anti_hermitian:
                            tpqrs = tpqrs - \
                                openfermion.hermitian_conjugated(tpqrs)
                        tpqrs = openfermion.normal_ordered(tpqrs)
                        if (tpqrs.many_body_order() > 0):
                            T2_doubles.append(tpqrs)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_molecule_uccsd_symmetry_reduced(
        n_orb, n_orb_occ, anti_hermitian: bool = True,
        orbsym: numpy.ndarray = None,
        prod_table: numpy.ndarray = None):
    """
    Generate UCCSD ansatz operator pool for molecular systems and use point
    group symmetry to reduce the number of operators.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.
        orbsym (numpy.ndarray): The irreducible representation of each
            spatial orbital.
        prod_table (numpy.ndarray): The direct production table of orbsym.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.

    Note:
        Only singlet ground states are considered. If the ground state is not
        a spin-singlet state (with Ag symmetry), the result may be incorrect!

    References:
        [1]. Changsu Cao et al. Towards a Larger Molecular Simulation on the
            Quantum Computer: Up to 28 Qubits Systems Accelerated by Point
            Group Symmetry. arXiv:2109.02110

    """
    def _check_symmetry_conversing_excitations(occ_indices_spatial: list,
                                               vir_indices_spatial: list):
        """
        Assume that occ_indices_spin are taken from doubly occupied spatial
        orbitals, and vir_indices_spin are taken from zero occupied spatial
        orbitals.
        """
        unique_orbsym = list(numpy.unique(orbsym))
        t = []
        all_indices = occ_indices_spatial + vir_indices_spatial
        n_indices = len(all_indices)
        symm_result = 0
        for i in range(n_indices):
            symm_result = prod_table[symm_result][
                unique_orbsym.index(orbsym[all_indices[i]])]
        if symm_result == 0:
            return True
        return False

    if orbsym is None or prod_table is None:
        return generate_molecule_uccsd(n_orb, n_orb_occ, anti_hermitian)

    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb_occ)]
    vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]

    T1_singles = []
    T2_doubles = []
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(len(occ_indices)):
            q = occ_indices[q_idx]

            if not _check_symmetry_conversing_excitations(
                    [q], [p]):
                continue
            tpq_list = spin_adapted_T1(p, q)
            for idx in range(len(tpq_list)):
                tpq = tpq_list[idx]
                if anti_hermitian:
                    tpq = tpq - openfermion.hermitian_conjugated(tpq)
                tpq = openfermion.normal_ordered(tpq)
                if (tpq.many_body_order() > 0):
                    T1_singles.append(tpq)

    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(p_idx, len(vir_indices)):
            q = vir_indices[q_idx]
            for r_idx in range(len(occ_indices)):
                r = occ_indices[r_idx]
                for s_idx in range(r_idx, len(occ_indices)):
                    s = occ_indices[s_idx]

                    if not _check_symmetry_conversing_excitations(
                            [r, s], [p, q]):
                        continue
                    tpqrs_list = spin_adapted_T2([p, q], [r, s])
                    for idx in range(len(tpqrs_list)):
                        tpqrs = tpqrs_list[idx]
                        if anti_hermitian:
                            tpqrs = tpqrs - \
                                openfermion.hermitian_conjugated(tpqrs)
                        tpqrs = openfermion.normal_ordered(tpqrs)
                        if (tpqrs.many_body_order() > 0):
                            T2_doubles.append(tpqrs)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_molecule_uccsd_t1t2_reduced(n_orb, n_orb_occ, anti_hermitian=True,
                                         t1: numpy.ndarray = None,
                                         t1_thresh: float = 1e-3,
                                         t2: numpy.ndarray = None,
                                         t2_thresh: float = 1e-3):
    """
    Generate UCCSD ansatz operator pool for molecular systems.
    Use pre-calculated T1 or T2 to reduce the number of elements.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.

    """

    if t1 is None and t2 is None:
        return generate_molecule_uccsd(n_orb, n_orb_occ, anti_hermitian)
    else:
        print("Reducing the number of UCCSD parameters using \
provided t1 or t2.")

    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb_occ)]
    vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]

    T1_singles = []
    T2_doubles = []
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(len(occ_indices)):
            q = occ_indices[q_idx]
            if t1 is not None:
                if abs(t1[q_idx][p_idx]) < abs(t1_thresh):
                    continue

            tpq_list = spin_adapted_T1(p, q)
            for idx in range(len(tpq_list)):
                tpq = tpq_list[idx]
                if anti_hermitian:
                    tpq = tpq - openfermion.hermitian_conjugated(tpq)
                tpq = openfermion.normal_ordered(tpq)
                if (tpq.many_body_order() > 0):
                    T1_singles.append(tpq)

    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(p_idx, len(vir_indices)):
            q = vir_indices[q_idx]
            for r_idx in range(len(occ_indices)):
                r = occ_indices[r_idx]
                for s_idx in range(r_idx, len(occ_indices)):
                    s = occ_indices[s_idx]
                    if t2 is not None:
                        if abs(t2[s_idx][r_idx][q_idx][p_idx]) < \
                                abs(t2_thresh):
                            continue

                    tpqrs_list = spin_adapted_T2([p, q], [r, s])
                    for idx in range(len(tpqrs_list)):
                        tpqrs = tpqrs_list[idx]
                        if anti_hermitian:
                            tpqrs = tpqrs - \
                                openfermion.hermitian_conjugated(tpqrs)
                        tpqrs = openfermion.normal_ordered(tpqrs)
                        if (tpqrs.many_body_order() > 0):
                            T2_doubles.append(tpqrs)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_molecule_qubit4ent(n_orb: int, remove_z: bool = False,
                                ratio: float = 1.0):
    """
    Generate a QCC-like ansatz using Pauli strings with lengh smaller than 4.

    Args:
        n_orb (int): Number of spatial orbitals.
        remove_z (bool): Whether to remove the operators which contains Z.
        ratio (float): The ratio of randomly selected 4-qubit pairs.

    Returns:
        qubit4ent_operator_pool_ferm_op (list): An empty list, for
            competibility.
        qubit4ent_operator_pool_qubit_op (list): Contains qubit4ent operators.
    """
    def _convert_to_qubit_op_tuple(pauli_str_list: list):
        # Just to remove the (pos, "I") term.
        new_pauli_str_list = []
        for (pos, pauli_symbol) in pauli_str_list:
            if pauli_symbol != "I":
                new_pauli_str_list.append((pos, pauli_symbol))
        return new_pauli_str_list

    def _count_y(pauli_str_list: list):
        y_count = 0
        for (pos, pauli_symbol) in pauli_str_list:
            if pauli_symbol == "Y":
                y_count += 1
        return y_count

    def _exist_z(pauli_str_list: list):
        for (pos, pauli_symbol) in pauli_str_list:
            if pauli_symbol == "Z":
                return True
        return False
    qubit4ent_operator_pool_ferm_op = []
    qubit4ent_operator_pool_qubit_op = []
    orb_idx_list = []
    for p in range(n_orb * 2):
        for q in range(p, n_orb * 2):
            for r in range(q, n_orb * 2):
                for s in range(q, n_orb * 2):
                    if (numpy.random.rand(1) <= ratio):
                        orb_idx_list.append([p, q, r, s])
    for (p, q, r, s) in orb_idx_list:
        for (p1, p2, p3, p4) in itertools.product(*[["X", "Y", "Z", "I"]] * 4):
            pauli_str_list = _convert_to_qubit_op_tuple(
                [(p, p1), (q, p2), (r, p3), (s, p4)])
            qubit_op = openfermion.QubitOperator(pauli_str_list, 1.0)
            pauli_str_list = list(qubit_op.terms.keys())[0]
            if _count_y(pauli_str_list) % 2 == 0:
                # Remove operators with even number of Ys.
                continue
            if remove_z:
                if _exist_z(pauli_str_list):
                    continue
            qubit_op = openfermion.QubitOperator(
                pauli_str_list, 1.0)
            if openfermion.hermitian_conjugated(qubit_op) == qubit_op:
                qubit_op = qubit_op * 1.j
            if (qubit_op.many_body_order() > 0):
                qubit4ent_operator_pool_qubit_op.append(qubit_op)
    qubit4ent_operator_pool_qubit_op_str = [
        term.__str__() for term in qubit4ent_operator_pool_qubit_op]
    qubit4ent_operator_pool_qubit_op_str_set = set(
        qubit4ent_operator_pool_qubit_op_str)
    qubit4ent_operator_pool_qubit_op_new = [qubit4ent_operator_pool_qubit_op[
        qubit4ent_operator_pool_qubit_op_str.index(i)]
        for i in qubit4ent_operator_pool_qubit_op_str_set]
    qubit4ent_operator_pool_qubit_op = qubit4ent_operator_pool_qubit_op_new
    return qubit4ent_operator_pool_ferm_op, qubit4ent_operator_pool_qubit_op


def _generate_molecule_qubit4ent_mp_worker(args):
    proc_idx = args[0]
    n_workers = args[1]
    n_orb = args[2]
    remove_z = args[3]
    ratio = args[4]
    occupied_spin_orbs = args[5]
    vir_spin_orbs = None
    if occupied_spin_orbs is not None:
        vir_spin_orbs = [i for i in range(n_orb * 2)
                         if i not in occupied_spin_orbs]

    def _convert_to_qubit_op_tuple(pauli_str_list: list):
        # Just to remove the (pos, "I") term.
        new_pauli_str_list = []
        for (pos, pauli_symbol) in pauli_str_list:
            if pauli_symbol != "I":
                new_pauli_str_list.append((pos, pauli_symbol))
        return new_pauli_str_list

    def _count_y(pauli_str_list: list):
        y_count = 0
        for (pos, pauli_symbol) in pauli_str_list:
            if pauli_symbol == "Y":
                y_count += 1
        return y_count

    def _exist_z(pauli_str_list: list):
        for (pos, pauli_symbol) in pauli_str_list:
            if pauli_symbol == "Z":
                return True
        return False
    qubit4ent_operator_pool_qubit_op_set = set()
    len_set = len(qubit4ent_operator_pool_qubit_op_set)
    qubit4ent_operator_pool_qubit_op = []
    orb_idx_list = []
    for p in range(proc_idx, n_orb * 2, n_workers):
        for q in range(p, n_orb * 2):
            for r in range(q, n_orb * 2):
                for s in range(q, n_orb * 2):
                    if (numpy.random.rand(1) <= ratio):
                        orb_idx_list.append([p, q, r, s])
    for (p, q, r, s) in orb_idx_list:
        for (p1, p2, p3, p4) in itertools.product(*[["X", "Y", "Z", "I"]] * 4):
            pauli_str_list = _convert_to_qubit_op_tuple(
                [(p, p1), (q, p2), (r, p3), (s, p4)])
            qubit_op = openfermion.QubitOperator(pauli_str_list, 1.0)
            pauli_str_list = list(qubit_op.terms.keys())[0]
            if _count_y(pauli_str_list) % 2 == 0:
                # Remove operators with even number of Ys.
                continue
            if remove_z:
                if _exist_z(pauli_str_list):
                    continue

            if occupied_spin_orbs is not None:
                # Then we generate the pseudo-conserving operator
                n_pauli_occ = 0
                n_pauli_vir = 0
                for (pos, pauli_symbol) in pauli_str_list :
                    if pos in occupied_spin_orbs:
                        n_pauli_occ += 1
                    elif pos in vir_spin_orbs:
                        n_pauli_vir += 1
                if n_pauli_occ != n_pauli_vir:
                    continue

            qubit_op = openfermion.QubitOperator(
                pauli_str_list, 1.0)
            if openfermion.hermitian_conjugated(qubit_op) == qubit_op:
                qubit_op = qubit_op * 1.j
            if (qubit_op.many_body_order() > 0):
                qubit4ent_operator_pool_qubit_op_set.add(qubit_op.__str__())
                if len(qubit4ent_operator_pool_qubit_op_set) > len_set:
                    len_set = len(qubit4ent_operator_pool_qubit_op_set)
                    qubit4ent_operator_pool_qubit_op.append(qubit_op)

    return qubit4ent_operator_pool_qubit_op


def generate_molecule_qubit4ent_mp(
        n_orb: int, remove_z: bool = False, n_procs: int = 1,
        ratio: float = 1.0,
        occupied_spin_orbs: list = None):
    """
    Generate a QCC-like ansatz using Pauli strings with lengh smaller than 4.

    Args:
        n_orb (int): Number of spatial orbitals.
        remove_z (bool): Whether to remove the operators which contains Z.
        ratio (float): The ratio of randomly selected 4-qubit pairs.
        occupied_spin_orbs (list): Occupied spin orbitals to generate  pseudo-
            particle-conserving operators. Set to None if particle-conserving
            is not required.

    Returns:
        qubit4ent_operator_pool_ferm_op (list): An empty list,
            for competibility.
        qubit4ent_operator_pool_qubit_op (list): Contains qubit4ent operators.
    """
    if ratio < 0.:
        ratio = 0.
    n_terms = 2 * n_orb
    n_workers = min(n_terms, n_procs)
    if (n_workers != n_procs):
        print("Warning: change n_procs to %d" % (n_workers))

    chunk_size = n_terms // n_workers
    chunk_list = [chunk_size for i in range(n_workers)]
    for i in range(n_terms - chunk_size * n_workers):
        chunk_list[i] += 1

    import multiprocessing
    args_workers = []
    start_idx = 0
    end_idx = 0
    for i in range(n_workers):
        start_idx = end_idx
        end_idx += chunk_list[i]
        args_workers.append((i, n_workers, n_orb, remove_z, ratio,
                             occupied_spin_orbs))

    Pool = multiprocessing.Pool(n_workers)
    map_result = Pool.map(_generate_molecule_qubit4ent_mp_worker, args_workers)
    Pool.close()
    Pool.join()

    qubit4ent_operator_pool_ferm_op = []
    qubit4ent_operator_pool_qubit_op = []
    for i in range(n_workers):
        qubit4ent_operator_pool_qubit_op += map_result[i]

    qubit4ent_operator_pool_qubit_op_set = set()
    qubit4ent_operator_pool_qubit_op_new = []
    len_set = len(qubit4ent_operator_pool_qubit_op_set)

    for qubit_op in qubit4ent_operator_pool_qubit_op:
        qubit4ent_operator_pool_qubit_op_set.add(qubit_op.__str__())
        if len(qubit4ent_operator_pool_qubit_op_set) > len_set:
            len_set = len(qubit4ent_operator_pool_qubit_op_set)
            qubit4ent_operator_pool_qubit_op_new.append(qubit_op)

    qubit4ent_operator_pool_qubit_op = qubit4ent_operator_pool_qubit_op_new

    return qubit4ent_operator_pool_ferm_op, qubit4ent_operator_pool_qubit_op


def generate_molecule_uccgsd(n_orb, n_orb_occ,
                             anti_hermitian: bool = True):
    """
    Generate UCCGSD ansatz operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCGSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCGSD Qubit operators under JW transformation.
    """

    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb)]
    vir_indices = [i for i in range(n_orb)]

    T1_singles = []
    T2_doubles = []
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(len(occ_indices)):
            q = occ_indices[q_idx]

            tpq_list = spin_adapted_T1(p, q)
            for idx in range(len(tpq_list)):
                tpq = tpq_list[idx]
                if anti_hermitian:
                    tpq = tpq - openfermion.hermitian_conjugated(tpq)
                tpq = openfermion.normal_ordered(tpq)
                if (tpq.many_body_order() > 0):
                    T1_singles.append(tpq)

    pq = -1
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(p_idx, len(vir_indices)):
            q = vir_indices[q_idx]
            pq += 1
            rs = -1
            for r_idx in range(len(occ_indices)):
                r = occ_indices[r_idx]
                for s_idx in range(r_idx, len(occ_indices)):
                    s = occ_indices[s_idx]
                    rs += 1
                    if (pq > rs):
                        continue

                    tpqrs_list = spin_adapted_T2([p, q], [r, s])
                    for idx in range(len(tpqrs_list)):
                        tpqrs = tpqrs_list[idx]
                        if anti_hermitian:
                            tpqrs = tpqrs - \
                                openfermion.hermitian_conjugated(tpqrs)
                        tpqrs = openfermion.normal_ordered(tpqrs)
                        if (tpqrs.many_body_order() > 0):
                            T2_doubles.append(tpqrs)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_molecule_uccgsd_symmetry_reduced(
        n_orb, n_orb_occ, anti_hermitian: bool = True,
        orbsym: numpy.ndarray = None,
        prod_table: numpy.ndarray = None):
    """
    Generate UCCGSD ansatz operator pool for molecular systems and use point
    group symmetry to reduce the number of operators.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.
        orbsym (numpy.ndarray): The irreducible representation of each
            spatial orbital.
        prod_table (numpy.ndarray): The direct production table of orbsym.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCGSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCGSD Qubit operators under JW transformation.

    Note:
        Only singlet ground states are considered. If the ground state is not
        a spin-singlet state (with Ag symmetry), the result may be incorrect!

    References:
        [1]. Changsu Cao et al. Towards a Larger Molecular Simulation on the
            Quantum Computer: Up to 28 Qubits Systems Accelerated by Point
            Group Symmetry. arXiv:2109.02110
    """
    def _check_symmetry_conversing_excitations(occ_indices_spatial: list,
                                               vir_indices_spatial: list):
        """
        Assume that occ_indices_spin are taken from doubly occupied spatial
        orbitals, and vir_indices_spin are taken from zero occupied spatial
        orbitals.
        """
        unique_orbsym = list(numpy.unique(orbsym))
        t = []
        all_indices = occ_indices_spatial + vir_indices_spatial
        n_indices = len(all_indices)
        symm_result = 0
        for i in range(n_indices):
            symm_result = prod_table[symm_result][
                unique_orbsym.index(orbsym[all_indices[i]])]
        if symm_result == 0:
            return True
        return False

    if orbsym is None or prod_table is None:
        return generate_molecule_uccgsd(n_orb, n_orb_occ, anti_hermitian)

    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb)]
    vir_indices = [i for i in range(n_orb)]

    T1_singles = []
    T2_doubles = []
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(len(occ_indices)):
            q = occ_indices[q_idx]

            if not _check_symmetry_conversing_excitations(
                    [q], [p]):
                continue

            tpq_list = spin_adapted_T1(p, q)
            for idx in range(len(tpq_list)):
                tpq = tpq_list[idx]
                if anti_hermitian:
                    tpq = tpq - openfermion.hermitian_conjugated(tpq)
                tpq = openfermion.normal_ordered(tpq)
                if (tpq.many_body_order() > 0):
                    T1_singles.append(tpq)

    pq = -1
    for p_idx in range(len(vir_indices)):
        p = vir_indices[p_idx]
        for q_idx in range(p_idx, len(vir_indices)):
            q = vir_indices[q_idx]
            pq += 1
            rs = -1
            for r_idx in range(len(occ_indices)):
                r = occ_indices[r_idx]
                for s_idx in range(r_idx, len(occ_indices)):
                    s = occ_indices[s_idx]

                    if not _check_symmetry_conversing_excitations(
                            [r, s], [p, q]):
                        continue

                    rs += 1
                    if (pq > rs):
                        continue

                    tpqrs_list = spin_adapted_T2([p, q], [r, s])
                    for idx in range(len(tpqrs_list)):
                        tpqrs = tpqrs_list[idx]
                        if anti_hermitian:
                            tpqrs = tpqrs - \
                                openfermion.hermitian_conjugated(tpqrs)
                        tpqrs = openfermion.normal_ordered(tpqrs)
                        if (tpqrs.many_body_order() > 0):
                            T2_doubles.append(tpqrs)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def _verify_kconserv(kpts,
                     k_idx_creation, k_idx_annihilation,
                     lattice_vec):
    """
    Helper function to check momentum conservation condition.

    Args:
        kpts (numpy.ndarray): Coordinates of k-points in reciprocal space.
        k_idx_creation (list): Indices of k-points which
            corresponds to creation operators.
        k_idx_annihilation (list): Indices of k-points which
            corresponds to annihilation operators.
        lattice_vec (numpy.ndarray): Lattice vectors.
    """
    sum_kpts = numpy.zeros(kpts[0].shape)
    for a in range(len(k_idx_creation)):
        sum_kpts += kpts[k_idx_creation[a]]
    for i in range(len(k_idx_annihilation)):
        sum_kpts -= kpts[k_idx_annihilation[i]]
    dots = numpy.dot(sum_kpts, lattice_vec / (2 * numpy.pi))
    """
    Every element in dots should be int if kconserv.
    """
    if ((abs(numpy.rint(dots) - dots) <= 1e-8).nonzero()[0].shape[0] == 3):
        return True
    return False


def generate_pbc_uccsd(n_orb, n_orb_occ,
                       kpts, m2k, lattice_vec,
                       complementary_pool=False,
                       anti_hermitian=True):
    """
    Generate UCCSD ansatz operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        complementary_pool (bool): Whether to add complementary terms
            in the operator pool.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.
    """
    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb_occ)]
    vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]

    T1_singles = []
    T2_doubles = []
    # Complementary operator pool
    T1_singles_c = []
    T2_doubles_c = []
    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(len(occ_indices)):
            q = occ_indices[q_spatial]
            kq_idx = m2k[q][0]

            if (_verify_kconserv(kpts, [kp_idx], [kq_idx],
                                 lattice_vec) is True):
                tpq_list = spin_adapted_T1(p, q)
                for idx in range(len(tpq_list)):
                    tpq = tpq_list[idx]
                    tpq_c = 1.j * (tpq + openfermion.hermitian_conjugated(tpq))
                    if anti_hermitian:
                        tpq = tpq - openfermion.hermitian_conjugated(tpq)
                    tpq = openfermion.normal_ordered(tpq)
                    tpq_c = openfermion.normal_ordered(tpq_c)
                    if (tpq.many_body_order() > 0):
                        T1_singles.append(tpq)
                    if (tpq_c.many_body_order() > 0):
                        T1_singles_c.append(tpq_c)

    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(p_spatial, len(vir_indices)):
            q = vir_indices[q_spatial]
            kq_idx = m2k[q][0]
            for r_spatial in range(len(occ_indices)):
                r = occ_indices[r_spatial]
                kr_idx = m2k[r][0]
                for s_spatial in range(r_spatial, len(occ_indices)):
                    s = occ_indices[s_spatial]
                    ks_idx = m2k[s][0]

                    if (_verify_kconserv(kpts,
                                         [kp_idx, kq_idx], [kr_idx, ks_idx],
                                         lattice_vec) is True):
                        tpqrs_list = spin_adapted_T2([p, q], [r, s])
                        for idx in range(len(tpqrs_list)):
                            tpqrs = tpqrs_list[idx]
                            tpqrs_c = 1.j * \
                                (tpqrs +
                                 openfermion.hermitian_conjugated(tpqrs))
                            if anti_hermitian:
                                tpqrs = tpqrs - \
                                    openfermion.hermitian_conjugated(tpqrs)
                            tpqrs = openfermion.normal_ordered(tpqrs)
                            tpqrs_c = openfermion.normal_ordered(tpqrs_c)
                            if (tpqrs.many_body_order() > 0):
                                T2_doubles.append(tpqrs)
                            if (tpqrs_c.many_body_order() > 0):
                                T2_doubles_c.append(tpqrs_c)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    if (complementary_pool is True):
        uccsd_operator_pool_fermOp += (T1_singles_c + T2_doubles_c)

    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_pbc_uccgsd(n_orb, n_orb_occ,
                        kpts, m2k, lattice_vec,
                        complementary_pool=False,
                        anti_hermitian=True):
    """
    Generate UCCGSD ansatz operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        complementary_pool (bool): Whether to add complementary terms
            in the operator pool.
        anti_hermitian (bool): Whether to substract the hermitian conjugate.

    Returns:
        uccsd_operator_pool_fermOp (list):
            UCCSD Fermionic operators.
        uccsd_operator_pool_QubitOp (list):
            UCCSD Qubit operators under JW transformation.

    """
    n_orb_vir = n_orb - n_orb_occ
    occ_indices = [i for i in range(n_orb)]
    vir_indices = [i for i in range(n_orb)]

    T1_singles = []
    T2_doubles = []
    # Complementary operator pool
    T1_singles_c = []
    T2_doubles_c = []
    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(len(occ_indices)):
            q = occ_indices[q_spatial]
            kq_idx = m2k[q][0]

            if (_verify_kconserv(kpts, [kp_idx], [kq_idx],
                                 lattice_vec) is True):
                tpq_list = spin_adapted_T1(p, q)
                for idx in range(len(tpq_list)):
                    tpq = tpq_list[idx]
                    tpq_c = 1.j * (tpq + openfermion.hermitian_conjugated(tpq))
                    if anti_hermitian:
                        tpq = tpq - openfermion.hermitian_conjugated(tpq)
                    tpq = openfermion.normal_ordered(tpq)
                    tpq_c = openfermion.normal_ordered(tpq_c)
                    if (tpq.many_body_order() > 0):
                        T1_singles.append(tpq)
                    if (tpq_c.many_body_order() > 0):
                        T1_singles_c.append(tpq_c)

    pq = -1
    for p_spatial in range(len(vir_indices)):
        p = vir_indices[p_spatial]
        kp_idx = m2k[p][0]
        for q_spatial in range(p_spatial, len(vir_indices)):
            q = vir_indices[q_spatial]
            kq_idx = m2k[q][0]
            pq += 1
            rs = -1
            for r_spatial in range(len(occ_indices)):
                r = occ_indices[r_spatial]
                kr_idx = m2k[r][0]
                for s_spatial in range(r_spatial, len(occ_indices)):
                    s = occ_indices[s_spatial]
                    ks_idx = m2k[s][0]
                    rs += 1
                    if (pq > rs):
                        continue

                    if (_verify_kconserv(kpts,
                                         [kp_idx, kq_idx], [kr_idx, ks_idx],
                                         lattice_vec) is True):
                        tpqrs_list = spin_adapted_T2([p, q], [r, s])
                        for idx in range(len(tpqrs_list)):
                            tpqrs = tpqrs_list[idx]
                            tpqrs_c = 1.j * \
                                (tpqrs +
                                 openfermion.hermitian_conjugated(tpqrs))
                            if anti_hermitian:
                                tpqrs = tpqrs - \
                                    openfermion.hermitian_conjugated(tpqrs)
                            tpqrs = openfermion.normal_ordered(tpqrs)
                            tpqrs_c = openfermion.normal_ordered(tpqrs_c)
                            if (tpqrs.many_body_order() > 0):
                                T2_doubles.append(tpqrs)
                            if (tpqrs_c.many_body_order() > 0):
                                T2_doubles_c.append(tpqrs_c)

    uccsd_operator_pool_fermOp = T1_singles + T2_doubles
    if (complementary_pool is True):
        uccsd_operator_pool_fermOp += (T1_singles_c + T2_doubles_c)

    uccsd_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in uccsd_operator_pool_fermOp]

    return uccsd_operator_pool_fermOp, uccsd_operator_pool_QubitOp


def generate_molecule_eomip(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMIP operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomip_operator_pool_fermOp (list):
            EOMIP Fermionic operators.
        eomip_operator_pool_QubitOp (list):
            EOMIP Qubit operators under JW transformation.
    """
    n_orb_vir = n_orb - n_orb_occ

    IP1_singles = []
    IP2_doubles = []
    for i in range(n_orb_occ):
        ia = 2 * i
        ib = 2 * i + 1
        ri = openfermion.FermionOperator(
            ((ia, 0)),
            1.
        )
        IP1_singles.append(ri)
        if deexcitation:
            IP1_singles.append(openfermion.hermitian_conjugated(ri))

    for i in range(n_orb_occ):
        ia = 2 * i
        ib = 2 * i + 1
        for j in range(n_orb_occ):
            ja = 2 * j
            jb = 2 * j + 1
            for b in range(n_orb_vir):
                ba = 2 * n_orb_occ + 2 * b
                bb = 2 * n_orb_occ + 2 * b + 1
                rbji = openfermion.FermionOperator(
                    ((ba, 1), (ja, 0), (ia, 0)),
                    1. / 2.
                )
                rbji += openfermion.FermionOperator(
                    ((bb, 1), (jb, 0), (ia, 0)),
                    1. / 2.
                )
                IP2_doubles.append(rbji)
                if deexcitation:
                    IP2_doubles.append(openfermion.hermitian_conjugated(rbji))

    eomip_operator_pool_fermOp = IP1_singles + IP2_doubles
    eomip_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomip_operator_pool_fermOp]
    return eomip_operator_pool_fermOp, eomip_operator_pool_QubitOp


def generate_molecule_eomea(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMEA operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomea_operator_pool_fermOp (list):
            EOMEA Fermionic operators.
        eomea_operator_pool_QubitOp (list):
            EOMEA Qubit operators under JW transformation.
    """
    n_orb_vir = n_orb - n_orb_occ
    EA1_singles = []
    EA2_doubles = []
    for a in range(n_orb_vir):
        aa = 2 * n_orb_occ + 2 * a
        ab = 2 * n_orb_occ + 2 * a + 1
        ra = openfermion.FermionOperator(
            ((aa, 1)),
            1.
        )
        EA1_singles.append(ra)
        if deexcitation:
            EA1_singles.append(openfermion.hermitian_conjugated(ra))

    for a in range(n_orb_vir):
        aa = 2 * n_orb_occ + 2 * a
        ab = 2 * n_orb_occ + 2 * a + 1
        for b in range(n_orb_vir):
            ba = 2 * n_orb_occ + 2 * b
            bb = 2 * n_orb_occ + 2 * b + 1
            for j in range(n_orb_occ):
                ja = 2 * j
                jb = 2 * j + 1
                rabj = openfermion.FermionOperator(
                    ((aa, 1), (ba, 1), (ja, 0)),
                    1. / 2.
                )
                rabj += openfermion.FermionOperator(
                    ((aa, 1), (bb, 1), (jb, 0)),
                    1. / 2.
                )
                EA2_doubles.append(rabj)
                if deexcitation:
                    EA2_doubles.append(openfermion.hermitian_conjugated(rabj))

    eomea_operator_pool_fermOp = EA1_singles + EA2_doubles
    eomea_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomea_operator_pool_fermOp]
    return eomea_operator_pool_fermOp, eomea_operator_pool_QubitOp


def generate_molecule_eomee(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMEE operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomee_operator_pool_fermOp (list):
            EOMEE Fermionic operators.
        eomee_operator_pool_QubitOp (list):
            EOMEE Qubit operators under JW transformation.
    """
    eomee_operator_pool_fermOp_, eomee_operator_pool_QubitOp_ = \
        generate_molecule_uccsd(n_orb, n_orb_occ, anti_hermitian=False)
    n_terms = len(eomee_operator_pool_fermOp_)
    eomee_operator_pool_fermOp = []
    eomee_operator_pool_QubitOp = []
    for i in range(n_terms):
        fermOp_i = eomee_operator_pool_fermOp_[i]
        qubitOp_i = eomee_operator_pool_QubitOp_[i]
        eomee_operator_pool_fermOp.append(fermOp_i)
        eomee_operator_pool_QubitOp.append(qubitOp_i)
        if deexcitation:
            eomee_operator_pool_fermOp.append(
                openfermion.hermitian_conjugated(fermOp_i))
            eomee_operator_pool_QubitOp.append(
                openfermion.hermitian_conjugated(qubitOp_i))
    return eomee_operator_pool_fermOp, eomee_operator_pool_QubitOp


def generate_molecule_eomee_unrestricted(n_orb, n_orb_occ, deexcitation=False):
    """
    Generate EOMEE operator pool for molecular systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomee_operator_pool_fermOp (list):
            EOMEE Fermionic operators.
        eomee_operator_pool_QubitOp (list):
            EOMEE Qubit operators under JW transformation.
    """
    n_qubits = n_orb * 2
    occ_indices_spin = [i for i in range(2 * n_orb_occ)]
    vir_indices_spin = [i + 2 * n_orb_occ
                        for i in range(n_qubits - 2 * n_orb_occ)]
    EE1_singles = []
    EE2_doubles = []
    for a in vir_indices_spin:
        for i in occ_indices_spin:
            rai = openfermion.FermionOperator(
                ((a, 1), (i, 0)),
                1.
            )
            rai = openfermion.normal_ordered(rai)
            if rai.many_body_order() > 0:
                EE1_singles.append(rai)
                if deexcitation:
                    EE1_singles.append(openfermion.hermitian_conjugated(rai))

    for a in vir_indices_spin:
        for b in vir_indices_spin:
            if vir_indices_spin.index(b) <= vir_indices_spin.index(a):
                continue
            for i in occ_indices_spin:
                for j in occ_indices_spin:
                    if occ_indices_spin.index(j) <= occ_indices_spin.index(i):
                        continue
                    rabij = openfermion.FermionOperator(
                        ((a, 1), (b, 1), (i, 0), (j, 0)),
                        1.
                    )
                    rabij = openfermion.normal_ordered(rabij)
                    if rabij.many_body_order() > 0:
                        EE2_doubles.append(rabij)
                        if deexcitation:
                            EE2_doubles.append(
                                openfermion.hermitian_conjugated(rabij))

    eomee_operator_pool_fermOp = EE1_singles + EE2_doubles
    eomee_operator_pool_QubitOp = [openfermion.jordan_wigner(i)
                                   for i in eomee_operator_pool_fermOp]
    return eomee_operator_pool_fermOp, eomee_operator_pool_QubitOp


def generate_pbc_eomip(n_orb, n_orb_occ,
                       kpts, m2k, lattice_vec,
                       deexcitation=False):
    """
    Generate EOMIP operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomip_operator_pool_fermOp (list):
            EOMIP Fermionic operators.
        eomip_operator_pool_QubitOp (list):
            EOMIP Qubit operators under JW transformation.
    """
    kshift = 0
    n_orb_vir = n_orb - n_orb_occ

    IP1_singles = []
    IP2_doubles = []
    for i in range(n_orb_occ):
        ki_idx = m2k[i][0]
        if (ki_idx != kshift):
            continue
        ia = 2 * i
        ib = 2 * i + 1
        ri = openfermion.FermionOperator(
            ((ia, 0)),
            1.
        )
        IP1_singles.append(ri)
        if deexcitation:
            IP1_singles.append(openfermion.hermitian_conjugated(ri))

    for i in range(n_orb_occ):
        ki_idx = m2k[i][0]
        ia = 2 * i
        ib = 2 * i + 1
        for j in range(n_orb_occ):
            kj_idx = m2k[j][0]
            ja = 2 * j
            jb = 2 * j + 1
            for b in range(n_orb_vir):
                kb_idx = m2k[n_orb_occ + b][0]
                ba = 2 * (n_orb_occ + b)
                bb = 2 * (n_orb_occ + b) + 1
                if (_verify_kconserv(
                        kpts, [kb_idx, kshift], [ki_idx, kj_idx],
                        lattice_vec) is True):
                    rbji = openfermion.FermionOperator(
                        ((ba, 1), (ja, 0), (ia, 0)),
                        1. / 2.
                    )
                    rbji += openfermion.FermionOperator(
                        ((bb, 1), (jb, 0), (ia, 0)),
                        1. / 2.
                    )
                    IP2_doubles.append(rbji)
                    if deexcitation:
                        IP2_doubles.append(
                            openfermion.hermitian_conjugated(rbji))

    eomip_operator_pool_fermOp = IP1_singles + IP2_doubles
    eomip_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomip_operator_pool_fermOp]
    return eomip_operator_pool_fermOp, eomip_operator_pool_QubitOp


def generate_pbc_eomea(n_orb, n_orb_occ,
                       kpts, m2k, lattice_vec,
                       deexcitation=False):
    """
    Generate EOMEA operator pool for periodic systems.

    Args:
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        kpts (numpy.ndarray): Coordinates of k-points.
        m2k: m2k returned by init_scf_pbc()
        lattice_vec (numpy.ndarray): Lattice vectors.
        deexcitation (bool): Whether to include deexcitation operators.

    Returns:
        eomea_operator_pool_fermOp (list):
            EOMEA Fermionic operators.
        eomea_operator_pool_QubitOp (list):
            EOMEA Qubit operators under JW transformation.
    """
    kshift = 0
    n_orb_vir = n_orb - n_orb_occ

    EA1_singles = []
    EA2_doubles = []
    for a in range(n_orb_vir):
        ka_idx = m2k[n_orb_occ + a][0]
        if (ka_idx != kshift):
            continue
        aa = 2 * (n_orb_occ + a)
        ab = 2 * (n_orb_occ + a) + 1
        ra = openfermion.FermionOperator(
            ((aa, 1)),
            1.
        )
        EA1_singles.append(ra)
        if deexcitation:
            EA1_singles.append(openfermion.hermitian_conjugated(ra))

    for a in range(n_orb_vir):
        ka_idx = m2k[n_orb_occ + a][0]
        aa = 2 * (n_orb_occ + a)
        ab = 2 * (n_orb_occ + a) + 1
        for b in range(n_orb_vir):
            kb_idx = m2k[n_orb_occ + b][0]
            ba = 2 * (n_orb_occ + b)
            bb = 2 * (n_orb_occ + b) + 1
            for j in range(n_orb_occ):
                kj_idx = m2k[j][0]
                ja = 2 * j
                jb = 2 * j + 1
                if (_verify_kconserv(
                    kpts, [ka_idx, kb_idx], [kj_idx, kshift],
                        lattice_vec) is True):
                    rabj = openfermion.FermionOperator(
                        ((aa, 1), (ba, 1), (ja, 0)),
                        1. / 2.
                    )
                    rabj += openfermion.FermionOperator(
                        ((aa, 1), (bb, 1), (jb, 0)),
                        1. / 2.
                    )
                    EA2_doubles.append(rabj)
                    if deexcitation:
                        EA2_doubles.append(
                            openfermion.hermitian_conjugated(rabj))

    eomea_operator_pool_fermOp = EA1_singles + EA2_doubles
    eomea_operator_pool_QubitOp = [openfermion.jordan_wigner(op)
                                   for op in eomea_operator_pool_fermOp]
    return eomea_operator_pool_fermOp, eomea_operator_pool_QubitOp


def qubit_adapt_pool(n: int):
    operator_pool = []
    assert(n >= 3)
    if (n == 3):
        operator_pool.append(
            openfermion.QubitOperator(((2, "Z"), (1, "Z"), (0, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((2, "Z"), (1, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((2, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((1, "Y"),), 1.)
        )
    else:
        operator_pool_ = qubit_adapt_pool(n - 1)
        for i in operator_pool_:
            operator_pool.append(
                openfermion.QubitOperator(((n - 1, "Z"),), 1.) * i
            )
        operator_pool.append(
            openfermion.QubitOperator(((n - 1, "Y"),), 1.)
        )
        operator_pool.append(
            openfermion.QubitOperator(((n - 2, "Y"),), 1.)
        )
    return operator_pool


def convert_to_qubit_adaptvqe_pool(operator_pool_qubitOp: list):
    qubit_adaptvqe_operator_pool = []
    for i in range(len(operator_pool_qubitOp)):
        qubitOp_i = operator_pool_qubitOp[i]
        for term, coeff in qubitOp_i.terms.items():
            term_list = list(term)
            term_list_new = [i for i in term_list if i[1] != "Z"]
            term_new = tuple(term_list_new)
            y_count = 0
            for term_new_term in term_new:
                if term_new_term[1] == "Y":
                    y_count += 1
            if y_count % 2 == 0:
                continue
            qubit_adaptvqe_operator_pool.append(
                openfermion.QubitOperator(term_new, coeff)
            )
    return qubit_adaptvqe_operator_pool


def _qubit_excit(i: int):
    qubitOp1 = openfermion.QubitOperator(
        ((i, "X"),), 1.
    )
    qubitOp2 = openfermion.QubitOperator(
        ((i, "Y"),), 1.j
    )
    qubitOp = (qubitOp1 - qubitOp2) * 0.5
    return qubitOp


def _qubit_deexcit(i: int):
    qubitOp1 = openfermion.QubitOperator(
        ((i, "X"),), 1.
    )
    qubitOp2 = openfermion.QubitOperator(
        ((i, "Y"),), 1.j
    )
    qubitOp = (qubitOp1 + qubitOp2) * 0.5
    return qubitOp


def QubitExcitationOperator(term: tuple, coeff: complex):
    """
    Create Qubit excitation operators. Takes the same arguments as
        openfermion's FermionOperator.

    Definition:
        Q^+_n = 1/2 (X_n - iY_n)
        Q_n = 1/2 (X_n + iY_n)

    Returns:
        qubitOp (openfermion.QubitOperator): Qubit excitation operators
            represented by openfermion's QubitOperator

    Examples:
        >>> import openfermion
        >>> from utils import QubitExcitationOperator
        >>> qubitEOp = QubitExcitationOperator(((2, 0), (0, 1)), 2.j)
        >>> fermOp = openfermion.FermionOperator(((2, 0), (0, 1)), 2.j)
        >>> qubitOp = openfermion.jordan_wigner(fermOp)
        >>> qubitEOp
        0.5j [X0 X2] +
        (-0.5+0j) [X0 Y2] +
        (0.5+0j) [Y0 X2] +
        0.5j [Y0 Y2]
        >>> qubitOp
        -0.5j [X0 Z1 X2] +
        (0.5+0j) [X0 Z1 Y2] +
        (-0.5+0j) [Y0 Z1 X2] +
        -0.5j [Y0 Z1 Y2]
    """
    qubitEOp = openfermion.QubitOperator((), 1.)
    for term_i in term:
        qubitEOp_i = _qubit_excit(term_i[0]) if term_i[1] == 1 \
            else _qubit_deexcit(term_i[0])
        qubitEOp *= qubitEOp_i
    qubitEOp *= coeff
    return qubitEOp


def convert_fermOp_to_qubitEOp(fermOp_list):
    if (isinstance(fermOp_list, openfermion.FermionOperator)):
        qubitEOp = openfermion.QubitOperator()
        for term, coeff in fermOp_list.terms.items():
            qubitEOp += \
                QubitExcitationOperator(
                    term,
                    coeff
                )
        return qubitEOp
    assert(isinstance(fermOp_list, list))
    qubitEOp_list = []
    for fermOp_i in fermOp_list:
        qubitEOp_i = openfermion.QubitOperator()
        for term, coeff in fermOp_i.terms.items():
            qubitEOp_i += \
                QubitExcitationOperator(
                    term,
                    coeff
                )
        qubitEOp_list.append(qubitEOp_i)
    return qubitEOp_list


def geigh(H: numpy.ndarray, S: numpy.ndarray, sort_eigs=False):
    """
    Solve the generalized eigenvalue problem HC=SCE
    """
    import scipy
    if (numpy.linalg.norm(H - H.T.conj()) > 1e-9):
        print("WARNING: H not hermitian !")
        print("Norm of H - H.T.conj(): %20.16f" %
              (numpy.linalg.norm(H - H.T.conj())))
        print("AbsMax of H - H.T.conj(): %20.16f" %
              (numpy.abs(H - H.T.conj()).max()))
    if (numpy.linalg.norm(S - S.T.conj()) > 1e-9):
        print("WARNING: S not hermitian !")
        print("Norm of S - S.T.conj(): %20.16f" %
              (numpy.linalg.norm(S - S.T.conj())))
        print("AbsMax of S - S.T.conj(): %20.16f" %
              (numpy.abs(S - S.T.conj()).max()))
    D = None
    V = None
    try:
        D, V = scipy.linalg.eigh(H, S)
    except numpy.linalg.LinAlgError:
        try:
            print("WARNING: scipy's built in eigh() failed. Try using SVD.")
            U, S0, Vh = numpy.linalg.svd(S)
            U1 = U[:, numpy.abs(S0) > 0.0]
            T = (U1.T.conj()).dot(H.dot(U1))
            G = numpy.diag(S0[: U1.shape[1]])
            D, V = scipy.linalg.eig(T, G)
            V = U1.dot(V.dot(U1.T.conj()))
        except numpy.linalg.LinAlgError:
            print("SVD failed. Using bare eig(). The calculated \
eigenvalues and eigenvectors may have duplicates!")
            D, V = scipy.linalg.eig(H, S)
        else:
            pass
    else:
        pass
    V = V / numpy.linalg.norm(V, axis=0)
    if (sort_eigs is True):
        idx_sorted = D.real.argsort()
        return D[idx_sorted], V[idx_sorted]
    else:
        return D, V


def decompose_trottered_qubit_op(qubit_op: openfermion.QubitOperator,
                                 coeff_idx: int):
    """
    Simulate exp(i * (qubit_op / i) * coeff). coeff is some external
    coefficient. qubit_op may contain multiple terms. The i is
    contained in the qubit_op.

    Args:
        qubit_op (openfermion.QubitOperator): The qubit operator which
            is put on the exponential term.
        coeff_idx: The index of the coefficient.

    Returns:
        gates_apply (list): A list of gates which is to apply on the qubits.
            The order is:
            gates_apply[N]...gates_apply[1] gates_apply[0] |psi>

    Notes:
        HY = 2 ** 0.5 / 2. * numpy.array(
            [[1.0, -1.j],
             [1.j, -1.0]])

    Notes:
        The element of the returned gates_apply is tuples. There are three
        types:
        1: ("H", idx) or ("HY", idx):
            "H" or "HY": The H gate or HY gate.
            idx: The index of the qubit on which the gate will apply.
        2: ("CNOT", (ctrl_idx, apply_idx)):
            "CNOT": The CNOT gate.
            ctrl_idx: The control qubit.
            apply_idx: The active qubit.
        3: ("RZ", (idx, coeff, coeff_idx)):
            "RZ": The RZ gate.
            idx: The index of the qubit on which the gate will apply.
            coeff: The preceding factor.
            coeff_idx: The index of the external coefficient.

    Notes:
        The coeff_idx is an indicator of parameters. For example, we want to
        obtain the Trottered parametric circuit for P = c1 P1 + c2 P2 where Pi
        is a Pauli string, then we can call decompose_trottered_qubit_op(P1, idx1)
        and decompose_trottered_qubit_op(P2, idx2), assuming the indices for c1
        and c2 are idx1 and idx2 respectively in a list C. Then, the coefficient
        of the RZ gate in exp(c1*P1) is equal to the preceding factor * C[idx1],
        where C[idx1] = c1

    Examples:
        >>> from utils import decompose_trottered_qubit_op
        >>> from openfermion import QubitOperator
        >>> qubit_op = QubitOperator("X0 Y1", 2.j) + QubitOperator("Z0 X1", -1.j)
        >>> qubit_op
        2j [X0 Y1] +
        -1j [Z0 X1]
        >>> coeffs = [1.]
        >>> gates_list = decompose_trottered_qubit_op(qubit_op, 0)
        >>> gates_list
        [('H', 0), ('HY', 1), ('CNOT', (1, 0)), ('RZ', (0, (-4+0j), 0)), ('CNOT', (1, 0)), ('H', 0), ('HY', 1), ('H', 1), ('CNOT', (1, 0)), ('RZ', (0, (2+0j), 0)), ('CNOT', (1, 0)), ('H', 1)]
    """

    gates_apply_dict = {
        "X": "H",
        "Y": "HY"
    }
    gates_apply = []
    qubit_op_list = list(qubit_op)
    for i in range(len(qubit_op_list)):
        qubit_op_i = qubit_op_list[i]
        terms_i = list(qubit_op_i.terms.keys())[0]
        coeff_i = qubit_op_i.terms[terms_i]
        assert(type(coeff_i) in [complex, numpy.complex128])
        assert(numpy.isclose(coeff_i.real, 0.))
        if len(terms_i) == 0:
            """
            We ignore global phase.
            """
            continue
        parameter_i = coeff_i / 1.j
        idx_list_i = []
        qubit_gate_i = []
        for idx_apply, qubit_gate_apply in terms_i:
            idx_list_i.append(idx_apply)
            qubit_gate_i.append(qubit_gate_apply)

        for idx_op in range(len(idx_list_i)):
            if qubit_gate_i[idx_op] in gates_apply_dict:
                gates_apply.append(
                    (gates_apply_dict[qubit_gate_i[idx_op]],
                     idx_list_i[idx_op]))
        for idx_op in reversed(range(len(idx_list_i) - 1)):
            gates_apply.append(
                ("CNOT", (idx_list_i[idx_op + 1], idx_list_i[idx_op]))
            )
        gates_apply.append(
            ("RZ", (idx_list_i[0], - 2 * parameter_i, coeff_idx))
        )
        for idx_op in range(len(idx_list_i) - 1):
            gates_apply.append(
                ("CNOT", (idx_list_i[idx_op + 1], idx_list_i[idx_op]))
            )
        for idx_op in range(len(idx_list_i)):
            if qubit_gate_i[idx_op] in gates_apply_dict:
                gates_apply.append(
                    (gates_apply_dict[qubit_gate_i[idx_op]],
                     idx_list_i[idx_op]))
    return gates_apply


def decompose_trottered_qubitOp(qubitOp: openfermion.QubitOperator,
                                coeff_idx: int):
    return decompose_trottered_qubit_op(qubitOp, coeff_idx)


def parity_transformation(fermOp: openfermion.FermionOperator,
                          n_qubits: int,
                          taper_two_qubits: bool = False):
    fermOp_ = openfermion.reorder(fermOp, openfermion.up_then_down, n_qubits)
    result = openfermion.binary_code_transform(
        fermOp_, openfermion.parity_code(n_qubits))
    if (taper_two_qubits is False):
        return result
    else:
        qubitOp_remove_two = openfermion.QubitOperator()
        fermOp_remove_two = openfermion.FermionOperator()
        openfermion.symmetry_conserving_bravyi_kitaev
        remove_idx0 = 0  # n_qubits // 2 - 1
        remove_idx1 = n_qubits // 2 - 1  # n_qubits - 1
        for term, coeff in result.terms.items():
            new_term = []
            for i in term:
                # if i[0] < remove_idx0:
                #     new_term.append(i)
                # elif i[0] > remove_idx0 and i[0] < remove_idx1:
                #     new_term.append((i[0] - 1, i[1]))
                # else:
                #     pass
                if i[0] < remove_idx1:
                    new_term.append((i[0] - 1, i[1]))
                else:
                    new_term.append((i[0] - 2, i[1]))
            qubitOp_remove_two += openfermion.QubitOperator(
                tuple(new_term), coeff
            )
        return qubitOp_remove_two


def tapering_two(fermOp, n_qubits, n_electrons):
    fermOp_reorder = openfermion.reorder(
        fermOp, openfermion.up_then_down, n_qubits)
    checksum = openfermion.checksum_code(
        n_qubits // 2, (n_electrons // 2) % 2) * 2
    qubitOp = openfermion.binary_code_transform(fermOp_reorder, checksum)
    return qubitOp


def minimize(fun, x0: numpy.ndarray, args: tuple = (),
             method: str = "Adam", options: dict = {},
             jac: bool = True,
             callback=None):
    """
    A wrapper for scipy and pyTorch's optimizers.

    Args:
        fun (callable): A function which can evaluate fun(x).
        x0 (numpy.ndarray): Initial guess.
        args (tuple): Additional parameters for fun.
        method (str): Optimization methods.
        options (dict): Options for the optimizer.
        jac (bool): Whether the jacobian is calculated by fun.
        callback (callable): A call-back function. Not used
            for PyTorch's optimizers.

    Supported methods:
        "Adam",
        "Adadelta",
        "Adagrad",
        "AdamW",
        "Adamax",
        "ASGD",
        "NAdam",
        "RAdam",
        "RMSProp",
        "Rprop",
        "SGD"

    Settings in options:
        "lr" (float): default 0.1
        "momentum" (float): default 0.0, only used when method="SGD".
        "maxiter" (int): default 999.
        "ftol" (float): default 1e-8.
        "gtol" (float): default 1e-5.
        "disp" (bool): default False.
    """

    scipy_methods = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov"
    ]
    if method in scipy_methods:
        import scipy.optimize
        result_scipy = scipy.optimize.minimize(
            fun=fun,
            x0=x0,
            args=args,
            method=method,
            options=options,
            jac=jac,
            callback=callback
        )
        return result_scipy

    if jac is not True:
        raise ValueError("jac must be True.")

    import torch

    class result_class(object):
        def __init__(self, x: numpy.ndarray, fun: numpy.float64):
            self.x = x
            self.fun = fun

    x = torch.from_numpy(x0)
    x.requires_grad_(True)
    x.grad = torch.zeros(x.shape, dtype=x.dtype)

    lr = 0.1
    maxiter = 32768
    ftol = 1e-8
    gtol = 1e-5
    disp = False
    # PyTorch optimizer only
    n_params = len(x0)
    batchsize = n_params
    if "lr" in options:
        lr = options["lr"]
    if "maxiter" in options:
        maxiter = options["maxiter"]
    if "ftol" in options:
        ftol = options["ftol"]
    if "gtol" in options:
        gtol = options["gtol"]
    if "disp" in options:
        disp = options["disp"]
    if "batchsize" in options:
        batchsize = options["batchsize"]

    optimizer = None
    support_methods = [
        "Adam",
        "Adadelta",
        "Adagrad",
        "AdamW",
        "Adamax",
        "ASGD",
        "NAdam",
        "RAdam",
        "RMSProp",
        "Rprop",
        "SGD",
        "LBFGS"
    ]
    if method == "Adam":
        optimizer = torch.optim.Adam([x], lr=lr)
    elif method == "Adadelta":
        optimizer = torch.optim.Adadelta([x], lr=lr)
    elif method == "Adagrad":
        optimizer = torch.optim.Adagrad([x], lr=lr)
    elif method == "AdamW":
        optimizer = torch.optim.AdamW([x], lr=lr)
    elif method == "Adamax":
        optimizer = torch.optim.Adamax([x], lr=lr)
    elif method == "ASGD":
        optimizer = torch.optim.ASGD([x], lr=lr)
    elif method == "NAdam":
        optimizer = torch.optim.NAdam([x], lr=lr)
    elif method == "RAdam":
        optimizer = torch.optim.RAdam([x], lr=lr)
    elif method == "RMSProp":
        optimizer = torch.optim.RMSprop([x], lr=lr)
    elif method == "Rprop":
        optimizer = torch.optim.Rprop([x], lr=lr)
    elif method == "SGD":
        momentum = 0.
        if "momentum" in options:
            momentum = options["momentum"]
        optimizer = torch.optim.SGD([x], lr=lr, momentum=momentum)
    elif method == "LBFGS":
        optimizer = torch.optim.LBFGS(
            [x], lr=lr, line_search_fn="strong_wolfe")
    else:
        raise NotImplementedError("method must be one of these: {}\
".format(support_methods))

    # These methods should have batch size == n_params otherwise the
    # convergence will be bad.
    if method in ["LBFGS"]:
        if batchsize != n_params:
            print("Warning: change batchsize to %d." % (n_params))
        batchsize = n_params

    all_indices = numpy.arange(n_params)
    remain_indices = numpy.arange(n_params)

    def _get_batch(batch_size: int):
        nonlocal remain_indices
        n_remain = len(remain_indices)
        n_select = min(n_remain, batch_size)
        selected_indices = numpy.random.choice(
            n_remain, n_select, replace=False)
        remain_indices = numpy.delete(remain_indices, selected_indices)
        if len(remain_indices) == 0:
            remain_indices = numpy.arange(n_params)
        return selected_indices

    iter_count = 0
    f_diff = ftol * 9999 + 9999.
    f_last = None
    f_val_dict = {}
    g_norm_dict = {}

    def _closure():
        global f_last
        optimizer.zero_grad()
        x_func = x.detach().numpy().copy()
        f, df = fun(x_func, *args)
        x_grad = torch.from_numpy(df)
        f_val_dict.update({hash(x_func.tobytes()): f})
        g_norm_dict.update(
            {hash(x_func.tobytes()): torch.linalg.norm(x_grad).item()})
        current_batch_indices = _get_batch(batchsize)
        x_grad_clone = x_grad.clone()
        x_grad_clone[numpy.delete(all_indices, current_batch_indices)] = 0.0
        x.grad = x_grad_clone
        return f

    finish_type = 0
    while iter_count < maxiter:
        if iter_count > 0:
            if grad_last <= gtol:
                finish_type = 1
                break
            if f_diff <= ftol:
                finish_type = 2
                break
        x_func_last = x.detach().numpy().copy()
        optimizer.step(_closure)
        callback(x_func_last)
        grad_last = g_norm_dict[hash(x_func_last.tobytes())]
        f_cur = f_val_dict[hash(x_func_last.tobytes())]
        if f_last is not None:
            f_diff = abs(f_cur - f_last)
        f_last = f_cur
        if disp:
            print("Iter %5d f=%20.16f |g|=%20.16f" %
                  (iter_count, f_cur, grad_last))
        iter_count += 1

    if disp:
        finish_reason = ""
        if finish_type == 0:
            finish_reason = "maxiter"
        elif finish_type == 1:
            finish_reason = "gtol"
        elif finish_type == 2:
            finish_reason = "ftol"
        print("Finished due to %s" % (finish_reason))

    result = result_class(x_func_last, f_last)

    return result


def save_binary_qubit_op(op: openfermion.QubitOperator,
                         filename: str = "qubit_op.data"):
    """
    Convert the op into a binay file representation.
    The file structure is:
    float: 0x4026828f5c28f5c3 (11.2552), an identifier,
    int32: Number of qubits,
    double, double: Real and imaginary part of the coefficient,
    int32, int32, ...: X/Y/XI, X/Y/Z/I, ... (repeat n_qubits times).

    Args:
        op (QubitOperator): The qubit operator to be stored.
        filename (str): The name of the file.

    Returns:
        size_file (int): The size of the file in bytes.

    Notes:
        All ints are int32, double is float64.
    """
    if type(op) is not openfermion.QubitOperator:
        raise TypeError("op must be a QubitOperator but got {}.\
".format(type(op)))
    if type(filename) is not str:
        raise TypeError("filename must be a string but got {}.\
".format(type(filename)))

    n_qubits = openfermion.count_qubits(op)
    size_file = 0

    f = open(filename, "wb")
    n_qubits_array = numpy.array([n_qubits], dtype=numpy.int32)
    coeffs_array = numpy.zeros([2], dtype=numpy.float64)
    pauli_str_array = numpy.zeros([n_qubits], dtype=numpy.int32)
    get_pauli_number = {
        "I": 0,
        "X": 1,
        "Y": 2,
        "Z": 3
    }

    # f.write(bytes.fromhex("4026828f5c28f5c3"))  # 0x402682a9930be0df
    f.write(numpy.array([11.2552], dtype=numpy.float64).tobytes())
    size_file += 8

    f.write(n_qubits_array.tobytes())
    size_file += 4
    for pauli_term in op.terms:
        coeff = complex(op.terms[pauli_term])
        coeffs_array[0] = coeff.real
        coeffs_array[1] = coeff.imag
        f.write(coeffs_array.tobytes())
        size_file += 16
        coeffs_array.fill(0)

        for pos, pauli_symbol in pauli_term:
            pauli_str_array[pos] = get_pauli_number[pauli_symbol]
        f.write(pauli_str_array.tobytes())
        size_file += n_qubits * 4
        pauli_str_array.fill(0)

    f.close()

    return size_file


def read_binary_qubit_op(filename: str = "qubit_op.data"):
    """
    """
    f = open(filename, "rb")
    identifier = f.read(8)
    # if identifier != bytes.fromhex("4026828f5c28f5c3"):  # 0x402682a9930be0df
    if numpy.frombuffer(identifier, dtype=numpy.float64) != 11.2552:
        raise ValueError("The file is not saved by QCQC.")

    n_qubits = numpy.frombuffer(f.read(4), dtype=numpy.int32)
    n_qubits = int(n_qubits)

    get_pauli_symbol = {
        0: "I",
        1: "X",
        2: "Y",
        3: "Z"
    }

    qubit_op = openfermion.QubitOperator()

    pauli_str_array = numpy.zeros([n_qubits], dtype=numpy.int32)
    chunk_size = n_qubits * 4
    coeff_bin = f.read(16)
    pauli_str_bin = f.read(chunk_size)
    while len(coeff_bin) != 0 and len(pauli_str_bin) != 0:
        assert(len(pauli_str_bin) == chunk_size)
        coeffs_array = numpy.frombuffer(coeff_bin, dtype=numpy.float64)
        pauli_str_array = numpy.frombuffer(pauli_str_bin, dtype=numpy.int32)
        coeff = coeffs_array[0] + 1.j * coeffs_array[1]
        term_list = []
        for pos, pauli_number in enumerate(pauli_str_array):
            pauli_symbol = get_pauli_symbol[pauli_number]
            if pauli_symbol != "I":
                term_list.append((pos, pauli_symbol))
        op_cur = openfermion.QubitOperator(tuple(term_list), coeff)
        qubit_op += op_cur

        coeff_bin = f.read(16)
        pauli_str_bin = f.read(chunk_size)

    return qubit_op


def particle_number_operator(n_qubits: int):
    """
    Generate the particle number operator of Fermionic wave functions.

    Args:
        n_qubits (int): Number of qubits.

    Return:
        particle_num_op (openfermion.FermionOperator): Particle number operator.
    """
    particle_num_op = openfermion.FermionOperator()
    for i in range(n_qubits):
        particle_num_op += openfermion.FermionOperator(
            ((i, 1), (i, 0)), 1.0)
    return particle_num_op


def spin_z_operator(n_qubits: int):
    """
    Generate the spin-Z operator.

    Args:
        n_qubits (int): Number of qubits.

    Return:
        spin_z_op (openfermion.FermionOperator): Particle number operator.
    """
    spin_z_op = openfermion.FermionOperator()
    for a in range(n_qubits):
        for b in range(n_qubits):
            if b != a:
                continue
            sz_a = 0.5 if a % 2 == 0 else -0.5
            fermion_op = openfermion.FermionOperator(
                ((a, 1), (b, 0)), 1.) * sz_a
            spin_z_op += fermion_op
    return spin_z_op


def _spin2_matrix_elements(sz):
    """
    Copied from PennyLane:
    https://pennylane.readthedocs.io/en/stable/code/api/pennylane_qchem.qchem.obs._spin2_matrix_elements.html#pennylane_qchem.qchem.obs._spin2_matrix_elements
    """
    r"""
    Builds the table of matrix elements
    :math:`\langle \bm{\alpha}, \bm{\beta} \vert \hat{s}_1 \cdot \hat{s}_2 \vert
    \bm{\gamma}, \bm{\delta} \rangle` of the two-particle spin operator
    :math:`\hat{s}_1 \cdot \hat{s}_2`.

    The matrix elements are evaluated using the expression

    .. math::

        \langle ~ (\alpha, s_{z_\alpha});~ (\beta, s_{z_\beta}) ~ \vert \hat{s}_1 &&
        \cdot \hat{s}_2 \vert ~ (\gamma, s_{z_\gamma}); ~ (\delta, s_{z_\gamma}) ~ \rangle =
        \delta_{\alpha,\delta} \delta_{\beta,\gamma} \\
        && \times \left( \frac{1}{2} \delta_{s_{z_\alpha}, s_{z_\delta}+1}
        \delta_{s_{z_\beta}, s_{z_\gamma}-1} + \frac{1}{2} \delta_{s_{z_\alpha}, s_{z_\delta}-1}
        \delta_{s_{z_\beta}, s_{z_\gamma}+1} + s_{z_\alpha} s_{z_\beta}
        \delta_{s_{z_\alpha}, s_{z_\delta}} \delta_{s_{z_\beta}, s_{z_\gamma}} \right),

    where :math:`\alpha` and :math:`s_{z_\alpha}` refer to the quantum numbers of the spatial
    function and the spin projection, respectively, of the single-particle state
    :math:`\vert \bm{\alpha} \rangle \equiv \vert \alpha, s_{z_\alpha} \rangle`.

    Args:
        sz (array[float]): spin-projection of the single-particle states

    Returns:
        array: NumPy array with the table of matrix elements. The first four columns
        contain the indices :math:`\bm{\alpha}`, :math:`\bm{\beta}`, :math:`\bm{\gamma}`,
        :math:`\bm{\delta}` and the fifth column stores the computed matrix element.

    **Example**

    >>> sz = np.array([0.5, -0.5])
    >>> print(_spin2_matrix_elements(sz))
    [[ 0.    0.    0.    0.    0.25]
     [ 0.    1.    1.    0.   -0.25]
     [ 1.    0.    0.    1.   -0.25]
     [ 1.    1.    1.    1.    0.25]
     [ 0.    1.    0.    1.    0.5 ]
     [ 1.    0.    1.    0.    0.5 ]]
    """

    n = numpy.arange(sz.size)

    alpha = n.reshape(-1, 1, 1, 1)
    beta = n.reshape(1, -1, 1, 1)
    gamma = n.reshape(1, 1, -1, 1)
    delta = n.reshape(1, 1, 1, -1)

    # we only care about indices satisfying the following boolean mask
    mask = numpy.logical_and(alpha // 2 == delta // 2, beta // 2 == gamma // 2)

    # diagonal elements
    diag_mask = numpy.logical_and(
        sz[alpha] == sz[delta], sz[beta] == sz[gamma])
    diag_indices = numpy.argwhere(numpy.logical_and(mask, diag_mask))
    diag_values = (sz[alpha] * sz[beta]).flatten()

    diag = numpy.vstack([diag_indices.T, diag_values]).T

    # off-diagonal elements
    m1 = numpy.logical_and(sz[alpha] == sz[delta] +
                           1, sz[beta] == sz[gamma] - 1)
    m2 = numpy.logical_and(sz[alpha] == sz[delta] -
                           1, sz[beta] == sz[gamma] + 1)

    off_diag_mask = numpy.logical_and(mask, numpy.logical_or(m1, m2))
    off_diag_indices = numpy.argwhere(off_diag_mask)
    off_diag_values = numpy.full([len(off_diag_indices)], 0.5)

    off_diag = numpy.vstack([off_diag_indices.T, off_diag_values]).T

    # combine the off diagonal and diagonal tables into a single table
    return numpy.vstack([diag, off_diag])


def total_spin_operator(n_qubits: int, n_electrons: int):
    """
    Generator the total spin operator S2.

    Args:
        n_qubits (int): Number of qubits.
        n_electrons (int): Number of electrons.

    Return:
        total_spin_op (openfermion.FermionOperator): Total spin operator.
    """
    sz = numpy.where(numpy.arange(n_qubits) % 2 == 0, 0.5, -0.5)
    table = _spin2_matrix_elements(sz)

    # create the list of ``FermionOperator`` objects
    s2_op = openfermion.ops.FermionOperator("") * 3 / 4 * n_electrons
    for i in table:
        s2_op += openfermion.ops.FermionOperator(
            ((int(i[0]), 1), (int(i[1]), 1), (int(i[2]), 0), (int(i[3]), 0)),
            i[4]
        )

    return s2_op


def _jordan_wigner_mp_worker(args):
    n_workers = args[0]
    worker_idx = args[1]
    global global_ferm_op_for_jordan_wigner_mp
    operator = global_ferm_op_for_jordan_wigner_mp
    term_idx = 0

    transformed_operator = openfermion.QubitOperator()
    for term in operator.terms:
        if term_idx % n_workers != worker_idx:
            term_idx += 1
            continue
        term_idx += 1
        # Initialize identity matrix.
        transformed_term = openfermion.QubitOperator((), operator.terms[term])
        # Loop through operators, transform and multiply.
        for ladder_operator in term:
            z_factors = tuple(
                (index, 'Z') for index in range(ladder_operator[0]))
            pauli_x_component = openfermion.QubitOperator(
                z_factors + ((ladder_operator[0], 'X'),), 0.5)
            if ladder_operator[1]:
                pauli_y_component = openfermion.QubitOperator(
                    z_factors + ((ladder_operator[0], 'Y'),), -0.5j)
            else:
                pauli_y_component = openfermion.QubitOperator(
                    z_factors + ((ladder_operator[0], 'Y'),), 0.5j)
            transformed_term *= pauli_x_component + pauli_y_component
        transformed_operator += transformed_term
    return transformed_operator


def jordan_wigner_mp(ferm_op: openfermion.FermionOperator,
                     n_procs: int = 2):
    """
    Perform jordan-wigner transformation in parallel.

    Args:
        ferm_op (FermionOperator): Fermion operator.
        n_procs (int): Number of processes.

    Return:
        qubit_op (QubitOperator): Qubit operator under jordan-wigner
            transformation.
    """
    if type(ferm_op) is not openfermion.FermionOperator:
        raise TypeError("ferm_op should be a FermionOperator but got {}.\
".format(type(ferm_op)))
    if type(n_procs) is not int:
        raise TypeError("n_procs should be an integer but got {}.\
".format(type(n_procs)))

    n_terms = len(ferm_op.terms)
    n_workers = min(n_terms, n_procs)
    if (n_workers != n_procs):
        print("Warning: change n_procs to %d" % (n_workers))

    chunk_size = n_terms // n_workers
    chunk_list = [chunk_size for i in range(n_workers)]
    for i in range(n_terms - chunk_size * n_workers):
        chunk_list[i] += 1

    global global_ferm_op_for_jordan_wigner_mp
    global_ferm_op_for_jordan_wigner_mp = ferm_op

    import multiprocessing

    args_worker = []
    for i in range(n_workers):
        args_worker.append((n_workers, i))

    pool = multiprocessing.Pool(n_workers)

    map_result = pool.map(_jordan_wigner_mp_worker, args_worker)
    pool.close()
    pool.join()

    qubit_op = openfermion.QubitOperator()
    for i in range(n_workers):
        qubit_op += map_result[i]

    return qubit_op


def jordan_wigner_mp_memory_opt(
        ferm_op: openfermion.FermionOperator,
        n_procs: int = 2,
        n_terms_limit: int = 2**10):
    """
    Perform jordan-wigner transformation in parallel.

    Args:
        ferm_op (FermionOperator): Fermion operator.
        n_procs (int): Number of processes.
        n_terms_limit (int): Maximum number of terms to transform
            at the same time.

    Return:
        qubit_op (QubitOperator): Qubit operator under jordan-wigner
            transformation.
    """
    if type(ferm_op) is not openfermion.FermionOperator:
        raise TypeError("ferm_op should be a FermionOperator but got {}.\
".format(type(ferm_op)))
    if type(n_procs) is not int:
        raise TypeError("n_procs should be an integer but got {}.\
".format(type(n_procs)))

    n_terms = len(ferm_op.terms)
    n_loops = 0
    chunk_size_per_loop = []
    n_workers_per_loop = []
    for i in range(0, n_terms, n_terms_limit):
        n_loops += 1
        chunk_size_i = min(n_terms_limit, n_terms - i)
        chunk_size_per_loop.append(chunk_size_i)
        n_workers_per_loop.append(min(n_procs, chunk_size_i))
    n_workers_tot = sum(n_workers_per_loop)
    print("Divided into %d chunks, number of processes each chunk: \n"
          % (n_loops), n_workers_per_loop)

    global global_ferm_op_for_jordan_wigner_mp
    global_ferm_op_for_jordan_wigner_mp = ferm_op
    import multiprocessing

    qubit_op = openfermion.QubitOperator()

    for i in range(n_loops):
        args_worker_i = []
        n_workers_i = n_workers_per_loop[i]
        for j in range(n_workers_i):
            args_worker_i.append(
                (n_workers_tot, j + sum(n_workers_per_loop[:i])))
        pool_i = multiprocessing.Pool(n_workers_i)
        map_result_i = pool_i.map(_jordan_wigner_mp_worker, args_worker_i)
        pool_i.close()
        pool_i.join()
        qubit_op_i = openfermion.QubitOperator()
        for j in range(n_workers_i):
            qubit_op_i += map_result_i[j]
        openfermion.save_operator(
            qubit_op_i,
            file_name="tmp_jw_mp_qubit_op_%d" % (i),
            data_directory="./",
            allow_overwrite=True)
        qubit_op += qubit_op_i

    return qubit_op


def generate_n_particle_rdm_ferm_op(
        n_orb: int, n_particle: int = 1) -> numpy.array:
    """
    Generate Fermion operators for constructing N-particle RDM using
    spin-orbitals.

    Args:
        n_orb (int): Number of spatial orbitals.

    Return:
        rdm_ferm_op_list (numpy.array): A two-dimension array containing
            FermionOperator for RDM elements.
    """
    rdm_ferm_op_list = numpy.array(
        [None] * (n_orb * 2)**(n_particle * 2)).reshape(
            [(n_orb * 2)] * (n_particle * 2))
    for indices in itertools.product(
            *[[i for i in range(n_orb * 2)] for j in range(n_particle * 2)]):
        rdm_ferm_op_list[indices] = openfermion.FermionOperator(
            [(i, 1) for i in indices[:n_particle]] +
            [(j, 0) for j in indices[n_particle:]],
            1.0)
    return rdm_ferm_op_list


def get_single_orbital_operators(orb_idx: int):
    """
    Single-orbital operators in the format of FermionOperator.

    Reference:
    [1]. J. Chem. Theory Comput. 2013, 9, 2959-2973
        https://dx.doi.org/10.1021/ct400247p
        Table 4.
    """

    def _cc_down(orb_idx: int):
        op = openfermion.FermionOperator(
            ((orb_idx * 2, 1)),
            1.0
        )
        return op

    def _cc_up(orb_idx: int):
        op = openfermion.FermionOperator(
            ((orb_idx * 2 + 1, 1)),
            1.0
        )
        return op

    def _ca_down(orb_idx: int):
        op = openfermion.FermionOperator(
            ((orb_idx * 2, 0)),
            1.0
        )
        return op

    def _ca_up(orb_idx: int):
        op = openfermion.FermionOperator(
            ((orb_idx * 2 + 1, 0)),
            1.0
        )
        return op

    def _n_down(orb_idx: int):
        op = _cc_down(orb_idx) * _ca_down(orb_idx)
        return op

    def _n_up(orb_idx: int):
        op = _cc_up(orb_idx) * _ca_up(orb_idx)
        return op

    o1 = 1 - _n_up(orb_idx) - _n_down(orb_idx) + \
        _n_up(orb_idx) * _n_down(orb_idx)
    o2 = _ca_down(orb_idx) - _n_up(orb_idx) * _ca_down(orb_idx)
    o3 = _ca_up(orb_idx) - _n_down(orb_idx) * _ca_up(orb_idx)
    o4 = _ca_down(orb_idx) * _ca_up(orb_idx)
    o5 = _cc_down(orb_idx) - _n_up(orb_idx) * _cc_down(orb_idx)
    o6 = _n_down(orb_idx) - _n_up(orb_idx) * _n_down(orb_idx)
    o7 = _cc_down(orb_idx) * _ca_up(orb_idx)
    o8 = -_n_down(orb_idx) * _ca_up(orb_idx)
    o9 = _cc_up(orb_idx) - _n_down(orb_idx) * _cc_up(orb_idx)
    o10 = _ca_down(orb_idx) * _cc_up(orb_idx)
    o11 = _n_up(orb_idx) - _n_up(orb_idx) * _n_down(orb_idx)
    o12 = _n_up(orb_idx) * _ca_down(orb_idx)
    o13 = _cc_down(orb_idx) * _cc_up(orb_idx)
    o14 = -_n_down(orb_idx) * _cc_up(orb_idx)
    o15 = _n_up(orb_idx) * _cc_down(orb_idx)
    o16 = _n_up(orb_idx) * _n_down(orb_idx)

    op = [o1, o2, o3, o4,
          o5, o6, o7, o8,
          o9, o10, o11, o12,
          o13, o14, o15, o16]
    return op


def get_single_orbital_operator(orb_idx: int, op_idx: int):

    op = get_single_orbital_operators(orb_idx)

    return op[op_idx - 1]


def get_one_orbital_rdms(n_orb: int) -> numpy.array:
    """
    One-orbital RDM in the format of FermionOperator.
    """
    one_orb_rdm_list = numpy.array(
        [openfermion.FermionOperator()] * n_orb * 16).reshape([n_orb, 4, 4])
    for i in range(n_orb):
        one_orb_rdm_i = one_orb_rdm_list[i, :, :]
        orbital_operators_i = [None] + get_single_orbital_operators(orb_idx=i)
        one_orb_rdm_i[0, 0] = orbital_operators_i[1]
        one_orb_rdm_i[1, 1] = orbital_operators_i[6]
        one_orb_rdm_i[2, 2] = orbital_operators_i[11]
        one_orb_rdm_i[3, 3] = orbital_operators_i[16]
        one_orb_rdm_list[i] = one_orb_rdm_i
    return one_orb_rdm_list


def get_two_orbital_rdms(n_orb: int) -> numpy.array:
    """
    Two-orbital RDM in the format of FermionOperator.

    Warnings:
        This function seems to be incorrect at present.
    """
    two_orb_rdm_list = numpy.array(
        [openfermion.FermionOperator()] * n_orb**2 * 256).reshape(
            [n_orb, n_orb, 16, 16])
    for i in range(n_orb):
        for j in range(n_orb):
            two_orb_rdm_ij = two_orb_rdm_list[i, j, :, :]
            orbital_operators_i = [None] + \
                get_single_orbital_operators(orb_idx=i)
            orbital_operators_j = [None] + \
                get_single_orbital_operators(orb_idx=j)

            # 1/1
            two_orb_rdm_ij[0, 0] = \
                orbital_operators_i[1] * \
                orbital_operators_j[1]

            # 1/6 2/5
            # 5/2 6/1
            two_orb_rdm_ij[1, 1] = \
                orbital_operators_i[1] * \
                orbital_operators_j[6]
            two_orb_rdm_ij[1, 2] = \
                orbital_operators_i[2] * \
                orbital_operators_j[5]
            two_orb_rdm_ij[2, 1] = \
                orbital_operators_i[5] * \
                orbital_operators_j[2]
            two_orb_rdm_ij[2, 2] = \
                orbital_operators_i[6] * \
                orbital_operators_j[1]

            # 1/11 3/9
            # 9/3  11/1
            two_orb_rdm_ij[3, 3] = \
                orbital_operators_i[1] * \
                orbital_operators_j[11]
            two_orb_rdm_ij[3, 4] = \
                orbital_operators_i[3] * \
                orbital_operators_j[9]
            two_orb_rdm_ij[4, 3] = \
                orbital_operators_i[9] * \
                orbital_operators_j[3]
            two_orb_rdm_ij[4, 4] = \
                orbital_operators_i[11] * \
                orbital_operators_j[1]

            # 6/6
            two_orb_rdm_ij[5, 5] = \
                orbital_operators_i[6] * \
                orbital_operators_j[6]

            # 1/16 2/15 3/14 4/13
            # 5/12 6/11 7/10 8/9
            # 9/8  10/7 11/6 12/5
            # 13/4 14/3 15/2 16/1
            two_orb_rdm_ij[6, 6] = \
                orbital_operators_i[1] * \
                orbital_operators_j[16]
            two_orb_rdm_ij[6, 7] = \
                orbital_operators_i[2] * \
                orbital_operators_j[15]
            two_orb_rdm_ij[6, 8] = \
                orbital_operators_i[3] * \
                orbital_operators_j[14]
            two_orb_rdm_ij[6, 9] = \
                orbital_operators_i[4] * \
                orbital_operators_j[13]
            two_orb_rdm_ij[7, 6] = \
                orbital_operators_i[5] * \
                orbital_operators_j[12]
            two_orb_rdm_ij[7, 7] = \
                orbital_operators_i[6] * \
                orbital_operators_j[11]
            two_orb_rdm_ij[7, 8] = \
                orbital_operators_i[7] * \
                orbital_operators_j[10]
            two_orb_rdm_ij[7, 9] = \
                orbital_operators_i[8] * \
                orbital_operators_j[9]
            two_orb_rdm_ij[8, 6] = \
                orbital_operators_i[9] * \
                orbital_operators_j[8]
            two_orb_rdm_ij[8, 7] = \
                orbital_operators_i[10] * \
                orbital_operators_j[7]
            two_orb_rdm_ij[8, 8] = \
                orbital_operators_i[11] * \
                orbital_operators_j[6]
            two_orb_rdm_ij[8, 9] = \
                orbital_operators_i[12] * \
                orbital_operators_j[5]
            two_orb_rdm_ij[9, 6] = \
                orbital_operators_i[13] * \
                orbital_operators_j[4]
            two_orb_rdm_ij[9, 7] = \
                orbital_operators_i[14] * \
                orbital_operators_j[3]
            two_orb_rdm_ij[9, 8] = \
                orbital_operators_i[15] * \
                orbital_operators_j[2]
            two_orb_rdm_ij[9, 9] = \
                orbital_operators_i[16] * \
                orbital_operators_j[1]

            # 11/11
            two_orb_rdm_ij[10, 10] = \
                orbital_operators_i[11] * \
                orbital_operators_j[11]

            # 6/16 8/14
            # 14/8 16/6
            two_orb_rdm_ij[11, 11] = \
                orbital_operators_i[6] * \
                orbital_operators_j[16]
            two_orb_rdm_ij[11, 12] = \
                orbital_operators_i[8] * \
                orbital_operators_j[14]
            two_orb_rdm_ij[12, 11] = \
                orbital_operators_i[14] * \
                orbital_operators_j[8]
            two_orb_rdm_ij[12, 12] = \
                orbital_operators_i[16] * \
                orbital_operators_j[6]

            # 11/16 12/15
            # 15/12 16/11
            two_orb_rdm_ij[13, 13] = \
                orbital_operators_i[11] * \
                orbital_operators_j[16]
            two_orb_rdm_ij[13, 14] = \
                orbital_operators_i[12] * \
                orbital_operators_j[15]
            two_orb_rdm_ij[14, 13] = \
                orbital_operators_i[15] * \
                orbital_operators_j[12]
            two_orb_rdm_ij[14, 14] = \
                orbital_operators_i[16] * \
                orbital_operators_j[11]

            # 16/16
            two_orb_rdm_ij[15, 15] = \
                orbital_operators_i[16] * \
                orbital_operators_j[16]

    return two_orb_rdm_list


def generate_orbital_rotation_ferm_op_from_u(u: numpy.ndarray):
    r"""
    U = expm(-\kappa)
    \hat{\kappa} = \sum_{ij}{\kappa_{ij} a^{\dagger}_{i} a_{j}}
    \hat{R} = exp(-\hat{\kappa})

    Notes:
        The returned term is \hat{\kappa} NOT \hat{R}!.
    """
    import scipy.linalg

    n_orb = u.shape[0]
    assert(n_orb == u.shape[1])
    assert(len(u.shape) == 2)
    error_unitary = numpy.linalg.norm(u.T.conj().dot(u) - numpy.eye(n_orb))
    assert(numpy.isclose(error_unitary, 0.0))
    kappa = scipy.linalg.logm(u) * -1
    u_check = scipy.linalg.expm(-kappa)
    error_check = numpy.linalg.norm(u_check - u)
    assert(numpy.isclose(error_check, 0.0))
    kappa_op = openfermion.FermionOperator()
    for i in range(n_orb * 2):
        for j in range(n_orb * 2):
            if i % 2 + j % 2 == 1:
                continue
            kappa_op += openfermion.FermionOperator(
                ((i, 1), (j, 0)), kappa[i // 2][j // 2])
    return kappa_op


def save_binary_qubit_op_list(
        op_list: list, filename: str = "qubit_ops.vdata"):
    """
    Convert the op into a binay file representation.
    The file structure is:
    float: 0x4026828f5c28f5c3 (11.2552), an identifier,
    uint64: Number of qubit operators.
    { (Repeat over all qubit operators)
        uint8: 4 (Indicates the start of a qubit operator)
        uint64: Number of terms in this qubit operator
        float64, float64: Real and imaginary part of the coefficient
        { (Repeat over all Pauli strings in this qubit operator)
            uint8, uint16, uint8, uint16, ... uint8:
                Pauli, Qubit index, Pauli, Qubit index, ..., 5
                Pauli: (0->I, 1->X, 2->Y, 3->Z)
                Qubit index: starts from 0.
                5: Indicates the end of current Pauli string
        }
    }
    uint8: 5, Indicates end of the dataset.

    Args:
        op (QubitOperator): The qubit operator to be stored.
        filename (str): The name of the file.

    Returns:
        size_file (int): The size of the file in bytes.
    """
    if type(op_list) is not list:
        raise TypeError("op_list must be a list but got{}.\
".format(type(op_list)))
    for op in op_list:
        if type(op) is not openfermion.QubitOperator:
            raise TypeError("op must be a QubitOperator but got {}.\
    ".format(type(op)))
    if type(filename) is not str:
        raise TypeError("filename must be a string but got {}.\
".format(type(filename)))

    size_file = 0
    get_pauli_number = {
        "I": numpy.array([0], dtype=numpy.uint8),
        "X": numpy.array([1], dtype=numpy.uint8),
        "Y": numpy.array([2], dtype=numpy.uint8),
        "Z": numpy.array([3], dtype=numpy.uint8)
    }

    f = open(filename, "wb")
    f.write(numpy.array([11.2552], dtype=numpy.float64).tobytes())
    size_file += 8

    n_qubit_ops = len(op_list)
    f.write(numpy.array([n_qubit_ops], dtype=numpy.uint64).tobytes())
    size_file += 8

    for op in op_list:
        f.write(numpy.array([4], dtype=numpy.uint8).tobytes())
        size_file += 1
        n_terms = len(op.terms)
        f.write(numpy.array([n_terms], dtype=numpy.uint64).tobytes())
        size_file += 8
        for pauli_str in op.terms:
            coeff = op.terms[pauli_str]
            f.write(numpy.array([coeff.real, coeff.imag],
                                dtype=numpy.float64).tobytes())
            size_file += 16
            for pauli_term in pauli_str:
                pauli_number = get_pauli_number[pauli_term[1]]
                f.write(pauli_number.tobytes())
                size_file += 1
                qubit_idx = pauli_term[0]
                f.write(numpy.array([qubit_idx], dtype=numpy.uint16).tobytes())
                size_file += 2
            f.write(numpy.array([5], dtype=numpy.uint8).tobytes())
            size_file += 1

    f.write(numpy.array([5], dtype=numpy.uint8).tobytes())
    size_file += 1

    f.close()
    return size_file


def read_binary_qubit_op_list(filename: str = "qubit_ops.vdata"):
    """
    """
    f = open(filename, "rb")
    identifier = f.read(8)
    if numpy.frombuffer(identifier, dtype=numpy.float64, count=1) != 11.2552:
        raise ValueError("The file is not saved by QCQC.")
    get_pauli_symbol = {
        0: "I",
        1: "X",
        2: "Y",
        3: "Z"
    }
    qubit_op_list = []
    n_qubit_ops_buffer = f.read(8)
    n_qubits_ops = numpy.frombuffer(
        n_qubit_ops_buffer, dtype=numpy.uint64,
        count=1)[0]

    for i in range(n_qubits_ops):
        tmp_buffer = f.read(1)
        assert(numpy.frombuffer(tmp_buffer, dtype=numpy.uint8, count=1)[0]
               == 4)
        qubit_op_i = openfermion.QubitOperator()
        n_terms_buffer = f.read(8)
        n_terms = numpy.frombuffer(
            n_terms_buffer, dtype=numpy.uint64,
            count=1)[0]
        term_idx = 0
        for j in range(n_terms):
            coeff_ij_buffer = f.read(16)
            coeff_ij = numpy.frombuffer(
                coeff_ij_buffer, dtype=numpy.complex128,
                count=1)[0]
            pauli_str_ij = []
            while (True):
                pauli_number_buffer = f.read(1)
                pauli_number = numpy.frombuffer(
                    pauli_number_buffer,
                    dtype=numpy.uint8,
                    count=1)[0]
                if pauli_number == 5:
                    break
                else:
                    qubit_idx_buffer = f.read(2)
                    qubit_idx = numpy.frombuffer(
                        qubit_idx_buffer,
                        dtype=numpy.uint16,
                        count=1)[0]
                    pauli_str_ij.append(
                        (int(qubit_idx), get_pauli_symbol[pauli_number]))
            qubit_op_ij = openfermion.QubitOperator(pauli_str_ij, coeff_ij)
            qubit_op_i += qubit_op_ij
            term_idx += 1
        qubit_op_list.append(qubit_op_i)

    # End
    tmp_buffer = f.read(1)
    assert(numpy.frombuffer(tmp_buffer, dtype=numpy.uint8, count=1)[0] == 5)

    f.close()
    return qubit_op_list
