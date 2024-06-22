import functools

import numpy
import scipy.stats
import scipy.linalg
import pyscf
import pyscf.lo
import pyscf.symm
import pyscf.cc
import pyscf.fci
import openfermion


def visualize_orbitals(
        molden_filename: str,
        mol: pyscf.gto.M,
        mo_coeff: numpy.ndarray):
    assert(molden_filename[-7:] == ".molden")
    pyscf.tools.molden.from_mo(
        mol=mol, filename=molden_filename,
        mo_coeff=mo_coeff)
    n_orb_draw = mo_coeff.shape[1]
    for orb_idx in range(n_orb_draw):
        spt_filename = "{}_{}.spt".format(
            molden_filename[:-7], orb_idx + 1)
        f = open(spt_filename, "w")
        f.write("load {}\n".format(molden_filename))
        f.write("isoSurface MO %03d" % (orb_idx + 1))
        f.close()
    return


def get_orbsym_order(mf: pyscf.scf.rhf.RHF):
    orbsym = mf.orbsym if hasattr(mf, "orbsym") else \
        numpy.zeros(mf.mo_coeff.shape[1], numpy.int32)
    sorted_args = list(numpy.argsort(orbsym))
    return sorted_args


def get_fiedler_order(
        mo_coeff: numpy.ndarray, molecule: pyscf.gto.M, mf: pyscf.scf.rhf.RHF):
    """
    References:
        [1]. Olivares-Amaya et al. J. Chem. Phys. 142, 034102 (2015).
        [2]. G. BARCZA et al. PHYSICAL REVIEW A 83, 012508 (2011)
    """
    dm = mf.make_rdm1(mo_coeff=mo_coeff)
    k = mf.get_k(mol=molecule, dm=dm)
    d = numpy.diagflat(numpy.sum(numpy.abs(k), axis=1))
    l = d - numpy.abs(k)
    assert(numpy.isclose(numpy.linalg.norm(l - l.T.conj()), 0.0))
    eigs, eigvs = numpy.linalg.eigh(l)
    eigv_fiedler = eigvs[:, 1]
    eigv_fiedler = eigv_fiedler - numpy.mean(eigv_fiedler)
    eigv_fiedler = eigv_fiedler / numpy.linalg.norm(eigv_fiedler)
    eigv_fiedler_1 = eigv_fiedler.copy()

    n_orb = mo_coeff.shape[1]

    # def _get_fiedler_distance(x: numpy.ndarray):
    #     x = x - numpy.mean(x)
    #     x = x / numpy.linalg.norm(x)
    #     s = 0.0
    #     for i in range(n_orb):
    #         for j in range(n_orb):
    #             s += (x[i] - x[j])**2 * abs(k[i][j])
    #     return s

    # x0 = numpy.random.rand(n_orb)
    # res = scipy.optimize.minimize(
    #     fun=_get_fiedler_distance,
    #     x0=x0)
    # x_opt = res.x
    # x_opt = x_opt - numpy.mean(x_opt)
    # x_opt = x_opt / numpy.linalg.norm(x_opt)
    # eigv_fiedler = x_opt

    order_fiedler = numpy.argsort(eigv_fiedler)
    return eigv_fiedler, order_fiedler


def get_hamiltonian_ferm_op_from_ints(one_body_int: numpy.ndarray,
                                      two_body_int: numpy.ndarray,
                                      eps: float = 0.0):
    """
    Construct the one- and two-body terms of the Hamiltonian for a given
    one-electron integral in (p+, q) order and a give two-electron integral
    in (p+, q+, r, s) order.

    Args:
        one_body_int (numpy.ndarray): one-electron integral in (p+, q) order.
        two_body_int (numpy.ndarray): two-electron integral in
            (p+, q+, r, s) order.
        eps (float): cut-off threshold.

    Notes:
        The integrals are for spin-orbitals.
    """
    hamiltonian_fermOp_1 = openfermion.FermionOperator()
    hamiltonian_fermOp_2 = openfermion.FermionOperator()

    for (p, q) in zip(*((abs(one_body_int) > eps).nonzero())):
        p = int(p)
        q = int(q)
        hamiltonian_fermOp_1 += openfermion.FermionOperator(
            ((p, 1), (q, 0)),
            one_body_int[p][q]
        )
    for (p, q, r, s) in zip(*((abs(two_body_int) > eps).nonzero())):
        p = int(p)
        q = int(q)
        r = int(r)
        s = int(s)
        hamiltonian_fermOp_2 += openfermion.FermionOperator(
            ((p, 1), (q, 1), (r, 0), (s, 0)),
            two_body_int[p][q][r][s] * 0.5
        )

    hamiltonian_fermOp_1 = openfermion.normal_ordered(hamiltonian_fermOp_1)
    hamiltonian_fermOp_2 = openfermion.normal_ordered(hamiltonian_fermOp_2)
    return hamiltonian_fermOp_1, hamiltonian_fermOp_2


def get_hamiltonian_ferm_op_from_mo_ints(
        one_body_mo: numpy.ndarray,
        two_body_mo: numpy.ndarray,
        eps: float = 0.0):
    """
    Construct the one- and two-body terms of the Hamiltonian for a given
    one-electron MO integral in (p+, q) order and a give two-electron MO
    integral in (p+, s, q+, r) order (PySCF's ordering).

    Args:
        one_body_mo (numpy.ndarray): one-electron integral in (p+, q) order.
        two_body_mo (numpy.ndarray): two-electron integral in
            (p+, s, q+, r) order.
        eps (float): cut-off threshold.

    Notes:
        The integrals are for spatial-orbitals.
    """
    global two_body_mo_pqrs_global
    two_body_mo_pqrs = numpy.moveaxis(
        two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
    hamiltonian_ferm_op_1 = openfermion.FermionOperator()
    hamiltonian_ferm_op_2 = openfermion.FermionOperator()
    for (p, q) in zip(*((abs(one_body_mo) > eps).nonzero())):
        p = int(p)
        q = int(q)
        pa = p * 2
        pb = p * 2 + 1
        qa = q * 2
        qb = q * 2 + 1
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pa, 1), (qa, 0)),
            one_body_mo[p][q]
        )
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pb, 1), (qb, 0)),
            one_body_mo[p][q]
        )
    for (p, q, r, s) in zip(*((abs(two_body_mo_pqrs) > eps).nonzero())):
        p = int(p)
        q = int(q)
        r = int(r)
        s = int(s)
        pa = p * 2
        pb = p * 2 + 1
        qa = q * 2
        qb = q * 2 + 1
        ra = r * 2
        rb = r * 2 + 1
        sa = s * 2
        sb = s * 2 + 1
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pa, 1), (qa, 1), (ra, 0), (sa, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pb, 1), (qb, 1), (rb, 0), (sb, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pa, 1), (qb, 1), (rb, 0), (sa, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
        hamiltonian_ferm_op_2 += openfermion.FermionOperator(
            ((pb, 1), (qa, 1), (ra, 0), (sb, 0)),
            two_body_mo_pqrs[p][q][r][s] * 0.5
        )
    hamiltonian_ferm_op_1 = openfermion.normal_ordered(hamiltonian_ferm_op_1)
    hamiltonian_ferm_op_2 = openfermion.normal_ordered(hamiltonian_ferm_op_2)
    return hamiltonian_ferm_op_1, hamiltonian_ferm_op_2


def _get_hamiltonian_ferm_op_from_mo_ints_mp_worker(args_worker):
    global two_body_mo_pqrs_global
    start_idx = args_worker[0]
    end_idx = args_worker[1]
    eps = args_worker[2]
    n_orb = two_body_mo_pqrs_global.shape[0]
    hamiltonian_ferm_op_2_worker = openfermion.FermionOperator()
    for p in range(start_idx, end_idx):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    if abs(two_body_mo_pqrs_global[p, q, r, s]) < eps:
                        continue
                    pa = p * 2
                    pb = p * 2 + 1
                    qa = q * 2
                    qb = q * 2 + 1
                    ra = r * 2
                    rb = r * 2 + 1
                    sa = s * 2
                    sb = s * 2 + 1
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pa, 1), (qa, 1), (ra, 0), (sa, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pb, 1), (qb, 1), (rb, 0), (sb, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pa, 1), (qb, 1), (rb, 0), (sa, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
                    hamiltonian_ferm_op_2_worker += openfermion.FermionOperator(
                        ((pb, 1), (qa, 1), (ra, 0), (sb, 0)),
                        two_body_mo_pqrs_global[p][q][r][s] * 0.5
                    )
    return hamiltonian_ferm_op_2_worker


def get_hamiltonian_ferm_op_from_mo_ints_mp(
        one_body_mo: numpy.ndarray,
        two_body_mo: numpy.ndarray,
        eps: float = 0.0,
        n_procs: int = 1):
    global two_body_mo_pqrs_global
    two_body_mo_pqrs_global = numpy.moveaxis(
        two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
    hamiltonian_ferm_op_1 = openfermion.FermionOperator()
    hamiltonian_ferm_op_2 = openfermion.FermionOperator()
    for (p, q) in zip(*((abs(one_body_mo) > eps).nonzero())):
        p = int(p)
        q = int(q)
        pa = p * 2
        pb = p * 2 + 1
        qa = q * 2
        qb = q * 2 + 1
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pa, 1), (qa, 0)),
            one_body_mo[p][q]
        )
        hamiltonian_ferm_op_1 += openfermion.FermionOperator(
            ((pb, 1), (qb, 0)),
            one_body_mo[p][q]
        )
    n_orb = two_body_mo.shape[0]
    n_workers = min(n_procs, n_orb)
    if (n_workers != n_procs):
        print("Warning: change n_procs to %d" % (n_workers))

    chunk_size = n_orb // n_workers
    chunk_list = [chunk_size for i in range(n_workers)]
    for i in range(n_orb - chunk_size * n_workers):
        chunk_list[i] += 1

    import multiprocessing
    args_workers = []
    start_idx = 0
    end_idx = 0
    for i in range(n_workers):
        start_idx = end_idx
        end_idx += chunk_list[i]
        args_workers.append((start_idx, end_idx, eps))

    Pool = multiprocessing.Pool(n_workers)
    map_result = Pool.map(_get_hamiltonian_ferm_op_from_mo_ints_mp_worker,
                          args_workers)
    Pool.close()
    Pool.join()

    hamiltonian_ferm_op_2 = openfermion.FermionOperator()
    for i in range(n_workers):
        hamiltonian_ferm_op_2 = hamiltonian_ferm_op_2 + map_result[i]

    hamiltonian_ferm_op_1 = openfermion.normal_ordered(hamiltonian_ferm_op_1)
    hamiltonian_ferm_op_2 = openfermion.normal_ordered(hamiltonian_ferm_op_2)
    return hamiltonian_ferm_op_1, hamiltonian_ferm_op_2


def get_one_norm_from_mo_ints(one_body_mo: numpy.ndarray,
                              two_body_mo: numpy.ndarray,
                              constant: float = 0.0):
    """
    Calculate the 1-norm of the qubit Hamiltonian for a given
    one-electron MO integral in (p+, q) order and a give two-electron MO
    integral in (p+, s, q+, r) order (PySCF's ordering) without
    Fermion-to-Qubit mapping.

    Args:
        one_body_mo (numpy.ndarray): one-electron integral in (p+, q) order.
        two_body_mo (numpy.ndarray): two-electron integral in
            (p+, s, q+, r) order.
        constant (float): Nuclear repulsion energy.

    Notes:
        The integrals are for spatial-orbitals.

    References:
        [1]. Emiel Koridon et al. Phys. Rev. Research 3 (2021), 033127
    """

    n_orb = one_body_mo.shape[0]
    c_term = 0.0
    for p in range(n_orb):
        c_term += one_body_mo[p, p]
    for p in range(n_orb):
        for r in range(n_orb):
            c_term += 0.5 * two_body_mo[p, p, r, r]
    for p in range(n_orb):
        for r in range(n_orb):
            c_term += -0.25 * two_body_mo[p, r, r, p]
    c_term = abs(c_term)
    t_term = 0.0
    for p in range(n_orb):
        for q in range(n_orb):
            t_term_pq = one_body_mo[p, q]
            for r in range(n_orb):
                t_term_pq += two_body_mo[p, q, r, r]
                t_term_pq += -0.5 * two_body_mo[p, r, r, q]
            t_term += abs(t_term_pq)
    v_term = 0.0
    v_term_1 = 0.0
    v_term_2 = 0.0
    for r in range(n_orb):
        for p in range(r + 1, n_orb):
            for q in range(n_orb):
                for s in range(q + 1, n_orb):
                    v_term_1 += abs(two_body_mo[p, q, r, s] - two_body_mo[p, s, r, q])
    v_term += 0.5 * v_term_1
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    v_term_2 += abs(two_body_mo[p, q, r, s])
    v_term += 0.25 * v_term_2
    one_norm = c_term + t_term + v_term + constant
    return one_norm


def get_mo_integrals_from_molecule_and_hf_orb(
        mol: pyscf.gto.Mole,
        mo_coeff: numpy.ndarray,
        debug: bool = False,
        hcore: numpy.ndarray = None):
    """
    Notes:
        For two_body_mo, the current order is (ps|qr). A transpose like
        numpy.moveaxis(two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3]) is necessary
        to get the (p+, q+, r, s) order or [p, q, r, s] indexing.
    """

    if hcore is None:
        hcore = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    one_body_mo = functools.reduce(numpy.dot, (mo_coeff.T, hcore, mo_coeff))
    # Refer to: pyscf.ao2mo.incore.general
    eri = mol.intor("int2e")
    two_body_mo = None
    try:
        import opt_einsum
        two_body_mo = opt_einsum.contract(
            "ijkl, ip, js, kq, lr->psqr",
            eri, mo_coeff.conj(), mo_coeff,
            mo_coeff.conj(), mo_coeff)
    except ModuleNotFoundError:
        try:
            two_body_mo = numpy.einsum(
                "ijkl, ip, js, kq, lr->psqr",
                eri, mo_coeff.conj(), mo_coeff,
                mo_coeff.conj(), mo_coeff)
        except ValueError:
            print(
                "The system is too large. Please install opt_einsum and re-run init_scf().")
            raise ValueError

    if debug:
        two_body_mo_check = pyscf.ao2mo.restore(1, pyscf.ao2mo.get_mo_eri(
            mol, mo_coeff, compact=False),
            mol.nao_nr()
        )
        error = numpy.linalg.norm(two_body_mo - two_body_mo_check)
        assert(numpy.isclose(error, 0.0))
    return one_body_mo, two_body_mo


def get_spin_integrals_from_mo(one_body_mo: numpy.ndarray,
                               two_body_mo: numpy.ndarray):
    """
    Get the spin-orbital integrals from MO integrals.

    Notes:
        The output two_body_int is in (p+, q+, r, s) order.
    """
    n_orb = one_body_mo.shape[0]
    one_body_int = numpy.zeros([n_orb * 2] * 2)
    two_body_int = numpy.zeros([n_orb * 2] * 4)

    # # Original implementation.
    # for (p, q) in zip(*((abs(one_body_mo) > eps).nonzero())):
    #     one_body_int[2 * p][2 * q] = one_body_mo[p][q]
    #     one_body_int[2 * p + 1][2 * q + 1] = one_body_mo[p][q]
    # for (p, q, r, s) in zip(*((abs(two_body_mo) > eps).nonzero())):
    #     two_body_int[2 * p][2 * q][2 * r][2 * s] = \
    #         two_body_mo[p][s][q][r]
    #     two_body_int[2 * p + 1][2 * q + 1][2 * r + 1][2 * s + 1] = \
    #         two_body_mo[p][s][q][r]
    #     two_body_int[2 * p + 1][2 * q][2 * r][2 * s + 1] = \
    #         two_body_mo[p][s][q][r]
    #     two_body_int[2 * p][2 * q + 1][2 * r + 1][2 * s] = \
    #         two_body_mo[p][s][q][r]

    # Taking advantage of numpy's vectorization.
    one_body_int[0::2, 0::2] = one_body_mo
    one_body_int[1::2, 1::2] = one_body_mo
    two_body_mo_pqrs = numpy.moveaxis(two_body_mo, [0, 2, 3, 1], [0, 1, 2, 3])
    two_body_int[0::2, 0::2, 0::2, 0::2] = two_body_mo_pqrs
    two_body_int[1::2, 1::2, 1::2, 1::2] = two_body_mo_pqrs
    two_body_int[1::2, 0::2, 0::2, 1::2] = two_body_mo_pqrs
    two_body_int[0::2, 1::2, 1::2, 0::2] = two_body_mo_pqrs
    return one_body_int, two_body_int


def get_localized_mo_coeff(mf: pyscf.scf.rhf.RHF, molecule: pyscf.gto.Mole,
                           localized_orbitals: str = "None"):

    if localized_orbitals == "None":
        localized_orbitals = None

    mo_coeff = mf.mo_coeff.copy()
    basis = molecule.basis
    n_mo_occ = numpy.count_nonzero(mf.mo_occ > 0)
    if localized_orbitals is not None:
        if localized_orbitals in ["iao", "IAO"]:
            print("Use IAO localization.")

            if basis != "sto3g":
                print("%s at large basis may lead to incorrect result. \
This may be fixed in the future." % (localized_orbitals))

            # # Still remains a problem for larger basis set.
            # # occupied orbitals
            # mo_occ_occupied = mf.mo_coeff[:, :n_mo_occ]
            # a_occupied = pyscf.lo.iao.iao(molecule, mo_occ_occupied)
            # a_occupied = pyscf.lo.vec_lowdin(a_occupied, mf.get_ovlp())
            # mo_occ_occupied = a_occupied.T.dot(
            #     mf.get_ovlp().dot(mo_occ_occupied))
            # mo_coeff_occupied = a_occupied.copy()
            # # virtual_orbitals
            # mo_occ_virtual = mf.mo_coeff[:, n_mo_occ:]
            # a_virtual = pyscf.lo.iao.iao(molecule, mo_occ_virtual)
            # a_virtual = pyscf.lo.vec_lowdin(a_virtual, mf.get_ovlp())
            # mo_occ_virtual = a_virtual.T.dot(
            #     mf.get_ovlp().dot(mo_occ_virtual))
            # mo_coeff_virtual = a_virtual.copy()
            # # Stack the occupied orbitals and virtual orbitals
            # mo_coeff = numpy.hstack((mo_coeff_occupied, mo_coeff_virtual))
            # mo_coeff = mf.get_ovlp().dot(mo_coeff).T
            # mo_coeff = pyscf.lo.vec_lowdin(mo_coeff, mf.get_ovlp())

            # # Still remains a problem for larger basis set.
            # Original
            mo_occ = mf.mo_coeff[:, :]  # mf.mo_coeff[:, :n_mo_occ]
            a = pyscf.lo.iao.iao(molecule, mo_occ)
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            mo_coeff = a.copy()

            # # Try method
            # a = pyscf.lo.iao.iao(molecule, mf.mo_coeff)
            # a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            # mo_coeff = _scdm(mf.mo_coeff, mf.get_ovlp(), a)
        elif localized_orbitals in ["ibo", "IBO"]:
            print("Use IBO localization.")

            if basis != "sto3g":
                print("%s at large basis may lead to incorrect result. \
This may be fixed in the future." % (localized_orbitals))

            # # Still remains a problem for larger basis set.
            # # occupied orbitals
            # mo_occ = mf.mo_coeff[:, :n_mo_occ]
            # a = pyscf.lo.ibo.ibo(molecule, mo_occ)
            # a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            # mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            # mo_coeff_occupied = mo_occ.copy()  # a.copy()
            # # virtual_orbitals
            # mo_occ = mf.mo_coeff[:, n_mo_occ:]
            # a = pyscf.lo.ibo.ibo(molecule, mo_occ)
            # a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            # mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            # mo_coeff_virtual = mo_occ.copy()  # a.copy()
            # # Stack the occupied orbitals and virtual orbitals
            # mo_coeff = numpy.hstack((mo_coeff_occupied, mo_coeff_virtual))
            # mo_coeff = pyscf.lo.vec_lowdin(mo_coeff, mf.get_ovlp())

            # # Still remains a problem for larger basis set.
            # Original
            mo_occ = mf.mo_coeff[:, :]
            a = pyscf.lo.iao.iao(molecule, mo_occ)
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            a = pyscf.lo.ibo.ibo(molecule, mo_occ, iaos=a, s=mf.get_ovlp())
            a = pyscf.lo.vec_lowdin(a, mf.get_ovlp())
            mo_occ = a.T.dot(mf.get_ovlp().dot(mo_occ))
            mo_coeff = a.copy()
        elif localized_orbitals in ["cholesky", "Cholesky", "Cho"]:
            print("Using cholesky localization.")
            mo_coeff_occupied = pyscf.lo.cholesky_mos(
                mf.mo_coeff[:, :n_mo_occ])
            mo_coeff_virtual = pyscf.lo.cholesky_mos(
                mf.mo_coeff[:, n_mo_occ:])
            mo_coeff = numpy.hstack((mo_coeff_occupied, mo_coeff_virtual))
        elif localized_orbitals in ["NAO", "nao"]:
            print("Using NAO localization.")
            mo_coeff = pyscf.lo.orth_ao(mf, method="nao")
        elif localized_orbitals in ["er", "ER", "Edmiston-Ruedenberg"]:
            print("Using Edmiston-Ruedenberg localization.")
            mo_coeff = pyscf.lo.ER(molecule, mo_coeff=mf.mo_coeff).kernel()
            mo_coeff = pyscf.lo.vec_lowdin(mo_coeff, mf.get_ovlp())
        elif localized_orbitals in ["boys", "Boys", "Foster-Boys"]:
            print("Using Foster-Boys localization.")
            mo_coeff = pyscf.lo.Boys(molecule, mo_coeff=mf.mo_coeff).kernel()
            mo_coeff = pyscf.lo.vec_lowdin(mo_coeff, mf.get_ovlp())
        elif type(localized_orbitals) is numpy.ndarray:
            assert(localized_orbitals.shape == mo_coeff.shape)
            mo_coeff = mo_coeff.dot(localized_orbitals)
        else:
            raise ValueError("Localization orbital %s \
not supported!" % (localized_orbitals))
    return mo_coeff


def get_active_space_effective_mo_integrals(
        one_body_mo: numpy.ndarray,
        two_body_mo: numpy.ndarray,
        freeze_indices_mo: list = None,
        active_indices_mo: list = None):
    """
    Calculate active-space integrals with PySCF ordering.

    Notes:
        The indices and integrals are for spatial MOs.

    References:
        [1]. Emiel Koridon et al. PHYSICAL REVIEW RESEARCH 3(2021), 033127
    """
    one_body_mo_active = None
    two_body_mo_active = None
    core_correction = 0.0
    if freeze_indices_mo is None:
        freeze_indices_mo = []
    if active_indices_mo is None:
        one_body_mo_active = one_body_mo.copy()
        two_body_mo_active = two_body_mo.copy()
    else:
        one_body_mo_new = numpy.copy(one_body_mo)
        for p in freeze_indices_mo:
            core_correction += 2. * one_body_mo[p][p]
            for q in freeze_indices_mo:
                core_correction += (2. * two_body_mo[q][q][p][p] -
                                    two_body_mo[q][p][p][q])
        for uu in active_indices_mo:
            for vv in active_indices_mo:
                for ii in freeze_indices_mo:
                    one_body_mo_new[uu][vv] += (
                        2. * two_body_mo[ii][ii][uu][vv] -
                        two_body_mo[ii][vv][uu][ii]
                    )
        one_body_mo_active = one_body_mo_new[numpy.ix_(
            active_indices_mo, active_indices_mo)]
        two_body_mo_active = two_body_mo[numpy.ix_(
            active_indices_mo, active_indices_mo,
            active_indices_mo, active_indices_mo)]
    return one_body_mo_active, two_body_mo_active, core_correction


def get_molecule_str(molecule: pyscf.gto.Mole):
    atom_coords: numpy.ndarray = molecule.atom_coords()
    n_atoms = atom_coords.shape[0]
    mol_symbol = "".join([molecule.atom_symbol(i) for i in range(n_atoms)])
    coords_hash = hash(atom_coords.tobytes())
    basis = molecule.basis
    mol_name = "mol_{}_coord_{}_basis_{}".format(mol_symbol, coords_hash, basis)
    return mol_name


def init_scf_pbc(geometry, lattice_vec,
                 nks=[1, 1, 1], kshift=[0., 0., 0.],
                 basis: str = "gth-szv", pseudo: str = "gth-pade",
                 spin: int = 0,
                 k2g: bool = False,
                 run_krccsd: bool = False,
                 fermion_to_qubit_mapping: str = "jw"):
    """
    Generate the system Hamiltonian and other quantities for periodic systems.

    Args:
        geometry (list): The structure of the molecule.
        lattice_vec (list): Lattice vectors.
        nks (list): The sampling grid for k-points.
        kshift (list): Center k-point shift.
        basis (str): Basis set for SCF calculations.
        spin (int): Describes multiplicity of the molecular system.
        k2g (bool): Whether to use K2G transformation. Reference:
            Jie Liu et al. J. Chem. Theory Comput. 16, 6904 (2020)
        run_kccsd (bool): Whether to run KRCCSD.
        fermion_to_qubit_mapping (str): The fermion-to-qubit mapping
            for Hamiltonian.

    Returns:
        molecule (pyscf.gto.M object): Contains various properties
            of the system.
        n_qubits (int): Number of qubits in the Hamiltonian.
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        occ_indices_spin (int): Occupied indices of spin orbitals.
        hamiltonian_fermOp (openfermion.FermionOperator): Fermionic
            Hamiltonian.
        hamiltonian_qubitOp (openfermion.QubitOperator): Qubit Hamiltonian
            under JW transformation.
        kpts (numpy.ndarray): Coordinates of k-points in reciprocal space.
        k2m (list): Describes the map from k-point index to
            molecular orbital index.
        m2k (list): Describes the map from molecular orbital index to
            k-point index.
        kconserv (list): A list describing momentum conservation in k-space.

    Usage of k2m and m2k:
        k2m:
        k2m[kpoint idx][local idx of spatial orbital]:
            global idx of spatial orbital
        The convention is, sorted by energy,
            the mo orbital with lower energy should have lower index.
        m2k: [ [kpoint idx, local idx of spatial orbital (of this kpoint)],
                ... ]
        m2k[global idx of spatial idx][0]: kpoint idx
        m2k[global idx of spatial idx][1]: local idx of spatial orbital

    """
    import pyscf.pbc
    import pyscf.pbc.cc
    from pyscf.pbc.tools import k2gamma

    cell = pyscf.pbc.gto.M(
        atom=geometry,
        a=lattice_vec,
        basis=basis,
        pseudo=pseudo,
        spin=spin
    )

    kpts = cell.make_kpts(nks, scaled_center=kshift, wrap_around=True)
    n_kpts = len(kpts)
    kconserv = pyscf.pbc.tools.get_kconserv(cell, kpts)
    kmf = pyscf.pbc.scf.KRHF(cell, kpts, exxdiv=None)
    if pseudo is None or pseudo == "None":
        kmf = kmf.mix_density_fit()
    print("Running KRHF...")
    kmf.kernel()

    energy_KRHF = kmf.e_tot
    energy_nuc = cell.energy_nuc()
    print("Hartree-Fock energy: %20.16f Ha" % (energy_KRHF))

    if run_krccsd:
        print("Running KRCCSD...")
        kmf_cc = pyscf.pbc.cc.KRCCSD(kmf)
        kmf_cc.kernel()
        energy_KRCCSD = kmf_cc.e_tot
        print("CCSD energy: %20.16f Ha" % (energy_KRCCSD))

    n_orb_unitcell = cell.nao_nr()
    n_orb = cell.nao_nr() * n_kpts
    n_orb_occ = sum(cell.nelec) // 2 * n_kpts
    occ_indices_spin = [i for i in range(n_orb_occ * 2)]
    hcore = kmf.get_hcore(cell, kpts)
    mo_coeff = kmf.mo_coeff
    mo_energy = numpy.hstack(kmf.mo_energy)
    sorted_idx = numpy.argsort(mo_energy)
    if (k2g is True):
        m2k = [[i // n_orb_unitcell, i % n_orb_unitcell]
               for i in range(n_orb)]
    else:
        m2k = [[sorted_idx[i] // n_orb_unitcell, sorted_idx[i] %
                n_orb_unitcell] for i in range(n_orb)]
    k2m = []
    for k_idx in range(n_kpts):
        k2m_klist = []
        for orb_idx in range(n_orb_unitcell):
            k2m_klist.append(m2k.index([k_idx, orb_idx]))
        k2m.append(k2m_klist)
    if (k2g is True):
        m2k = [[0, 0] for i in range(n_orb)]

    one_body_mo = numpy.zeros([n_orb] * 2, dtype=numpy.complex128)
    two_body_mo = numpy.zeros([n_orb] * 4, dtype=numpy.complex128)

    one_body_int = numpy.zeros([n_orb * 2] * 2, dtype=numpy.complex128)
    two_body_int = numpy.zeros([n_orb * 2] * 4, dtype=numpy.complex128)

    one_body = numpy.array(
        [functools.reduce(numpy.dot, (
            mo_coeff_k.T.conj(), hcore[k], mo_coeff_k))
            for k, mo_coeff_k in enumerate(mo_coeff)])

    two_body = []
    for kp in range(n_kpts):
        for kq in range(n_kpts):
            for kr in range(n_kpts):
                ks = kconserv[kp][kq][kr]
                two_body_k = kmf.with_df.ao2mo(
                    [mo_coeff[i] for i in (kp, kq, kr, ks)],
                    [kpts[i] for i in (kp, kq, kr, ks)], compact=False)
                two_body_k = two_body_k.reshape([n_orb_unitcell] * 4)
                two_body.append(two_body_k)
    two_body = numpy.array(two_body)

    for k_idx in range(n_kpts):
        for p in range(n_orb_unitcell):
            for q in range(n_orb_unitcell):
                p_idx_global = k2m[k_idx][p]
                q_idx_global = k2m[k_idx][q]
                one_body_mo[p_idx_global][q_idx_global] = \
                    one_body[k_idx][p][q] / n_kpts

    for kp_idx in range(n_kpts):
        for kq_idx in range(n_kpts):
            for kr_idx in range(n_kpts):
                ks_idx = kconserv[kp_idx][kq_idx][kr_idx]
                offset_k_idx = ((kp_idx * n_kpts) + kq_idx) * n_kpts + kr_idx
                for p in range(n_orb_unitcell):
                    for q in range(n_orb_unitcell):
                        for r in range(n_orb_unitcell):
                            for s in range(n_orb_unitcell):
                                p_idx_global = k2m[kp_idx][p]
                                q_idx_global = k2m[kq_idx][q]
                                r_idx_global = k2m[kr_idx][r]
                                s_idx_global = k2m[ks_idx][s]
                                two_body_mo[p_idx_global][q_idx_global][
                                    r_idx_global][s_idx_global] = \
                                    two_body[offset_k_idx][p][q][r][s] / \
                                    n_kpts ** 2

    if (k2g is True):
        scell, E_g, C, mo_phase = k2gamma.mo_k2gamma(
            cell, kmf.mo_energy, kmf.mo_coeff, kpts)
        U = numpy.zeros((n_orb, n_orb), dtype=numpy.complex128)
        for k in range(n_kpts):
            for m in range(n_orb_unitcell):
                U[k2m[k][m], :] = mo_phase[k, m, :]
        h = numpy.einsum("qp,qr,rs->ps", U.conj(), one_body_mo, U)
        for kp in range(n_kpts):
            for kq in range(n_kpts):
                for kr in range(n_kpts):
                    for ks in range(n_kpts):
                        eri_kpt = kmf.with_df.ao2mo(
                            [kmf.mo_coeff[i] for i in (kp, kq, kr, ks)],
                            [kpts[i] for i in (kp, kq, kr, ks)], compact=False)
                        eri_kpt = eri_kpt.reshape([n_orb_unitcell]*4)
                        for i in range(n_orb_unitcell):
                            for j in range(n_orb_unitcell):
                                for k in range(n_orb_unitcell):
                                    for l in range(n_orb_unitcell):
                                        p = k2m[kp][i]
                                        q = k2m[kq][j]
                                        r = k2m[kr][k]
                                        s = k2m[ks][l]
                                        two_body_mo[p, q, r, s] = \
                                            eri_kpt[i, j, k, l] / n_kpts ** 2
        g = two_body_mo
        g = numpy.einsum("pqrs,pl->lqrs", g, U.conj())
        g = numpy.einsum("lqrs,qm->lmrs", g, U)
        g = numpy.einsum("lmrs,rn->lmns", g, U.conj())
        g = numpy.einsum("lmns,so->lmno", g, U)
        one_body_mo = h
        two_body_mo = g

    for p in range(n_orb):
        pa = p * 2
        pb = p * 2 + 1
        for q in range(n_orb):
            qa = q * 2
            qb = q * 2 + 1
            one_body_int[pa][qa] = one_body_mo[p][q]
            one_body_int[pb][qb] = one_body_mo[p][q]

    for p in range(n_orb):
        pa = 2 * p
        pb = 2 * p + 1
        for q in range(n_orb):
            qa = 2 * q
            qb = 2 * q + 1
            for r in range(n_orb):
                ra = 2 * r
                rb = 2 * r + 1
                for s in range(n_orb):
                    sa = 2 * s
                    sb = 2 * s + 1
                    two_body_int[pa][qa][sa][ra] = two_body_mo[p][r][q][s]
                    two_body_int[pb][qb][sb][rb] = two_body_mo[p][r][q][s]
                    two_body_int[pb][qa][sa][rb] = two_body_mo[p][r][q][s]
                    two_body_int[pa][qb][sb][ra] = two_body_mo[p][r][q][s]

    hamiltonian_fermOp_1 = openfermion.FermionOperator()
    hamiltonian_fermOp_2 = openfermion.FermionOperator()

    for p in range(n_orb * 2):
        for q in range(n_orb * 2):
            hamiltonian_fermOp_1 += openfermion.FermionOperator(
                ((p, 1), (q, 0)),
                one_body_int[p][q]
            )
    for p in range(n_orb * 2):
        for q in range(n_orb * 2):
            for r in range(n_orb * 2):
                for s in range(n_orb * 2):
                    hamiltonian_fermOp_2 += openfermion.FermionOperator(
                        ((p, 1), (q, 1), (r, 0), (s, 0)),
                        two_body_int[p][q][r][s] * 0.5
                    )

    hamiltonian_fermOp_1 = openfermion.normal_ordered(hamiltonian_fermOp_1)
    hamiltonian_fermOp_2 = openfermion.normal_ordered(hamiltonian_fermOp_2)
    hamiltonian_fermOp = hamiltonian_fermOp_1 + hamiltonian_fermOp_2
    hamiltonian_fermOp += energy_nuc

    hamiltonian_qubitOp = None
    if fermion_to_qubit_mapping is not None:
        if fermion_to_qubit_mapping == "jw":
            hamiltonian_qubitOp = openfermion.jordan_wigner(hamiltonian_fermOp)
        else:
            raise NotImplementedError("Fermion-to-qubit mapping {} not \
implemented.".format(fermion_to_qubit_mapping))
    n_qubits = openfermion.count_qubits(hamiltonian_fermOp)

    # TODO: Change the first return value to openfermion's MolecularData
    # or other suitable for PBC
    return cell, n_qubits, n_orb, n_orb_occ, occ_indices_spin, \
        hamiltonian_fermOp, hamiltonian_qubitOp, \
        kpts, k2m, m2k, kconserv


def _generate_real_unitary_matrix(dim: int):
    mat = numpy.random.rand(dim * dim)
    mat = mat.reshape([dim, dim])
    u = scipy.linalg.expm((mat - mat.T.conj()))

    # u = scipy.stats.ortho_group.rvs(dim=dim)
    assert(numpy.isclose(
        numpy.linalg.norm(numpy.eye(dim) - u.dot(u.T.conj())), 0.0))
    return u


def _optimize_mo_coeff(mo_coeff: numpy.ndarray):
    import torch
    import torch.linalg
    mo_coeff_torch = torch.from_numpy(mo_coeff)
    dtype = mo_coeff_torch.dtype
    dim = mo_coeff_torch.shape[0]
    # mat = torch.eye(dim, dtype=dtype)
    # mat = torch.rand([dim, dim], dtype=dtype)
    mat = torch.zeros([dim, dim], dtype=dtype)
    mat[0][0] = 1.
    mat[2][1] = 1.
    mat[4][2] = 1.
    mat[1][3] = 1.
    mat[3][4] = 1.
    mat[5][5] = 1.
    mat.requires_grad_(True)

    def _ortho_from_mat(mat: torch.Tensor):
        dim = mat.shape[0]
        dtype = mat.dtype
        H = torch.eye(dim, dtype=dtype, requires_grad=False)
        for n in range(dim):
            x = mat[n][:dim - n]
            norm2 = torch.dot(x, x)
            x0 = x[0].item()
            # random sign, 50/50, but chosen carefully to avoid roundoff error
            # D = torch.from_numpy(numpy.array(numpy.sign(x[0].numpy()))) \
            #     if x[0] != 0 else 1
            D = torch.sign(x[0]) \
                if x[0] != 0 else 1
            x[0] = x[0] + D * torch.sqrt(norm2)
            x = x / torch.sqrt((norm2 - x0**2 + x[0]**2) / 2.)
            # Householder transformation
            tmp = -D * (H[:, n:] - torch.outer(
                torch.matmul(H[:, n:], x), x))
            H[:, n:] = H[:, n:].detach() + tmp
        return H

    def _ortho_qr(mat: torch.Tensor):
        q, r = torch.linalg.qr(mat)
        return q

    ortho_fn = _ortho_qr

    def _cost_fn(mat: torch.Tensor):
        # mat = mat / torch.linalg.norm(mat)
        # mat = mat - mat.T.conj()
        # u = torch.matrix_exp(mat)
        mat = mat / torch.linalg.norm(mat)
        u = _ortho_qr(mat)
        err = torch.eye(dim, dtype=dtype) - u.T.conj().mm(u)
        err = torch.linalg.norm(err).item()
        # assert(numpy.isclose(err, 0.0))
        mo_coeff_torch_new = mo_coeff_torch.mm(u)
        dm = mo_coeff_torch_new.T.conj().mm(mo_coeff_torch_new)
        y = (torch.sum(dm) - torch.trace(dm)) / torch.sum(dm)
        return y

    optimizer = torch.optim.Adam([mat], lr=0.5)

    disp = True
    maxiter = 0  # 1000
    ftol = 1e-8
    gtol = 1e-5
    iter_count = 0
    f_diff = ftol * 9999
    f_last = None
    f_val_dict = {}
    g_norm_dict = {}

    def _closure():
        global f_last
        optimizer.zero_grad()
        y = _cost_fn(mat)
        y.backward()
        hash_val = hash(mat)
        y_np = y.detach().numpy()
        grad_np = mat.grad.detach().numpy()
        f_val_dict.update({hash_val: y_np})
        g_norm_dict.update(
            {hash_val: numpy.linalg.norm(grad_np)})
        return y

    finish_type = 0
    while iter_count < maxiter:
        if iter_count > 0:
            if grad_last <= gtol:
                finish_type = 1
                break
            if f_diff <= ftol:
                finish_type = 2
                break
        hash_val = hash(mat)
        optimizer.step(_closure)
        grad_last = g_norm_dict[hash_val]
        f_cur = f_val_dict[hash_val]
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

    u_mat = ortho_fn(mat).detach().numpy()

    return u_mat


def init_scf(geometry, basis="sto-3g", spin=0,
             freeze_indices_spatial=[],
             active_indices_spatial=[],
             run_fci: bool = False,
             localized_orbitals: str = None,
             use_symmetry: bool = False,
             override_symmetry_group: str = None,
             fermion_to_qubit_mapping: str = "jw",
             sort_orbital_for_dmrg: bool = False,
             run_rccsd: bool = False,
             return_spin_orb_int: bool = False,
             n_procs: int = 1,
             freeze_occ_indices_spin: list = None,
             freeze_vir_indices_spin: list = None):
    """
    Generate the system Hamiltonian and other quantities for a give molecule.

    Args:
        geometry (list): The structure of the molecule.
        basis (str): Basis set for SCF calculations.
        spin (int): Describes multiplicity of the molecular system.
        freeze_indices_spatial (list): Occupied indices (frozen orbitals)
            of spatial orbitals.
        active_indices_spatial (list): Active indices of spatial
            orbitals.
        run_fci (bool): Whether FCI calculation is performed.
        localized_orbitals (str): Whether to use localized orbitals. If
            is None, no localization if performed.
        use_symmetry (bool): Whether to use symmetry and return the character
            table of orbitals. Exclusive with localized_orbitals.
        override_symmetry_group (str): Override the symmetry point group
            determined by PySCF.
        fermion_to_qubit_mapping (str): The fermion-to-qubit mapping
            for Hamiltonian.
        sort_orbital_for_dmrg (bool): Reorder the orbitals according to its
            electronic state symmetry. Can be used together with the
            override_symmetry_group.
        run_rccsd (bool): Whether the RCCSD is performed.
        return_spin_orb_int (bool): Whether to return the one- and two-electron
            integrals in spin-orbital basis.
        n_procs (int): Number of processes to contruct the Hamiltonian.
        freeze_occ_indices_spin (list): Indices of spin orbitals which are
            assumed to be occupied.
        freeze_vir_indices_spin (list): Indices of spin orbitals which are
            assumed to be unoccupied.

    Returns:
        molecule (pyscf.gto.M object): Contains various properties
            of the system.
        n_qubits (int): Number of qubits in the Hamiltonian.
        n_orb (int): Number of spatial orbitals.
        n_orb_occ (int): Number of occupied spatial orbitals.
        occ_indices_spin (int): Occupied indices of spin orbitals.
        hamiltonian_fermOp (openfermion.FermionOperator): Fermionic
            Hamiltonian.
        hamiltonian_qubitOp (openfermion.QubitOperator): Qubit Hamiltonian
            under JW transformation.
        orbsym (numpy.ndarray): The irreducible representation of each
            spatial orbital. Only returns when use_symmetry is True.
        prod_table (numpy.ndarray): The direct production table of orbsym.
            Only returns when use_symmetry is True.

    """
    eps = 0.0

    if localized_orbitals == "None":
        localized_orbitals = None

    if localized_orbitals is not None:
        if use_symmetry is True:
            print("Using localized orbitals will cause the returned orbsym \
and prod_table invalid. Handle with care!")

    molecule = pyscf.gto.M(
        atom=geometry,
        basis=basis,
        spin=spin
    )

    if use_symmetry or sort_orbital_for_dmrg:
        if override_symmetry_group is not None:
            molecule = pyscf.gto.M(
                atom=geometry,
                basis=basis,
                spin=spin,
                symmetry=override_symmetry_group
            )
        else:
            molecule = pyscf.gto.M(
                atom=geometry,
                basis=basis,
                spin=spin,
                symmetry=True
            )
        print("Use symmetry. Molecule point group: %s" % (molecule.topgroup))

    mf = pyscf.scf.RHF(molecule)
    print("Running RHF...")
    mf.kernel()
    mo_coeff = mf.mo_coeff

    mo_coeff = get_localized_mo_coeff(
        mf, molecule, localized_orbitals)

    # ***DEBUG***
    # u = _optimize_mo_coeff(mo_coeff)
    # print("mo_coeff Before: \n", mo_coeff)
    # mo_coeff = mo_coeff.dot(u)
    # print("mo_coeff After: \n", mo_coeff)

    if run_rccsd:
        print("Running RCCSD")
        mf_cc = pyscf.cc.RCCSD(mf)
        mf_cc.kernel()

    energy_RHF = mf.e_tot
    energy_nuc = molecule.energy_nuc()
    print("Hartree-Fock energy: %20.16f Ha" % (energy_RHF))
    if run_rccsd:
        energy_RCCSD = mf_cc.e_tot
        print("CCSD energy: %20.16f Ha" % (energy_RCCSD))

    if run_fci:
        mf_fci = pyscf.fci.FCI(mf)
        energy_fci = mf_fci.kernel()[0]
        print("FCI energy: %20.16f Ha" % (energy_fci))

    # return None # TODO WYJ
    n_orb = mo_coeff.shape[1]  # molecule.nao_nr()
    n_orb_occ = sum(molecule.nelec) // 2
    occ_indices_spin = [i for i in range(molecule.nelectron)]
    hcore = mf.get_hcore()
    one_body_mo, two_body_mo = get_mo_integrals_from_molecule_and_hf_orb(
        molecule, mo_coeff)
    core_correction = 0.0

#     if (len(freeze_indices_spatial) == 0) \
#             and (len(active_indices_spatial) == 0):
#         pass
#     elif (len(active_indices_spatial) != 0):
#         n_orb = len(active_indices_spatial)
#         n_orb_occ = (sum(molecule.nelec) -
#                      2 * len(freeze_indices_spatial)) // 2
#         occ_indices_spin = [i for i in range(n_orb_occ * 2)]
#         one_body_mo_new = numpy.copy(one_body_mo)
#         for p in freeze_indices_spatial:
#             core_correction += 2. * one_body_mo[p][p]
#             for q in freeze_indices_spatial:
#                 core_correction += (2. * two_body_mo[p][q][q][p] -
#                                     two_body_mo[p][q][p][q])
#         for uu in active_indices_spatial:
#             for vv in active_indices_spatial:
#                 for ii in freeze_indices_spatial:
#                     one_body_mo_new[uu][vv] += (
#                         2. * two_body_mo[ii][ii][uu][vv] -
#                         two_body_mo[ii][vv][uu][ii]
#                     )
#         one_body_mo = one_body_mo_new[numpy.ix_(
#             active_indices_spatial, active_indices_spatial)]
#         two_body_mo = two_body_mo.transpose(0, 2, 3, 1)[numpy.ix_(
#             active_indices_spatial, active_indices_spatial,
#             active_indices_spatial, active_indices_spatial)]
#         two_body_mo = two_body_mo.transpose(0, 3, 1, 2)
#     else:
#         print("active_indices_spatial must not be empty \
# if freeze_indices_spatial is non-empty !")
#         raise ValueError

    if len(freeze_indices_spatial) != 0 or len(active_indices_spatial) != 0:
        print("The freeze_indices_spatial and active_indices_spatial are\
currently not supported. Please use freeze_occ_indices_spin and\
freeze_vir_indices_spin instead.")

    # one_body_int, two_body_int = get_spin_integrals_from_mo(
    #     one_body_mo, two_body_mo)

    # hamiltonian_fermOp_1, hamiltonian_fermOp_2 = \
    #     get_hamiltonian_ferm_op_from_ints(
    #         one_body_int, two_body_int, eps)

    hamiltonian_ferm_op_1, hamiltonian_ferm_op_2 = \
        get_hamiltonian_ferm_op_from_mo_ints_mp(
            one_body_mo, two_body_mo, eps,
            n_procs=n_procs)

    hamiltonian_ferm_op = hamiltonian_ferm_op_1 + hamiltonian_ferm_op_2
    hamiltonian_ferm_op += energy_nuc + core_correction

    if sort_orbital_for_dmrg:
        print("Reordering orbitals according to orbital symmetry.")
        orbsym = mf.orbsym if hasattr(mf, "orbsym") else \
            numpy.zeros(n_orb, numpy.int32)
        sorted_args = list(numpy.argsort(orbsym))

        def _order_function(idx_ori: int, *args):
            # We // 2 or % 2 here since the idx_ori are for spin
            # orbitals.
            # idx_new = int(sorted_args[idx_ori // 2] * 2 + idx_ori % 2)
            # return idx_new
            idx_new = int(sorted_args.index(idx_ori // 2) * 2 + idx_ori % 2)
            return idx_new
        hamiltonian_ferm_op = openfermion.reorder(
            hamiltonian_ferm_op, order_function=_order_function)

        mo_occ_reordered = mf.mo_occ[sorted_args]
        occ_indices_spin = []
        for i in range(n_orb):
            if mo_occ_reordered[i] == 2:
                occ_indices_spin.append(2 * i)
                occ_indices_spin.append(2 * i + 1)
            elif mo_occ_reordered[i] == 1:
                occ_indices_spin.append(2 * i)
            else:
                pass

    if freeze_occ_indices_spin is not None and \
            freeze_vir_indices_spin is not None:
        hamiltonian_ferm_op = openfermion.freeze_orbitals(
            hamiltonian_ferm_op,
            occupied=freeze_occ_indices_spin,
            unoccupied=freeze_vir_indices_spin)
        n_qubits = openfermion.count_qubits(hamiltonian_ferm_op)
        occ_indices_spin_tmp = [
            i for i in occ_indices_spin
            if i not in freeze_occ_indices_spin + freeze_vir_indices_spin]
        min_occ_index_spin = min(occ_indices_spin_tmp)
        occ_indices_spin = [i - min_occ_index_spin
                            for i in occ_indices_spin_tmp]
        n_orb = n_qubits // 2
        n_orb_occ = len(occ_indices_spin) // 2 + len(occ_indices_spin) % 2

    hamiltonian_qubit_op = None
    if fermion_to_qubit_mapping is not None:
        if fermion_to_qubit_mapping == "jw":
            hamiltonian_qubit_op = openfermion.jordan_wigner(hamiltonian_ferm_op)
        else:
            raise NotImplementedError("Fermion-to-qubit mapping {} not \
implemented.".format(fermion_to_qubit_mapping))
    n_qubits = n_orb * 2  # openfermion.count_qubits(hamiltonian_fermOp)

    returned_vals = [molecule, n_qubits, n_orb, n_orb_occ, occ_indices_spin,
                     hamiltonian_ferm_op, hamiltonian_qubit_op]

    if use_symmetry:
        orbsym = mf.orbsym if hasattr(mf, "orbsym") else \
            numpy.zeros(n_orb, numpy.int32)
        # prod_table = pyscf.symm.direct_prod(
        #     numpy.unique(orbsym),
        #     numpy.unique(orbsym),
        #     molecule.topgroup)
        n_sym_ops = numpy.max(orbsym)
        prod_table = pyscf.symm.direct_prod(
            numpy.arange(n_sym_ops + 1),
            numpy.arange(n_sym_ops + 1),
            molecule.topgroup if override_symmetry_group is None
            else override_symmetry_group)
        # prod_table = pyscf.symm.direct_prod(
        #     numpy.arange(len(pyscf.symm.symm_ops(molecule.topgroup))),
        #     numpy.arange(len(pyscf.symm.symm_ops(molecule.topgroup))),
        #     molecule.topgroup)
        if freeze_occ_indices_spin is not None and \
                freeze_vir_indices_spin is not None:
            n_orb_ori = len(orbsym)
            except_indices_spin = freeze_occ_indices_spin + \
                freeze_vir_indices_spin
            orbsym = [orbsym[i] for i in range(n_orb_ori) if
                      (2 * i not in except_indices_spin and
                       2 * i + 1 not in except_indices_spin)]
            pass
        returned_vals.append(orbsym)
        returned_vals.append(prod_table)

    if return_spin_orb_int:
        one_body_int, two_body_int = get_spin_integrals_from_mo(
            one_body_mo=one_body_mo,
            two_body_mo=two_body_mo)
        returned_vals.append(one_body_int)
        returned_vals.append(two_body_int)

    # TODO: Change the first return value to openfermion's MolecularData
    return tuple(returned_vals)
