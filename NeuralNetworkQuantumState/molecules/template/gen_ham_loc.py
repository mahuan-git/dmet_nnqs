"""
This example tries to reproduce the benzene calculation in Ref.[1].

References:
    [1]. Janus J. Eriksen et al. J. Phys. Chem. Lett. 2020, 11, 20, 8922-8929
"""

import sys
sys.path.append("/root/wuyangjun/NeuralNetworkQuantumState/molecules")

import pyscf
import pyscf.cc
import pyscf.mp
import pyscf.lo
import pyscf.tools.molden
import openfermion

import pyscf_helper
import utils

# The structure in Ref.[1] table S1.
geometry_benzene_paper = [
    ["C", [+0.000000, +1.396792, 0.000000]],
    ["C", [+0.000000, -1.396792, 0.000000]],
    ["C", [+1.209657, +0.698396, 0.000000]],
    ["C", [-1.209657, -0.698396, 0.000000]],
    ["C", [-1.209657, +0.698396, 0.000000]],
    ["C", [+1.209657, -0.698396, 0.000000]],
    ["H", [+0.000000, +2.484212, 0.000000]],
    ["H", [+2.151390, +1.242106, 0.000000]],
    ["H", [-2.151390, -1.242106, 0.000000]],
    ["H", [-2.151390, +1.242106, 0.000000]],
    ["H", [+2.151390, -1.242106, 0.000000]],
    ["H", [+0.000000, -2.484212, 0.000000]],
]

geometry = [('H', (2, 0, 0)), ('H', (3, 0, 0))] # H2
geometry = [('Na', (5.4641, 0.25, 0)), ('Na', (2, 0.25, 0)), ('O', (4.5981, 0.75, 0)), ('O', (2.866, 0.75, 0)), ('O', (3.732, -0.75, 0)), ('C', (3.732, 0.25, 0))] # CNa2O3

if __name__ == "__main__":
    n_procs = 4
    # geometry = geometry_benzene_paper
    print("Geometry: \n", geometry)
    basis = "sto3g"  # "ccpvdz"
    mol = pyscf.gto.M(atom=geometry, basis=basis, symmetry=True)
    mf = pyscf.scf.RHF(mol).run()
    mf_cc = pyscf.cc.RCCSD(mf).run()
    mf_mp = pyscf.mp.MP2(mf).run()
    print("Hartree-Fock: %20.16f Ha" % (mf.e_tot))
    print("CCSD: %20.16f Ha" % (mf_cc.e_tot))
    # print("CCSDT: %20.16f Ha" % (mf_cc.e_tot + mf_cc.ccsd_t()))
    print("MP2: %20.16f Ha" % (mf_mp.e_tot))

    mo_coeff = mf.mo_coeff
    # mo_coeff_frz = mo_coeff[:, 0:6].copy()
    # mo_coeff_occ = mo_coeff[:, 6:18].copy()
    # mo_coeff_vir = mo_coeff[:, 18:].copy()
    # mo_coeff_occ_loc = pyscf.lo.ER(mol, mo_coeff_occ).kernel()
    # mo_coeff_vir_loc = pyscf.lo.ER(mol, mo_coeff_vir).kernel()
    # mo_coeff_loc = mo_coeff.copy()
    # mo_coeff_loc[:, 0:6] = mo_coeff_frz
    # mo_coeff_loc[:, 6:18] = mo_coeff_occ_loc
    # mo_coeff_loc[:, 18:] =  mo_coeff_vir_loc

    # pyscf.tools.molden.from_mo(mol, "cmo.molden", mo_coeff)
    # pyscf.tools.molden.from_mo(mol, "er_frz.molden", mo_coeff_frz)
    # pyscf.tools.molden.from_mo(mol, "er_occ.molden", mo_coeff_occ_loc)
    # pyscf.tools.molden.from_mo(mol, "er_vir.molden", mo_coeff_vir_loc)

    # one_body_mo, two_body_mo = pyscf_helper.get_mo_integrals_from_molecule_and_hf_orb(
    #     mol=mol, mo_coeff=mo_coeff_loc)
    one_body_mo, two_body_mo = pyscf_helper.get_mo_integrals_from_molecule_and_hf_orb(
        mol=mol, mo_coeff=mo_coeff)
    # one_body_mo_cas, two_body_mo_cas, core_correction = pyscf_helper.get_active_space_effective_mo_integrals(
    #     one_body_mo=one_body_mo, two_body_mo=two_body_mo,
    #     freeze_indices_mo=[0, 1, 2, 3, 4, 5],
    #     active_indices_mo=[i for i in range(6, mol.nao_nr())])
    # ham_ferm_op_1, ham_ferm_op_2 = pyscf_helper.get_hamiltonian_ferm_op_from_mo_ints_mp(
    #     one_body_mo=one_body_mo_cas, two_body_mo=two_body_mo_cas, n_procs=n_procs)
    # ham_ferm_op = ham_ferm_op_1 + ham_ferm_op_2 + core_correction + mol.energy_nuc()
    ham_ferm_op_1, ham_ferm_op_2 = pyscf_helper.get_hamiltonian_ferm_op_from_mo_ints_mp(
        one_body_mo=one_body_mo, two_body_mo=two_body_mo, n_procs=n_procs)
    core_correction = 0.0
    ham_ferm_op = ham_ferm_op_1 + ham_ferm_op_2 + core_correction + mol.energy_nuc()

    # filename = "benzene_none_qubit_op.data"
    # openfermion.save_operator(ham_ferm_op, file_name=filename, data_directory=".")
    # print("Saved as openfermion format in %s" % (filename))

    ham_qubit_op = utils.jordan_wigner_mp(ham_ferm_op, n_procs=n_procs)

    filename = "qubit_op.data"
    openfermion.save_operator(ham_qubit_op, file_name=filename, data_directory=".")
    print("Saved as openfermion format in %s" % (filename))
