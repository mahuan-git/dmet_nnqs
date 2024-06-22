import sys
import scipy
import openfermion
from openfermion import geometry_from_pubchem

# QCQC: https://gitlab.com/auroraustc/qcqc.git
sys.path.append("../")
import pyscf_helper
import utils

if len(sys.argv) > 1:
    dist = float(sys.argv[1])
    print("dist=", dist)

# Unit for length: Angstrom
# Unit for energy: Hartree

MOLECULE_LIST = ["H2", "F2", "HCl", "LiH", "H2O", "CH2", "O2", "BeH2","H2S",
                 "NH3", "N2", "CH4", "C2", "LiF", "PH3", "LiCL", "Li2O", "C2H4O"]
def get_geometry(molecule_name, verbose=True):
    if verbose and (molecule_name not in MOLECULE_LIST):
        print(f"Warning: {molecule_name} is not one of the molecules used in the paper" + 
               "- that's not wrong, but just know it's not recreating the published results!")
    
    if molecule_name=="C2":
        # C2 isn't in PubChem - don't know why.
        geometry = [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])]
    else:
        if molecule_name=="Li2O":
            # Li2O returns a different molecule - again, don't know why.
            molecule_name = "Lithium Oxide"
        geometry = geometry_from_pubchem(molecule_name)
        
    return geometry

# mol_name = "H2"
# mol_name = "Sulfuric Acid"
# mol_name = "Sodium Carbonate"
# geometry = get_geometry(mol_name)
# print(f"get from pubchem database: {geometry}")
# geometry = [('Be', (1, 0, 0)), ('Be', (1.2, 0, 0))] # Be2
# geometry = [('He', (1, 0, 0)), ('He', (1.2, 0, 0))] # He2
# geometry = [('He', (-1.023, 0, 0)), ('He', (1.2, 0, 0))] # He2
# geometry = [('He', (-dist/2+0.05, 0, 0)), ('He', (dist/2+0.05, 0, 0))] # He2
# geometry = [('Ne', (-dist/2+0.05, 0, 0)), ('Ne', (dist/2+0.05, 0, 0))] # Ne2
# geometry = [('Ne', (-1.023, 0, 0)), ('Ne', (1.2, 0, 0))] # Ne2
# geometry = [('H', (0, 0, 0)), ('H', (0, 0, 0.734))] # H2
# geometry = [('Li', (0, 0, 0)), ('H', (0, 0, 1.548))] # LiH

# geometry = [('H', (2, 0, 0)), ('H', (3, 0, 0))] # H2
geometry = [('H', (0, 0, 0)), ('H', (dist, 0, 0))] # H2
# H-H: 0.7414446816 (H-H)-(H-H): 1.32295
# geometry = [('H', (0, 0, 0)), ('H', (0.7414446816, 0, 0)), ('H', (2.0643946816, 0, 0)), ('H', (2.8058393632, 0, 0))] # H4
# geometry = [('H', (-5.6770499999999995, 0, 0)), ('H', (-4.9356053183999995, 0, 0)), ('H', (-3.6126553183999994, 0, 0)), ('H', (-2.8712106367999994, 0, 0)), ('H', (-1.5482606367999994, 0, 0)), ('H', (-0.8068159551999994, 0, 0)), ('H', (0.5161340448000007, 0, 0)), ('H', (1.2575787264000007, 0, 0)), ('H', (2.5805287264000007, 0, 0)), ('H', (3.3219734080000007, 0, 0)), ('H', (4.644923408, 0, 0)), ('H', (5.3863680896, 0, 0))] # H12
# H-H: 2.5 A
# geometry = [('H', (-12.5, 0, 0)), ('H', (-10.0, 0, 0)), ('H', (-7.5, 0, 0)), ('H', (-5.0, 0, 0)), ('H', (-2.5, 0, 0)), ('H', (0.0, 0, 0)), ('H', (2.5, 0, 0)), ('H', (5.0, 0, 0)), ('H', (7.5, 0, 0)), ('H', (10.0, 0, 0)), ('H', (12.5, 0, 0)), ('H', (15.0, 0, 0))] # H12
# geometry = [('F', (2, 0, 0)), ('F', (3, 0, 0))] # F2
# geometry = [('Cl', (0, 0, 0)), ('H', (0.6058, -0.2053, -1.1384))] # HCl
# geometry = [('Li', (3, 0, 0)), ('H', (2, 0, 0))] # LiH
# geometry = [('O', (-1.197, 0, 0)), ('O', (1.197, 0, 0)), ('C', (0, 0, 0))] # CO2
# geometry = [('O', (0, 0, 0)), ('H', (0.2774, 0.8929, 0.2544)), ('H', (0.6068, -0.2383, -0.7169))] # H2O
# geometry = [('C', (2.5369, -0.155, 0)), ('H', (3.0739, 0.155, 0)), ('H', (2, 0.155, 0))] # CH2 S=1
# geometry = [('O', (-0.616, 0, 0)), ('O', (0.616, 0, 0))] # O2 S=1
# geometry = [('Be', (2.5369, 0.155, 0)), ('H', (2, -0.155, 0)), ('H', (3.0739, -0.155, 0))] # BeH2 
# geometry = [('S', (0, 0, 0)), ('H', (0.4855, 1.2232, 0.2576)), ('H', (0.8868, -0.2325, -0.9787))] # H2S
# geometry = [('N', (0, 0, 0)), ('H', (-0.4417, 0.2906, 0.8711)), ('H', (0.7256, 0.6896, -0.1907)), ('H', (0.4875, -0.8701, 0.2089))] # NH3
# geometry = [('N', (-0.556, 0, 0)), ('N', (0.556, 0, 0))] # N2
# geometry = [('N', (-dist/2, 0, 0)), ('N', (dist/2, 0, 0))] # N2 for PES
# geometry = [('C', (0, 0, 0)), ('H', (0.5541, 0.7996, 0.4965)), ('H', (0.6833, -0.8134, -0.2536)), ('H', (-0.7782, -0.3735, 0.6692)), ('H', (-0.4593, 0.3874, -0.9121))] # CH4
# geometry = [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])] # C2
# geometry = [('F', (3, 0, 0)), ('Li', (2, 0, 0))] # LiF
# geometry = [('P', (0, 0, 0)), ('H', (-0.6323, 0.513, 1.1573)), ('H', (1.2032, 0.7159, 0.2052)), ('H', (0.461, -1.1757, 0.6383))] # PH3
# geometry = [('Cl', (3, 0, 0)), ('Li', (2, 0, 0))] # LiCL
# geometry = [('O', (2.866, -0.25, 0)), ('Li', (3.732, 0.25, 0)), ('Li', (2, 0.25, 0))] # Li2O
# geometry = [('O', (-0.0007, 0.8141, 0)), ('C', (0.7509, -0.4065, 0)), ('C', (-0.7502, -0.4076, 0)), ('H', (1.2625, -0.6786, 0.9136)), ('H', (1.2625, -0.6787, -0.9136)), ('H', (-1.2614, -0.6806, -0.9136)), ('H', (-1.2614, -0.6805, 0.9136))] # Ethylene Oxide, C2H4O
# geometry = [('C', (1.2818, -0.2031, 0)), ('C', (-0.0643, 0.4402, 0)), ('C', (-1.2175, -0.2371, 0)), ('H', (1.8429, 0.1063, -0.8871)), ('H', (1.2188, -1.2959, 0)), ('H', (1.8429, 0.1063, 0.8871)), ('H', (-0.095, 1.5262, 0)), ('H', (-2.1647, 0.2911, 0)), ('H', (-1.239, -1.3212, 0))] # Propene, C3H6
# geometry = [('O', (-0.3035, 1.289, -0.0002)), ('O', (-0.98, -0.8878, -0.0002)), ('C', (1.3743, -0.3516, -0.0002)), ('C', (-0.0907, -0.0496, 0.0006)), ('H', (1.8368, 0.057, -0.9021)), ('H', (1.84, 0.0676, 0.8952)), ('H', (1.5207, -1.4356, 0.0064)), ('H', (-1.2598, 1.5081, -0.0008))] # Acetic Acid, C2H4O2
# geometry = [('S', (0.0002, -0.0002, -0.1181)), ('O', (1.2744, 0.0355, 0.8975)), ('O', (-1.2765, -0.0326, 0.8949)), ('O', (-0.0336, 1.2577, -0.8391)), ('O', (0.0354, -1.2604, -0.8351)), ('H', (1.2111, 0.6912, 1.6312)), ('H', (-1.2148, -0.6859, 1.6307))] # Sulfuric Acid, H2O4S
# geometry = [('Na', (5.4641, 0.25, 0)), ('Na', (2, 0.25, 0)), ('O', (4.5981, 0.75, 0)), ('O', (2.866, 0.75, 0)), ('O', (3.732, -0.75, 0)), ('C', (3.732, 0.25, 0))] # CNa2O3
# run_fci = False
run_fci = True
# run_rccsd = False
run_rccsd = True
basis = '6-31g'
basis = 'sto-3g'
basis = "ccpvtz"
basis = "ccpvqz"
basis = "aug-cc-pvtz"
localized_orbitals = 'NAO'
localized_orbitals = None

freeze_occ_indices_spin, freeze_vir_indices_spin = None, None
# freeze_occ_indices_spin = [0, 1]
# freeze_vir_indices_spin = [6, 7]
print(f"geometry: {geometry}, basis: {basis} localized_orbitals: {localized_orbitals}")
print(f"freeze_occ_indices_spin: {freeze_occ_indices_spin}, freeze_vir_indices_spin: {freeze_vir_indices_spin}")
res = pyscf_helper.init_scf(
    geometry=geometry, run_fci=run_fci, run_rccsd=run_rccsd, basis=basis, 
    localized_orbitals=localized_orbitals, freeze_occ_indices_spin=freeze_occ_indices_spin, freeze_vir_indices_spin=freeze_vir_indices_spin)

# exit()

qubit_op = None
for i in res:
    if type(i) is openfermion.QubitOperator:
        qubit_op = i
        break

if freeze_occ_indices_spin is not None:
    e_freeze = scipy.sparse.linalg.eigsh(openfermion.get_sparse_operator(qubit_op), k=1, which="SA")[0]
    print(f"Freezed_FCI energy: {e_freeze[0]}")

# Not used.
mol = res[0]
n_qubits = openfermion.count_qubits(qubit_op)
n_electrons = mol.nelectron
particle_num_op = utils.particle_number_operator(n_qubits)
particle_num_op = openfermion.jordan_wigner(particle_num_op)
total_spin_op = utils.total_spin_operator(n_qubits, n_electrons)
total_spin_op = openfermion.jordan_wigner(total_spin_op)

print("Number of qubits: %d" % (openfermion.count_qubits(qubit_op)))
print("Number of electrons: %d" % (n_electrons))

filename = "qubit_op.data"
utils.save_binary_qubit_op(qubit_op, filename=filename)
print("Qubit Hamiltonian saved to %s." % (filename))
#utils.save_binary_qubit_op(particle_num_op, filename="qubit_op_n.data")
#utils.save_binary_qubit_op(total_spin_op, filename="qubit_op_s.data")
