using NPZ
#using Random, LinearAlgebra, DelimitedFiles

# push!(LOAD_PATH, ".")
include("../interface/julia/eloc.jl")

#function strToIntList(arrStr::String)
#    [parse(Int, c) for c in arrStr]
#end

# function test_rand_single(n_orb::Int=4, seed::Int=123)
#     eloc_ref, op_name, fci_name = check_eloc.gen_random_ham(n_orb=n_orb, eps=1e-12, seed=seed, is_save_ham=true)
#     println("eloc_ref: $eloc_ref op_name: $op_name fci_name: $fci_name")
#     eloc.init_hamiltonian(op_name)
#     ci_data = load(fci_name)
#     ci_probs, ci_states = ci_data["ci_probs"], ci_data["ci_states"]
#     ci_probs = complex.(ci_probs) # Convert to complex if needed

#     counts = real(ci_probs) .^ 2
#     eloc_val = eloc.energy(ci_states, ci_probs, counts, true)
#     eloc.free_hamiltonian()
#     err = abs(eloc_val - eloc_ref)
#     println("err: $err")
#     @assert err < 1e-8 "err: $err seed: $seed n_orb: $n_orb"
# end

# function test_rand()
#     n_qubits_l, n_qubits_r = 1, 8
#     n_rand = 10
#     rands = abs.(round.(Int, randn(n_rand) * 100))
#     for i in n_qubits_l:n_qubits_r
#         for seed in rands
#             test_rand_single(n_orb=i+1, seed=seed)
#         end
#     end
# end

function test_mol(ham_path::String, np_data::String; psis_dtype::String="ComplexF32", is_need_sort::Bool=false, is_trans::Bool=false, err_eps::Float64=1e-4)
    """
    ham_path: hamiltonian file path (qubit op format)
    np_data: states and coeffs data path
    psi_dtype: ComplexF32 or ComplexF64
    is_need_sort: cpu version must be true, gpu version can be false
    is_trans: read from numpy.savez data should be true
    err_eps: check precision epsilon
    """
    n_qubits = init_hamiltonian(ham_path)
    ci_data = npzread(np_data)
    ci_probs, ci_states, e_fci = ci_data["ci_probs"], ci_data["ci_states"], ci_data["e_fci"]
    # ci_probs, ci_states= ci_data["ci_probs"], ci_data["ci_states"]
    is_trans && (ci_states = reshape(transpose(ci_states), (n_qubits, size(ci_probs, 1))))
    ci_probs = (psis_dtype == "ComplexF32") ? ComplexF32.(ci_probs) : ComplexF64.(ci_probs)

    # println("[test] ci_porbs: $(size(ci_probs)) ci_states: $(size(ci_states))")
    counts = real(ci_probs) .^ 2
    eloc_expectation = energy(ci_states, ci_probs, counts, is_need_sort=is_need_sort)
    err = abs(e_fci - real(eloc_expectation))
    @assert err < err_eps "[test] CHECK FAIL == e_fci: $(e_fci) eloc_expectation: $(real(eloc_expectation)) err: $(err)"
    println("[test] PASS == n_qubits: $(n_qubits) eloc_expectation: $(real(eloc_expectation)) ABS(err): $(err)")
    free_hamiltonian()
end

if abspath(PROGRAM_FILE) == @__FILE__
    # generate by `python test/utils/check_eloc.py mol_test H2`
    np_data = "H2-fci.npz"
    ham_path = "H2-qubit_op.data"
    # generate by `python test/utils/check_eloc.py mol_test LiH`
    np_data = "LiH-fci.npz"
    ham_path = "LiH-qubit_op.data"
    # generate by `python test/utils/check_eloc.py rand_test`
    ham_path = "qubits8-seed111-qubit_op.data"
    np_data = "qubits8-seed111-fci.npz"

    psis_dtype = "ComplexF32" # or ComplexF64 (make cpu_fp64 or make gpu_fp64)
    is_need_sort = true # CPU version must be true
    is_trans = true # read from numpy.savez data format
    test_mol(ham_path, np_data, psis_dtype=psis_dtype, is_need_sort=is_need_sort, is_trans=is_trans)
end
