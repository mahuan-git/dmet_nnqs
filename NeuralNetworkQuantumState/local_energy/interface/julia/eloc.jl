using LinearAlgebra
# push!(LOAD_PATH, "interface/julia/")
include("calculate_local_energy_wrapper.jl")

# Constants
const __STRIDE = 64
const __EPS = 1e-12

__POWER2_LOOKUP = UInt64(2) .^ (0:63)

function __state2id_huge_batch(state::Array{Int, 2})
    id_width = ceil(Int, size(state, 1) / __STRIDE)
    ret_id = zeros(UInt64, id_width, size(state, 2))
    for i in 1:id_width
        _state_mask = clamp.(state[(i-1)*__STRIDE+1:min(i*__STRIDE, end), :], 0, 1)
        L = size(_state_mask, 1)
        ret_id[i, :] = reshape(__POWER2_LOOKUP[1:L], (1,:)) * _state_mask 
    end
    # println("stats:")
    # display(state)
    # println("ret_id: $(ret_id)")
    return ret_id
end

function calculate_local_energy(states, psis; is_need_sort=false)
    """
    Args:
        states: (n_qubits, n_samples)
        psis: (n_samples) complex32/complex64
        is_need_sort: sort samples in order for binary find (only need by CPU version)
    Return:
        local_energies: (n_samples) complex32/complex64
    """
    # Ensure the input is Array
    states = Array(states)
    psis = Array(psis)

    @assert size(states, 2) == length(psis)

    ptype_real = typeof(real(psis[1]))
    num_unique = length(psis)

    ks = __state2id_huge_batch(states)
    vs = vcat(real(psis)', imag(psis)') |> x -> convert(Array{ptype_real, 2}, x)
    # println("ks: $(Int64.(ks)) $(size(ks))")
    # display(vs)
    
    if is_need_sort
        idxs = reshape(sortperm(ks, dims=2), :)
        ks = ks[:, idxs]
        vs = vs[:, idxs]
        states = states[:, idxs]
        psis = psis[idxs]
    end

    #println("ptype_real: $(ptype_real) num_unique: $(num_unique) states: $(size(states))")
    #ist, ied = 0, num_unique
    ist, ied = 1, num_unique
    k_idxs = zeros(Int64, 1)

    #println("ks: $(size(ks)) vs: $(size(vs)) states: $(size(states))")
    local_energies = calculate_local_energy(states, ist, ied, k_idxs, ks, vs, 0, __EPS)

    if is_need_sort
        idxs_rev = sortperm(idxs)
        local_energies = local_energies[idxs_rev]
    end

    return local_energies
end

function energy(states, psis, weights; is_need_sort=false)
    @assert size(states, 2) == length(psis) == length(weights)
    
    # Ensure the input is Array
    states = Array(states)
    psis = Array(psis)
    weights = Array(weights)

    local_energies = calculate_local_energy(states, psis, is_need_sort=is_need_sort)
    weights /= sum(weights) # Normalize weights
    # println("local_energies: $(local_energies)")
    # println("weights: $(weights)")
    eloc_expectation = dot(local_energies, weights)
    return eloc_expectation
end

function init_hamiltonian(ham_file::String)
    n_qubits = init_hamiltonian_ccall(ham_file)
    return n_qubits
end

function free_hamiltonian()
    free_hamiltonian_ccall()
end

