
const datatype = Int8
const coeff_dtype = Float64

const LIBNAME = "libeloc"

function set_indices_ham_int_opt(n_qubits, K, NK, idxs, coeffs, pauli_mat12, pauli_mat23)

    ccall((:set_indices_ham_int_opt, LIBNAME),
        Cvoid,

        (Cint, Clonglong, Clonglong, Ptr{Clonglong}, Ptr{Cdouble},
        Ptr{datatype}, Ptr{datatype}),

        n_qubits, K, NK, idxs, coeffs,
        pauli_mat12, pauli_mat23
    )

end

# Serial of sampling
function calculate_local_energy(_states, ist::Int, ied::Int,
    k_idxs::Vector, ks::Union{Vector, Matrix}, vs::Matrix, rank::Int=0, eps::Float64=1e-14)

    batch_size = size(_states, 2)
    _states = reshape(_states, :)
    vs = reshape(vs, :)
    # current rank batch_size
    batch_size_cur_rank = ied - ist + 1

    psi_type = typeof(vs[1])
    #println("psi_type julia: $psi_type")
    if psi_type == Float32
        # store result of E_loc (complex type)
        res_eloc_batch_h = zeros(Float32, 2*batch_size_cur_rank)
        ccall(
            # (func_name, libname)
            (:calculate_local_energy, LIBNAME),
            # return type
            Cvoid,
            # arguments type
            (Clonglong,
             Ptr{Clonglong}, Clonglong, Clonglong,
             Ptr{Clonglong}, Ptr{Clonglong}, Ptr{Cfloat},
             Clonglong, Cdouble, Ptr{Cfloat}),
            # real arguments
            batch_size,
            _states, ist-1, ied,
            k_idxs, ks, vs,
            rank, eps, res_eloc_batch_h
        )
    else
        res_eloc_batch_h = zeros(Float64, 2*batch_size_cur_rank)
        ccall(
            # (func_name, libname)
            (:calculate_local_energy, LIBNAME),
            # return type
            Cvoid,
            # arguments type
            (Clonglong,
             Ptr{Clonglong}, Clonglong, Clonglong,
             Ptr{Clonglong}, Ptr{Clonglong}, Ptr{Cfloat},
             Clonglong, Cdouble, Ptr{Cdouble}),
            # real arguments
            batch_size,
            _states, ist-1, ied,
            k_idxs, ks, vs,
            rank, eps, res_eloc_batch_h
        )
    end

    # convert into complex type
    res_eloc_batch_h = reshape(res_eloc_batch_h, 2, batch_size_cur_rank)
    res_eloc_cplx = res_eloc_batch_h[1, :] + res_eloc_batch_h[2, :] * 1im
    return res_eloc_cplx
end

# Parallel of sampling
function calculate_local_energy(_states, ist::Int, ied::Int,
    ks_disp_idx::Int, all_batch_size::Int, ks::Vector, vs::Vector, rank::Int=0, eps::Float64=1e-14)

    batch_size = size(_states, 2)
    _states = reshape(_states, :)
    # current rank batch_size
    batch_size_cur_rank = ied - ist + 1

    # store result of E_loc (complex type)
    res_eloc_batch_h = zeros(Float32, 2*batch_size_cur_rank)
    ccall(
        # (func_name, libname)
        (:calculate_local_energy_sampling_parallel, LIBNAME),
        # return type
        Cvoid,
        # arguments type
        (Clonglong, Clonglong,
         Ptr{Clonglong}, Clonglong, Clonglong,
         Clonglong, Ptr{Clonglong}, Ptr{Cfloat},
         Clonglong, Cdouble, Ptr{Cfloat}),
        # real arguments
        all_batch_size, batch_size,
        _states, ist-1, ied,
        ks_disp_idx, ks, vs,
        rank, eps, res_eloc_batch_h
    )

    # convert into complex type
    res_eloc_batch_h = reshape(res_eloc_batch_h, 2, batch_size_cur_rank)
    res_eloc_cplx = res_eloc_batch_h[1, :] + res_eloc_batch_h[2, :] * 1im
    return res_eloc_cplx
end

# Parallel of sampling of BigInt
function calculate_local_energy(_states, ist::Int, ied::Int,
    ks_disp_idx::Int, all_batch_size::Int, ks::Vector{UInt64}, id_width::Int, vs::Vector, rank::Int=0, eps::Float64=1e-14)

    batch_size = size(_states, 2)
    _states = reshape(_states, :)
    # current rank batch_size
    batch_size_cur_rank = ied - ist + 1
    # log("JL batch_size: $batch_size all_batch_size: $(all_batch_size) id_width: $id_width ist: $ist ied: $ied _states: $(size(_states))")
    # store result of E_loc (complex type)
    psi_type = typeof(vs[1])
    # log("psi_type julia: $psi_type")
    if psi_type == Float32
        # Float32
        res_eloc_batch_h = zeros(Float32, 2*batch_size_cur_rank)
        ccall(
            # (func_name, libname)
            (:calculate_local_energy_sampling_parallel_bigInt, LIBNAME),
            # return type
            Cvoid,
            # arguments type
            (Clonglong, Clonglong,
            Ptr{Clonglong}, Clonglong, Clonglong,
            Clonglong, Ptr{Culonglong}, Clonglong, Ptr{Cfloat},
            Clonglong, Cdouble, Ptr{Cfloat}),
            # real arguments
            all_batch_size, batch_size,
            _states, ist-1, ied,
            ks_disp_idx, ks, id_width, vs,
            rank, eps, res_eloc_batch_h
        )
    else
        # Float64
        res_eloc_batch_h = zeros(Float64, 2*batch_size_cur_rank)
        ccall(
            # (func_name, libname)
            (:calculate_local_energy_sampling_parallel_bigInt, LIBNAME),
            # return type
            Cvoid,
            # arguments type
            (Clonglong, Clonglong,
            Ptr{Clonglong}, Clonglong, Clonglong,
            Clonglong, Ptr{Culonglong}, Clonglong, Ptr{Cdouble},
            Clonglong, Cdouble, Ptr{Cdouble}),
            # real arguments
            all_batch_size, batch_size,
            _states, ist-1, ied,
            ks_disp_idx, ks, id_width, vs,
            rank, eps, res_eloc_batch_h
        )
    end

    # convert into complex type
    res_eloc_batch_h = reshape(res_eloc_batch_h, 2, batch_size_cur_rank)
    res_eloc_cplx = res_eloc_batch_h[1, :] + res_eloc_batch_h[2, :] * 1im
    return res_eloc_cplx
end

function init_hamiltonian_ccall(ham_file::String)
    n_qubits = ccall(
        (:init_hamiltonian, LIBNAME),
        Cint,
        (Cstring,),
        ham_file
    )
    return n_qubits
end

function free_hamiltonian_ccall()
    ccall(
        (:free_hamiltonian, LIBNAME),
        Cvoid,
        ()
    )
end
