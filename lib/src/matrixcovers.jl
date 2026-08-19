# C ABI for MatrixCovers. Inputs and caller-owned output buffers contain
# `Float64`; matrices are column-major. Inputs are copied before calling
# MatrixCovers, and output sizes are checked before copying results back.
#
# # Penalty enum (`penalty::Int32`)
#
#   1 = AbsLog{1}()      2 = AbsLog{2}()      3 = AbsLinear{1}()      4 = AbsLinear{2}()
#
# Penalties are decoded with concrete branches to keep dispatch trim-safe.
# The `_min` entry points support only `AbsLog{2}` because other penalties
# require package extensions that are not linked. Soft covers and
# `cover_objective` support all four penalties.
#
# # linsolve enum (`linsolve::Int32`)
#
#   1 = :auto      2 = :dense      3 = :lsqr
#
# Used by `mc_symcover_min` and `mc_cover_min`.
#
# # Numeric tuning knobs
#
# Negative `maxiter`, `starts`, or `sigma` values omit that keyword, preserving
# MatrixCovers defaults. `starts`, `sigma`, and `seed` apply only to
# `AbsLinear` soft covers.
#
# # Status codes (`JLWStatus.code`)
#
#   0  ok
#   1  invalid penalty enum
#   2  penalty requires an extension not linked into this library
#   3  invalid linsolve enum
#   4  DimensionMismatch
#   6  ArgumentError
#   99 unexpected internal error
module matrixcovers

using JLWInterop
using MatrixCovers
using Random: MersenneTwister

# Results that contain both a status and a value.

struct MCBoolResult
    status::JLWStatus
    value::Int32
end

struct MCScalarResult
    status::JLWStatus
    value::Float64
end

# Boundary helpers

function _copyin_vec(v::CVector{Float64})
    out = Vector{Float64}(undef, length(v))
    copyto!(out, v)
    return out
end

function _copyin_mat(A::CMatrix{Float64})
    m, n = size(A)
    out = Matrix{Float64}(undef, m, n)
    copyto!(out, A)
    return out
end

# Explicit comparisons keep the mapping trim-safe.
function _linsolve_symbol(linsolve::Int32)
    linsolve == Int32(1) && return :auto
    linsolve == Int32(2) && return :dense
    linsolve == Int32(3) && return :lsqr
    return :invalid
end

# Inline exception handling so `trim=:safe` can narrow the exception type.
macro status_from_exception(e)
    quote
        let ex = $(esc(e))
            if ex isa DimensionMismatch
                m = ex.msg
                jlw_error(4, m isa String ? m : "dimension mismatch")
            elseif ex isa ArgumentError
                m = ex.msg
                jlw_error(6, m isa String ? m : "invalid argument")
            else
                jlw_error(99, "unexpected internal error")
            end
        end
    end
end

# soft_symcover / soft_cover kwarg plumbing
#
# Penalty types and explicit keyword combinations keep calls trim-safe.

function _soft_symcover_log(::Type{P}, M::Matrix{Float64}, maxiter::Int64) where {P<:MatrixCovers.AbstractCoverPenalty}
    return maxiter < 0 ? soft_symcover(P(), M) : soft_symcover(P(), M; maxiter = Int(maxiter))
end

function _soft_cover_log(::Type{P}, M::Matrix{Float64}, maxiter::Int64) where {P<:MatrixCovers.AbstractCoverPenalty}
    return maxiter < 0 ? soft_cover(P(), M) : soft_cover(P(), M; maxiter = Int(maxiter))
end

# `AbsLinear` solvers also accept `starts`, `sigma`, and `rng`.
function _soft_symcover_lin(::Type{P}, M::Matrix{Float64}, maxiter::Int64, starts::Int64,
                            sigma::Float64, seed::UInt64) where {P<:MatrixCovers.AbstractCoverPenalty}
    rng = MersenneTwister(seed)
    if maxiter < 0
        if starts < 0
            return sigma < 0 ? soft_symcover(P(), M; rng) :
                               soft_symcover(P(), M; sigma, rng)
        else
            return sigma < 0 ? soft_symcover(P(), M; starts = Int(starts), rng) :
                               soft_symcover(P(), M; starts = Int(starts), sigma, rng)
        end
    else
        if starts < 0
            return sigma < 0 ? soft_symcover(P(), M; maxiter = Int(maxiter), rng) :
                               soft_symcover(P(), M; maxiter = Int(maxiter), sigma, rng)
        else
            return sigma < 0 ? soft_symcover(P(), M; maxiter = Int(maxiter), starts = Int(starts), rng) :
                               soft_symcover(P(), M; maxiter = Int(maxiter), starts = Int(starts), sigma, rng)
        end
    end
end

function _soft_cover_lin(::Type{P}, M::Matrix{Float64}, maxiter::Int64, starts::Int64,
                         sigma::Float64, seed::UInt64) where {P<:MatrixCovers.AbstractCoverPenalty}
    rng = MersenneTwister(seed)
    if maxiter < 0
        if starts < 0
            return sigma < 0 ? soft_cover(P(), M; rng) :
                               soft_cover(P(), M; sigma, rng)
        else
            return sigma < 0 ? soft_cover(P(), M; starts = Int(starts), rng) :
                               soft_cover(P(), M; starts = Int(starts), sigma, rng)
        end
    else
        if starts < 0
            return sigma < 0 ? soft_cover(P(), M; maxiter = Int(maxiter), rng) :
                               soft_cover(P(), M; maxiter = Int(maxiter), sigma, rng)
        else
            return sigma < 0 ? soft_cover(P(), M; maxiter = Int(maxiter), starts = Int(starts), rng) :
                               soft_cover(P(), M; maxiter = Int(maxiter), starts = Int(starts), sigma, rng)
        end
    end
end

# Hard covers

Base.@ccallable function mc_symcover(A::CMatrix{Float64}, maxiter::Int64,
                                     a::CVector{Float64})::JLWStatus
    try
        M = _copyin_mat(A)
        av = maxiter < 0 ? symcover(M) : symcover(M; maxiter = Int(maxiter))
        length(a) == length(av) || return jlw_error(4, "output length must match matrix size")
        copyto!(a, av)
        return jlw_ok()
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_cover(A::CMatrix{Float64}, maxiter::Int64,
                                  a::CVector{Float64}, b::CVector{Float64})::JLWStatus
    try
        M = _copyin_mat(A)
        av, bv = maxiter < 0 ? cover(M) : cover(M; maxiter = Int(maxiter))
        (length(a) == length(av) && length(b) == length(bv)) ||
            return jlw_error(4, "output lengths must match matrix size")
        copyto!(a, av)
        copyto!(b, bv)
        return jlw_ok()
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_symcover_min(penalty::Int32, A::CMatrix{Float64}, maxiter::Int64,
                                         linsolve::Int32, a::CVector{Float64})::JLWStatus
    try
        if penalty == Int32(2)
            ls = _linsolve_symbol(linsolve)
            ls === :invalid && return jlw_error(3, "linsolve must be 1 (:auto), 2 (:dense), or 3 (:lsqr)")
            M = _copyin_mat(A)
            av = maxiter < 0 ? symcover_min(AbsLog{2}(), M; linsolve = ls) :
                                symcover_min(AbsLog{2}(), M; maxiter = Int(maxiter), linsolve = ls)
            length(a) == length(av) || return jlw_error(4, "output length must match matrix size")
            copyto!(a, av)
            return jlw_ok()
        elseif penalty == Int32(1)
            return jlw_error(2, "penalty AbsLog{1} requires the MatrixCoversJuMPExt extension (JuMP and HiGHS)")
        elseif penalty == Int32(3) || penalty == Int32(4)
            return jlw_error(2, "penalty AbsLinear requires the MatrixCoversIpoptExt extension (JuMP and Ipopt)")
        else
            return jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)")
        end
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_cover_min(penalty::Int32, A::CMatrix{Float64}, maxiter::Int64,
                                      linsolve::Int32, a::CVector{Float64},
                                      b::CVector{Float64})::JLWStatus
    try
        if penalty == Int32(2)
            ls = _linsolve_symbol(linsolve)
            ls === :invalid && return jlw_error(3, "linsolve must be 1 (:auto), 2 (:dense), or 3 (:lsqr)")
            M = _copyin_mat(A)
            av, bv = maxiter < 0 ? cover_min(AbsLog{2}(), M; linsolve = ls) :
                                    cover_min(AbsLog{2}(), M; maxiter = Int(maxiter), linsolve = ls)
            (length(a) == length(av) && length(b) == length(bv)) ||
                return jlw_error(4, "output lengths must match matrix size")
            copyto!(a, av)
            copyto!(b, bv)
            return jlw_ok()
        elseif penalty == Int32(1)
            return jlw_error(2, "penalty AbsLog{1} requires the MatrixCoversJuMPExt extension (JuMP and HiGHS)")
        elseif penalty == Int32(3) || penalty == Int32(4)
            return jlw_error(2, "penalty AbsLinear requires the MatrixCoversIpoptExt extension (JuMP and Ipopt)")
        else
            return jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)")
        end
    catch e
        return @status_from_exception(e)
    end
end

# Soft covers

Base.@ccallable function mc_soft_symcover(penalty::Int32, A::CMatrix{Float64}, maxiter::Int64,
                                          starts::Int64, sigma::Float64, seed::UInt64,
                                          a::CVector{Float64})::JLWStatus
    try
        M = _copyin_mat(A)
        av = if penalty == Int32(1)
            _soft_symcover_log(AbsLog{1}, M, maxiter)
        elseif penalty == Int32(2)
            _soft_symcover_log(AbsLog{2}, M, maxiter)
        elseif penalty == Int32(3)
            _soft_symcover_lin(AbsLinear{1}, M, maxiter, starts, sigma, seed)
        elseif penalty == Int32(4)
            _soft_symcover_lin(AbsLinear{2}, M, maxiter, starts, sigma, seed)
        else
            return jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)")
        end
        length(a) == length(av) || return jlw_error(4, "output length must match matrix size")
        copyto!(a, av)
        return jlw_ok()
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_soft_cover(penalty::Int32, A::CMatrix{Float64}, maxiter::Int64,
                                       starts::Int64, sigma::Float64, seed::UInt64,
                                       a::CVector{Float64}, b::CVector{Float64})::JLWStatus
    try
        M = _copyin_mat(A)
        av, bv = if penalty == Int32(1)
            _soft_cover_log(AbsLog{1}, M, maxiter)
        elseif penalty == Int32(2)
            _soft_cover_log(AbsLog{2}, M, maxiter)
        elseif penalty == Int32(3)
            _soft_cover_lin(AbsLinear{1}, M, maxiter, starts, sigma, seed)
        elseif penalty == Int32(4)
            _soft_cover_lin(AbsLinear{2}, M, maxiter, starts, sigma, seed)
        else
            return jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)")
        end
        (length(a) == length(av) && length(b) == length(bv)) ||
            return jlw_error(4, "output lengths must match matrix size")
        copyto!(a, av)
        copyto!(b, bv)
        return jlw_ok()
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_soft_symcover_min(penalty::Int32, A::CMatrix{Float64}, maxiter::Int64,
                                              a::CVector{Float64})::JLWStatus
    try
        if penalty == Int32(2)
            M = _copyin_mat(A)
            av = maxiter < 0 ? soft_symcover_min(AbsLog{2}(), M) :
                                soft_symcover_min(AbsLog{2}(), M; maxiter = Int(maxiter))
            length(a) == length(av) || return jlw_error(4, "output length must match matrix size")
            copyto!(a, av)
            return jlw_ok()
        elseif penalty == Int32(1)
            return jlw_error(2, "MatrixCovers does not implement AbsLog{1} for soft_symcover_min")
        elseif penalty == Int32(3) || penalty == Int32(4)
            return jlw_error(2, "penalty AbsLinear requires the MatrixCoversIpoptExt extension (JuMP and Ipopt)")
        else
            return jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)")
        end
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_soft_cover_min(penalty::Int32, A::CMatrix{Float64}, maxiter::Int64,
                                           a::CVector{Float64}, b::CVector{Float64})::JLWStatus
    try
        if penalty == Int32(2)
            M = _copyin_mat(A)
            av, bv = maxiter < 0 ? soft_cover_min(AbsLog{2}(), M) :
                                    soft_cover_min(AbsLog{2}(), M; maxiter = Int(maxiter))
            (length(a) == length(av) && length(b) == length(bv)) ||
                return jlw_error(4, "output lengths must match matrix size")
            copyto!(a, av)
            copyto!(b, bv)
            return jlw_ok()
        elseif penalty == Int32(1)
            return jlw_error(2, "MatrixCovers does not implement AbsLog{1} for soft_cover_min")
        elseif penalty == Int32(3) || penalty == Int32(4)
            return jlw_error(2, "penalty AbsLinear requires the MatrixCoversIpoptExt extension (JuMP and Ipopt)")
        else
            return jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)")
        end
    catch e
        return @status_from_exception(e)
    end
end

# Predicates and objectives

Base.@ccallable function mc_iscover_sym(a::CVector{Float64}, A::CMatrix{Float64},
                                        rtol::Float64, atol::Float64)::MCBoolResult
    try
        av = _copyin_vec(a)
        M = _copyin_mat(A)
        ok = iscover(av, M; rtol, atol)
        return MCBoolResult(jlw_ok(), ok ? Int32(1) : Int32(0))
    catch e
        return MCBoolResult(@status_from_exception(e), Int32(0))
    end
end

Base.@ccallable function mc_iscover(a::CVector{Float64}, b::CVector{Float64},
                                    A::CMatrix{Float64}, rtol::Float64,
                                    atol::Float64)::MCBoolResult
    try
        av = _copyin_vec(a)
        bv = _copyin_vec(b)
        M = _copyin_mat(A)
        ok = iscover(av, bv, M; rtol, atol)
        return MCBoolResult(jlw_ok(), ok ? Int32(1) : Int32(0))
    catch e
        return MCBoolResult(@status_from_exception(e), Int32(0))
    end
end

Base.@ccallable function mc_cover_objective_sym(penalty::Int32, a::CVector{Float64},
                                                A::CMatrix{Float64})::MCScalarResult
    try
        av = _copyin_vec(a)
        M = _copyin_mat(A)
        if penalty == Int32(1)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLog{1}(), av, M))
        elseif penalty == Int32(2)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLog{2}(), av, M))
        elseif penalty == Int32(3)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLinear{1}(), av, M))
        elseif penalty == Int32(4)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLinear{2}(), av, M))
        else
            return MCScalarResult(jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)"), NaN)
        end
    catch e
        return MCScalarResult(@status_from_exception(e), NaN)
    end
end

Base.@ccallable function mc_cover_objective(penalty::Int32, a::CVector{Float64},
                                            b::CVector{Float64},
                                            A::CMatrix{Float64})::MCScalarResult
    try
        av = _copyin_vec(a)
        bv = _copyin_vec(b)
        M = _copyin_mat(A)
        if penalty == Int32(1)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLog{1}(), av, bv, M))
        elseif penalty == Int32(2)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLog{2}(), av, bv, M))
        elseif penalty == Int32(3)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLinear{1}(), av, bv, M))
        elseif penalty == Int32(4)
            return MCScalarResult(jlw_ok(), cover_objective(AbsLinear{2}(), av, bv, M))
        else
            return MCScalarResult(jlw_error(1, "unknown penalty enum, expected 1 (AbsLog1), 2 (AbsLog2), 3 (AbsLinear1), or 4 (AbsLinear2)"), NaN)
        end
    catch e
        return MCScalarResult(@status_from_exception(e), NaN)
    end
end

# Gram covers

Base.@ccallable function mc_gramcover(a::CVector{Float64}, b::CVector{Float64},
                                      A::CMatrix{Float64}, s::CVector{Float64})::JLWStatus
    try
        av = _copyin_vec(a)
        bv = _copyin_vec(b)
        M = _copyin_mat(A)
        sv = Vector{Float64}(undef, size(M, 2))
        gramcover!(sv, av, bv, M)
        length(s) == length(sv) || return jlw_error(4, "output length must match the number of columns of A")
        copyto!(s, sv)
        return jlw_ok()
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_gramcover_weighted(a::CVector{Float64}, b::CVector{Float64},
                                               A::CMatrix{Float64}, w::CVector{Float64},
                                               s::CVector{Float64})::JLWStatus
    try
        av = _copyin_vec(a)
        bv = _copyin_vec(b)
        M = _copyin_mat(A)
        wv = _copyin_vec(w)
        sv = Vector{Float64}(undef, size(M, 2))
        gramcover!(sv, av, bv, M, wv)
        length(s) == length(sv) || return jlw_error(4, "output length must match the number of columns of A")
        copyto!(s, sv)
        return jlw_ok()
    catch e
        return @status_from_exception(e)
    end
end

Base.@ccallable function mc_gramcover_matrix(a::CVector{Float64}, b::CVector{Float64},
                                             A::CMatrix{Float64}, W::CMatrix{Float64},
                                             s::CVector{Float64})::JLWStatus
    try
        av = _copyin_vec(a)
        bv = _copyin_vec(b)
        M = _copyin_mat(A)
        Wm = _copyin_mat(W)
        sv = Vector{Float64}(undef, size(M, 2))
        gramcover!(sv, av, bv, M, Wm)
        length(s) == length(sv) || return jlw_error(4, "output length must match the number of columns of A")
        copyto!(s, sv)
        return jlw_ok()
    catch e
        return @status_from_exception(e)
    end
end

end # module
