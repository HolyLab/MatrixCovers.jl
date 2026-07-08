module SIASparseArrays

using LinearAlgebra
using SparseArrays
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLog, AbsLinear, _abslinear2_iter!, _abslinear1_iter!, _symcover_min_abslog2, _cover_min_abslog2

# ============================================================
# Support traversal
# ============================================================

function ScaleInvariantAnalysis.foreach_support(f, A::SparseMatrixCSC)
    rv, nzs = rowvals(A), nonzeros(A)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            v = abs(nzs[k])
            iszero(v) || f(rv[k], j, v)
        end
    end
    return nothing
end

function ScaleInvariantAnalysis.foreach_support_sym(f, A::SparseMatrixCSC)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(DimensionMismatch("foreach_support_sym requires a square matrix, got axes $(axes(A))"))
    rv, nzs = rowvals(A), nonzeros(A)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            i = rv[k]
            i <= j || continue
            v = abs(nzs[k])
            iszero(v) || f(i, j, v)
        end
    end
    return nothing
end

# Emitted pairs are canonical (row <= col) regardless of uplo: for uplo='L'
# the stored (i, j) with i >= j is reported as (j, i).
function ScaleInvariantAnalysis.foreach_support_sym(f,
        S::Union{Symmetric{<:Any,<:SparseMatrixCSC},Hermitian{<:Real,<:SparseMatrixCSC}})
    P = parent(S)
    ax = axes(P, 1)
    axes(P, 2) == ax || throw(DimensionMismatch("foreach_support_sym requires a square matrix, got axes $(axes(P))"))
    rv, nzs = rowvals(P), nonzeros(P)
    uplo = S.uplo
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            if uplo == 'U'
                i > j && continue
                v = abs(nzs[k])
                iszero(v) || f(i, j, v)
            else
                i < j && continue
                v = abs(nzs[k])
                iszero(v) || f(j, i, v)
            end
        end
    end
    return nothing
end

# ============================================================
# Native minimal-cover (MCM) solvers
# ============================================================

# Native AbsLog{2} MCM solvers on sparse supports default to the matrix-free LSQR
# inner solve, whose per-iteration cost is O(nnz) and whose accuracy tracks the
# conditioning of √W·R (≈ √κ) rather than that of the normal equations (≈ κ). This
# is the intended path when nnz ≪ n²; pass `linsolve=:auto`/`:dense` to force the
# dense factorization. Only AbsLog{2} is native; other penalties dispatch to the
# JuMP extension.
# The worker allocates its scale vectors with `similar(A, ...)`, which is a
# `SparseVector` for a sparse `A`; the scales are dense objects, so return plain
# `Vector`s, matching `cover`/`symcover` on the same input.
function ScaleInvariantAnalysis.symcover_min(ϕ::AbsLog{2}, A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(A; linsolve, kwargs...)
    return Vector(a)
end

function ScaleInvariantAnalysis.cover_min(ϕ::AbsLog{2}, A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    a, b, _ = _cover_min_abslog2(A; linsolve, kwargs...)
    return Vector(a), Vector(b)
end

function ScaleInvariantAnalysis.symcover_min(ϕ::AbsLog{2}, S::Symmetric{<:Any, <:SparseMatrixCSC}; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(S; linsolve, kwargs...)
    return Vector(a)
end

function ScaleInvariantAnalysis.symcover_min(ϕ::AbsLog{2}, H::Hermitian{<:Real, <:SparseMatrixCSC}; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(H; linsolve, kwargs...)
    return Vector(a)
end

end  # module SIASparseArrays
