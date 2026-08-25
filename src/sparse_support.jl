# Sparse-storage support traversal and the sparse defaults of the native solvers.

# ============================================================
# Support traversal
# ============================================================

function foreach_support(f, A::SparseMatrixCSC)
    rv, nzs = rowvals(A), nonzeros(A)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            v = abs(nzs[k])
            iszero(v) || f(rv[k], j, v)
        end
    end
    return nothing
end

function foreach_support_sym(f, A::SparseMatrixCSC)
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

# Upper bounds for sparse support traversals, used by `sizehint!`.
_support_sizehint(A::SparseMatrixCSC) = nnz(A)
_support_sizehint_sym(A::SparseMatrixCSC) = (nnz(A) + size(A, 1) + 1) >> 1
_support_sizehint_sym(S::Union{Symmetric{<:Any,<:SparseMatrixCSC},Hermitian{<:Any,<:SparseMatrixCSC}}) =
    nnz(parent(S))

# Match symmetric partners in O(nnz) with one cursor per column. `tr[c]` points
# to the first unpaired entry in column `c`; absent entries are zeros.
function require_abs_symmetric(A::SparseMatrixCSC, fname)
    ax = axes(A, 1)
    axes(A, 2) == ax ||
        throw(DimensionMismatch("$fname requires a square matrix, got axes $(string(axes(A)))"))
    rv, nzs = rowvals(A), nonzeros(A)
    tr = [first(nzrange(A, j)) for j in axes(A, 2)]
    for col in axes(A, 2)
        for p in tr[col]:last(nzrange(A, col))
            v = abs(nzs[p])
            iszero(v) && continue
            row = rv[p]
            row == col && continue
            row < col && _abs_asymmetry_error(fname, row, col, v, zero(v))
            off, stop = tr[row], last(nzrange(A, row)) + 1
            w = zero(v)
            while off < stop
                r2 = rv[off]
                r2 > col && break
                if r2 == col
                    w = abs(nzs[off])
                    tr[row] = off + 1
                    break
                end
                u = abs(nzs[off])
                iszero(u) || _abs_asymmetry_error(fname, r2, row, u, zero(u))
                off += 1
                tr[row] = off
            end
            _abs_symmetric(v, w) || _abs_asymmetry_error(fname, row, col, v, w)
        end
    end
    return nothing
end

# Emitted pairs are canonical (row <= col) regardless of uplo: for uplo='L'
# the stored (i, j) with i >= j is reported as (j, i). Complex `Hermitian` is
# admitted alongside the real case because only `abs` of a stored value is ever
# read, and `abs(A[i,j]) == abs(conj(A[j,i]))`.
function foreach_support_sym(f,
        S::Union{Symmetric{<:Any,<:SparseMatrixCSC},Hermitian{<:Any,<:SparseMatrixCSC}})
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

# The asymmetric traversal of a wrapped sparse matrix, which the asymmetric cover
# algorithms and `cover_objective` read even when the matrix is symmetric. Only the
# named triangle is stored, so each off-diagonal pair is emitted in both
# orientations and the diagonal once; the magnitudes agree in both, including for a
# complex `Hermitian`. Without this the wrappers fall back to the generic
# `AbstractMatrix` method and its full-grid `getindex` scan.
function foreach_support(f,
        S::Union{Symmetric{<:Any,<:SparseMatrixCSC},Hermitian{<:Any,<:SparseMatrixCSC}})
    foreach_support_sym(S) do i, j, v
        f(i, j, v)
        i == j || f(j, i, v)
    end
    return nothing
end

# ============================================================
# Native minimal-cover (MMC) solvers
# ============================================================

# Sparse `AbsLog{2}` solvers default to LSQR; `:auto` uses support density.
function symcover_min(ϕ::AbsLog{2}, A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(A; linsolve, kwargs...)
    return a
end

function cover_min(ϕ::AbsLog{2}, A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    a, b, _ = _cover_min_abslog2(A; linsolve, kwargs...)
    return a, b
end

function symcover_min(ϕ::AbsLog{2}, S::Symmetric{<:Any, <:SparseMatrixCSC}; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(S; linsolve, kwargs...)
    return a
end

function symcover_min(ϕ::AbsLog{2}, H::Hermitian{<:Any, <:SparseMatrixCSC}; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(H; linsolve, kwargs...)
    return a
end

# The refiners take the same sparse `linsolve` default as the solvers above.
function symcover_min!(ϕ::AbsLog{2}, a::AbstractVector,
        S::Union{SparseMatrixCSC,Symmetric{<:Any,<:SparseMatrixCSC},Hermitian{<:Any,<:SparseMatrixCSC}};
        linsolve::Symbol=:lsqr, kwargs...)
    _prepare_symcover_start!(a, S)
    anew, _ = _symcover_min_abslog2(S; start=a, linsolve, kwargs...)
    a .= anew
    return a
end

function cover_min!(ϕ::AbsLog{2}, a::AbstractVector, b::AbstractVector,
                    A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    _prepare_cover_start!(a, b, A)
    anew, bnew, _ = _cover_min_abslog2(A; start=(a, b), linsolve, kwargs...)
    a .= anew
    b .= bnew
    return a, b
end

# Soft minimizers use the same sparse default.
function soft_symcover_min(ϕ::AbsLog{2},
        S::Union{SparseMatrixCSC,Symmetric{<:Any,<:SparseMatrixCSC},Hermitian{<:Any,<:SparseMatrixCSC}};
        linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _soft_symcover_min_abslog2(S; linsolve, kwargs...)
    return a
end

function soft_cover_min(ϕ::AbsLog{2}, A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    a, b, _ = _soft_cover_min_abslog2(A; linsolve, kwargs...)
    return a, b
end
