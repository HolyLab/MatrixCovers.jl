# Traversal of a matrix's stored support, shared by the cover heuristics
# (geometric-mean init, feasibility boost, tightening) so each is written once
# instead of once per storage type.
#
# `foreach_support(f, A)` calls `f(i, j, v)` for every stored entry with
# `v = abs(A[i, j]) != 0`, in a storage-friendly order.
#
# `foreach_support_sym(f, A)` calls `f(i, j, v)` once per unordered index pair
# in a canonical triangle (including the diagonal), for use when `A` is known
# to be symmetric-valued. For `Tridiagonal`/`Bidiagonal`, whose two off-diagonal
# bands need not agree, the pair's value is `max(|A[i,i+1]|, |A[i+1,i]|)`
# (matching what a full-grid tighten already enforces on both entries).
#
# Both are higher-order functions rather than iterators so that `f` is
# specialized and inlined into a tight loop at each call site; every index
# used is the matrix's own (`axes`, `eachindex`), so offset axes are honored.

function foreach_support(f, A::AbstractMatrix)
    for j in axes(A, 2)
        for i in axes(A, 1)
            v = abs(A[i, j])
            iszero(v) || f(i, j, v)
        end
    end
    return nothing
end

function foreach_support_sym(f, A::AbstractMatrix)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(DimensionMismatch("foreach_support_sym requires a square matrix, got axes $(axes(A))"))
    for j in ax
        for i in first(ax):j
            v = abs(A[i, j])
            iszero(v) || f(i, j, v)
        end
    end
    return nothing
end

function foreach_support(f, D::Diagonal)
    for i in axes(D, 1)
        v = abs(D[i, i])
        iszero(v) || f(i, i, v)
    end
    return nothing
end
foreach_support_sym(f, D::Diagonal) = foreach_support(f, D)

function foreach_support(f, A::SymTridiagonal)
    ax = axes(A, 1)
    for i in ax
        v = abs(A[i, i])
        iszero(v) || f(i, i, v)
    end
    for i in first(ax):last(ax)-1
        v = abs(A[i, i+1])
        iszero(v) || (f(i, i+1, v); f(i+1, i, v))
    end
    return nothing
end

function foreach_support_sym(f, A::SymTridiagonal)
    ax = axes(A, 1)
    for i in ax
        v = abs(A[i, i])
        iszero(v) || f(i, i, v)
    end
    for i in first(ax):last(ax)-1
        v = abs(A[i, i+1])
        iszero(v) || f(i, i+1, v)
    end
    return nothing
end

function foreach_support(f, A::Bidiagonal)
    ax = axes(A, 1)
    for i in ax
        v = abs(A[i, i])
        iszero(v) || f(i, i, v)
    end
    if A.uplo == 'U'
        for i in first(ax):last(ax)-1
            v = abs(A[i, i+1])
            iszero(v) || f(i, i+1, v)
        end
    else
        for i in first(ax):last(ax)-1
            v = abs(A[i+1, i])
            iszero(v) || f(i+1, i, v)
        end
    end
    return nothing
end

function foreach_support(f, A::Tridiagonal)
    ax = axes(A, 1)
    for i in ax
        v = abs(A[i, i])
        iszero(v) || f(i, i, v)
    end
    for i in first(ax):last(ax)-1
        v = abs(A[i, i+1])
        iszero(v) || f(i, i+1, v)
        v = abs(A[i+1, i])
        iszero(v) || f(i+1, i, v)
    end
    return nothing
end

# Bidiagonal/Tridiagonal symmetric-contract traversal: a structural-zero band
# entry (e.g. the subdiagonal of an upper Bidiagonal) reads as exact zero via
# getindex, so `max` over both entries is correct without special-casing uplo.
function foreach_support_sym(f, A::Union{Bidiagonal,Tridiagonal})
    ax = axes(A, 1)
    for i in ax
        v = abs(A[i, i])
        iszero(v) || f(i, i, v)
    end
    for i in first(ax):last(ax)-1
        v = max(abs(A[i, i+1]), abs(A[i+1, i]))
        iszero(v) || f(i, i+1, v)
    end
    return nothing
end
