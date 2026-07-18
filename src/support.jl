# Traversal of a matrix's stored support, shared by the cover heuristics
# (geometric-mean init, feasibility boost, tightening) so each is written once
# instead of once per storage type.
#
# Both are higher-order functions rather than iterators so that `f` is
# specialized and inlined into a tight loop at each call site; every index
# used is the matrix's own (`axes`, `eachindex`), so offset axes are honored.

"""
    foreach_support(f, A)

Call `f(i, j, v)` once for every entry of `A` whose magnitude
`v = abs(A[i, j])` is nonzero, and return `nothing`. Entries that are zero are
skipped, so `f` never sees `v == 0`. The order is whatever suits `A`'s storage
and is not part of the contract; `i` and `j` are `A`'s own indices, so offset
axes are honored.

This is the hook through which cover algorithms read a matrix. Specializing it
is what lets a storage type be covered in time proportional to its support
rather than to `length(A)` — the package's own `SparseMatrixCSC` methods, which
walk `nzrange` instead of the full grid, are the model.

# Extending

To support a new matrix type, define

    MatrixCovers.foreach_support(f, A::MyMatrix)

which must call `f(i, j, abs(A[i, j]))` exactly once for each `(i, j)` with
`abs(A[i, j]) != 0`, must not call `f` for any other entry (a stored zero is
still a zero), and must return `nothing`. Emitting an entry twice double-counts
it in the objective; omitting one silently drops a constraint, yielding a
"cover" that does not cover.

See also: [`foreach_support_sym`](@ref).
"""
function foreach_support(f, A::AbstractMatrix)
    for j in axes(A, 2)
        for i in axes(A, 1)
            v = abs(A[i, j])
            iszero(v) || f(i, j, v)
        end
    end
    return nothing
end

"""
    foreach_support_sym(f, A)

Symmetric counterpart of [`foreach_support`](@ref): call `f(i, j, v)` once per
unordered index pair rather than once per entry, and return `nothing`. Pairs are
reported in the canonical orientation `i <= j`, the diagonal included, with
`v = abs(A[i, j])`; zero pairs are skipped. `A` must be square, or a
`DimensionMismatch` is thrown.

`A` must also be **symmetric in value**, not merely square — this is a
precondition the function cannot check cheaply and does not try to. It is what
makes reporting one member of each pair sufficient: a symmetric cover
constrains `a[i]*a[j]` by a single magnitude, so visiting `(j, i)` as well would
only duplicate it. Handing this an asymmetric matrix does not error; it silently
covers a symmetrization of it, and which one is not specified.

(The `Bidiagonal`/`Tridiagonal` methods, whose two off-diagonal bands are stored
separately and need not agree, report `max(|A[i,j]|, |A[j,i]|)` — the value a
full-grid tighten would enforce on both entries. Under the precondition the two
bands agree and this is just `abs(A[i,j])`; outside it, the choice is
robustness, not a promise.)

# Extending

To support a new matrix type, define

    MatrixCovers.foreach_support_sym(f, A::MyMatrix)

which must call `f(i, j, v)` exactly once for each pair `i <= j` with
`v = abs(A[i, j]) != 0`, must not call `f` for zero pairs, and must return
`nothing`. Reporting the same pair in both orientations double-counts it. A
type whose storage is triangular (`Symmetric{<:Any,<:SparseMatrixCSC}` in this
package's own extension) must map stored `(i, j)` with `i > j` back to `(j, i)`
rather than emit it as found.

See also: [`foreach_support`](@ref).
"""
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
