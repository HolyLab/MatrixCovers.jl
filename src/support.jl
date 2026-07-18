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
"cover" that does not cover. Whatever `f` returns is ignored, so a traversal
runs to completion and must not be stopped early on the strength of it.

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

`abs.(A)` must also be **symmetric**, not merely square. That is what makes
reporting one member of each pair sufficient: a symmetric cover constrains
`a[i]*a[j]` by a single magnitude, so visiting `(j, i)` as well would only
duplicate it. Note the predicate is on the magnitudes, so a complex `Hermitian`
satisfies it — `|A[i,j]| == |conj(A[j,i])|`.

This traversal does not check the precondition; the public `sym` entry points do,
before they call it (`MatrixCovers.require_abs_symmetric`).

# Extending

To support a new matrix type, define

    MatrixCovers.foreach_support_sym(f, A::MyMatrix)

which must call `f(i, j, v)` exactly once for each pair `i <= j` with
`v = abs(A[i, j]) != 0`, must not call `f` for zero pairs, and must return
`nothing`. Whatever `f` returns is ignored, so a traversal runs to completion
and must not be stopped early on the strength of it.
Reporting the same pair in both orientations double-counts it. A
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

# Width of the roundoff band the symmetry test allows, in ULPs of the larger of
# the two magnitudes. A matrix that is symmetric in exact arithmetic can land a
# ULP or two off after ordinary floating-point work — `D*A*D` for a diagonal
# rescale is the case that arises throughout this package — and rejecting that
# would reject input the algorithms handle perfectly well. The band sits far
# below any genuine asymmetry, which is what the check is for.
const ASYMMETRY_ULPS = 8

"""
    MatrixCovers.require_abs_symmetric(A, fname)

Throw unless `abs.(A)` is symmetric to within roundoff, naming `fname` and the
first offending index pair. Return `nothing` otherwise.

This is the precondition of [`foreach_support_sym`](@ref), enforced at the public
`sym` entry points rather than inside the traversal, which runs many times per
solve. An unchecked violation is not a visible failure: the cover returned would
be a cover of a symmetrization of `A`, plausible-looking and wrong.

The predicate is on the magnitudes rather than on `A` itself, which is both what
the traversal reads and what admits a complex `Hermitian`.
"""
function require_abs_symmetric(A::AbstractMatrix, fname)
    ax = axes(A, 1)
    axes(A, 2) == ax ||
        throw(DimensionMismatch("$fname requires a square matrix, got axes $(axes(A))"))
    foreach_support(A) do i, j, v
        w = abs(A[j, i])
        m = max(v, w)
        abs(v - w) <= ASYMMETRY_ULPS * eps(float(real(typeof(m)))) * m || throw(ArgumentError("""
        $fname requires `abs.(A)` to be symmetric, but abs(A[$i,$j]) = $v and \
        abs(A[$j,$i]) = $w. Wrap `A` in `Symmetric` (or `Hermitian`) to name the \
        triangle to read; that also skips this check."""))
    end
    return nothing
end

# Storage that makes the precondition structural: the wrapper or the type's own
# invariant already guarantees `abs(A[i,j]) == abs(A[j,i])`.
require_abs_symmetric(::Union{Symmetric,Hermitian,Diagonal,SymTridiagonal}, fname) = nothing

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

# The two band entries agree to within `ASYMMETRY_ULPS`, so `max` reports the one
# that dominates both without special-casing uplo.
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
