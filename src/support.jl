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

# Objective weighting

Because each pair is reported once, a caller accumulating a cover objective must
supply the multiplicity itself: `w = (i == j) ? 1 : 2`. That reproduces the
`∑_{i,j}` convention of [`cover_objective`](@ref), which runs over the full grid
and so counts each off-diagonal pair twice and each diagonal entry once. The
constraint set needs no such correction — `a[i]*a[j] >= |A[i,j]|` and its
transpose are the same constraint, so imposing it on the `i <= j` triangle alone
is equivalent to imposing it everywhere. Every solver in this package minimizes
the full-grid objective, so a cover's reported score and the quantity that was
minimized agree.

# Extending

To support a new matrix type, define

    MatrixCovers.foreach_support_sym(f, A::MyMatrix)

which must call `f(i, j, v)` exactly once for each pair `i <= j` with
`v = abs(A[i, j]) != 0`, must not call `f` for zero pairs, and must return
`nothing`. Whatever `f` returns is ignored, so a traversal runs to completion
and must not be stopped early on the strength of it.
Reporting the same pair in both orientations double-counts it: the off-diagonal
weight of 2 is the caller's to apply, per *Objective weighting* above, so a pair
emitted twice is weighted 4. A
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

# Support of `A` gathered into per-group neighbor lists, in compressed form: the
# entries of group `g` occupy the slots `_slots(S, g)`, with `S.idx[s]` the
# partner index and `S.val[s]` the magnitude.
#
# A coordinate-descent kernel revisits every row on every sweep. Reading them
# through `foreach_support` directly would mean one traversal per sweep, and a
# traversal cannot be restarted mid-row; gathering once up front costs O(nnz)
# storage and turns each sweep into a walk over the support instead of over the
# full grid.
#
# `ptr` is indexed by position within `ax` rather than by the index itself, so
# offset axes need no special case.
struct GroupedSupport{T,I,R<:AbstractUnitRange}
    ax::R
    ptr::Vector{Int}
    idx::Vector{I}
    val::Vector{T}
end

_slots(S::GroupedSupport, g) = (p = g - first(S.ax) + 1; S.ptr[p]:S.ptr[p+1]-1)

# Number of stored entries in group `g`.
_ngroup(S::GroupedSupport, g) = length(_slots(S, g))

# Build from a traversal. `each` is called twice with a callback `g(group,
# partner, v)`: once to count each group's entries, once to fill them.
function _grouped_support(each, ax::AbstractUnitRange, ::Type{I}, ::Type{T}) where {I,T}
    off = first(ax) - 1
    ptr = zeros(Int, length(ax) + 1)
    each() do g, _, _
        ptr[g-off+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    n = ptr[end] - 1
    idx = Vector{I}(undef, n)
    val = Vector{T}(undef, n)
    cursor = ptr[1:end-1]        # next free slot of each group, by position
    each() do g, p, v
        s = cursor[g-off]
        idx[s] = p
        val[s] = T(v)
        cursor[g-off] = s + 1
    end
    return GroupedSupport(ax, ptr, idx, val)
end

# Group the support of `A` by row; partners are column indices.
_row_support(A::AbstractMatrix, ::Type{T}) where {T} =
    _grouped_support(axes(A, 1), eltype(axes(A, 2)), T) do g
        foreach_support((i, j, v) -> g(i, j, v), A)
    end

# Group the support of `A` by column; partners are row indices.
_col_support(A::AbstractMatrix, ::Type{T}) where {T} =
    _grouped_support(axes(A, 2), eltype(axes(A, 1)), T) do g
        foreach_support((i, j, v) -> g(j, i, v), A)
    end

# Group the symmetric support of `A` by row: group `i` holds every `j` with
# `abs(A[i,j]) != 0`, the off-diagonal pairs entered in both orientations and the
# diagonal once. That is the full-grid reading of a row, so a kernel accumulating
# over these groups gets the `∑_{i,j}` weighting of `cover_objective` — each
# off-diagonal pair twice, each diagonal entry once — without applying a
# multiplicity factor of its own.
_sym_support(A::AbstractMatrix, ::Type{T}) where {T} =
    _grouped_support(axes(A, 1), eltype(axes(A, 1)), T) do g
        foreach_support_sym(A) do i, j, v
            g(i, j, v)
            i == j || g(j, i, v)
        end
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
