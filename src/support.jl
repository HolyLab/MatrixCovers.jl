# Traversal hooks for matrix support. Callbacks specialize at each call site,
# and indices follow the matrix axes.

"""
    foreach_support(f, A)

Call `f(i, j, abs(A[i,j]))` once per nonzero entry and return `nothing`.
Traversal order is unspecified; indices follow `axes(A)`.

Specialize this hook to support custom sparse storage in O(nnz) time.

# Extending

To support a new matrix type, define

    MatrixCovers.foreach_support(f, A::MyMatrix)

It must emit each nonzero entry exactly once, skip stored zeros, ignore callback
return values, and return `nothing`.

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
nonzero unordered pair in canonical order `i <= j`, including the diagonal.

`A` must be square and `abs.(A)` symmetric. Public symmetric solvers check this
precondition before calling the traversal.

# Objective weighting

For full-grid objective weighting, use multiplicity 1 on the diagonal and 2
off-diagonal. Constraints need no multiplicity.

# Extending

To support a new matrix type, define

    MatrixCovers.foreach_support_sym(f, A::MyMatrix)

It must emit each nonzero pair once in canonical order, skip zero pairs, ignore
callback return values, and return `nothing`. Triangular storage must map lower
entries back to `(j, i)`.

See also: [`foreach_support`](@ref).
"""
function foreach_support_sym(f, A::AbstractMatrix)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(DimensionMismatch("foreach_support_sym requires a square matrix, got axes $(string(axes(A)))"))
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

Throw unless `abs.(A)` is symmetric to within roundoff. The error names `fname`
and the first offending pair.
"""
function require_abs_symmetric(A::AbstractMatrix, fname)
    ax = axes(A, 1)
    axes(A, 2) == ax ||
        throw(DimensionMismatch("$fname requires a square matrix, got axes $(string(axes(A)))"))
    foreach_support(A) do i, j, v
        w = abs(A[j, i])
        m = max(v, w)
        abs(v - w) <= ASYMMETRY_ULPS * eps(float(real(typeof(m)))) * m || throw(ArgumentError("""
        $fname requires `abs.(A)` to be symmetric, but abs(A[$(string(i)),$(string(j))]) = $(string(v)) and \
        abs(A[$(string(j)),$(string(i))]) = $(string(w)). Wrap `A` in `Symmetric` (or `Hermitian`) to name the \
        triangle to read; that also skips this check."""))
    end
    return nothing
end

# Cache-blocked check for dense storage: the transposed reads of a column-major
# sweep miss on every entry once `A` outgrows the cache, while a block of rows and
# its transpose both fit. Only `i < j` needs testing, the diagonal being its own
# partner.
const SYMMETRY_BLOCK = 64

function require_abs_symmetric(A::StridedMatrix, fname)
    ax = axes(A, 1)
    axes(A, 2) == ax ||
        throw(DimensionMismatch("$fname requires a square matrix, got axes $(string(axes(A)))"))
    n = length(ax)
    o = first(ax) - 1
    for jb in 1:SYMMETRY_BLOCK:n
        jlast = min(jb + SYMMETRY_BLOCK - 1, n)
        for ib in 1:SYMMETRY_BLOCK:jlast
            for jp in jb:jlast
                for ip in ib:min(ib + SYMMETRY_BLOCK - 1, jp - 1)
                    i, j = ip + o, jp + o
                    v = abs(A[i, j])
                    w = abs(A[j, i])
                    m = max(v, w)
                    abs(v - w) <= ASYMMETRY_ULPS * eps(float(real(typeof(m)))) * m || throw(ArgumentError("""
                    $fname requires `abs.(A)` to be symmetric, but abs(A[$(string(i)),$(string(j))]) = $(string(v)) and \
                    abs(A[$(string(j)),$(string(i))]) = $(string(w)). Wrap `A` in `Symmetric` (or `Hermitian`) to name the \
                    triangle to read; that also skips this check."""))
                end
            end
        end
    end
    return nothing
end

# Storage that makes the precondition structural: the wrapper or the type's own
# invariant already guarantees `abs(A[i,j]) == abs(A[j,i])`.
require_abs_symmetric(::Union{Symmetric,Hermitian,Diagonal,SymTridiagonal}, fname) = nothing

# Connected components of the bipartite support graph. Labels and support
# counts use positions within each axis; unsupported rows and columns have label
# zero. Each component has an independent `a -> γ*a`, `b -> b/γ` gauge, so
# balancing must also be per component.
function _support_components(A::AbstractMatrix)
    m = length(axes(A, 1))
    n = length(axes(A, 2))
    parent = collect(1:(m + n))
    function find(p)
        while parent[p] != p
            parent[p] = parent[parent[p]]   # path halving
            p = parent[p]
        end
        return p
    end
    or = first(axes(A, 1)) - 1
    oc = first(axes(A, 2)) - 1
    nzrow = zeros(Int, m)
    nzcol = zeros(Int, n)
    foreach_support(A) do i, j, _
        p = i - or
        q = m + j - oc
        nzrow[p] += 1
        nzcol[q-m] += 1
        rp, rq = find(p), find(q)
        rp == rq || (parent[rp] = rq)
    end
    label = zeros(Int, m + n)
    ncomp = 0
    rowcomp = zeros(Int, m)
    colcomp = zeros(Int, n)
    for p in 1:(m + n)
        (p <= m ? nzrow[p] : nzcol[p-m]) > 0 || continue
        r = find(p)
        if label[r] == 0
            ncomp += 1
            label[r] = ncomp
        end
        if p <= m
            rowcomp[p] = label[r]
        else
            colcomp[p-m] = label[r]
        end
    end
    return rowcomp, colcomp, ncomp, nzrow, nzcol
end

"""
    SupportComponents

Connected components of a matrix's bipartite support graph, as returned by
[`support_components`](@ref): one vertex per row and one per column, one edge per
stored nonzero.

Component ids run `1:ncomponents(sc)`; unsupported rows and columns report `0`.
Use [`rowcomponent`](@ref) and [`colcomponent`](@ref) with the matrix's own
indices. Pass this object to [`gramcover`](@ref) to reuse the traversal.
"""
struct SupportComponents{R<:AbstractUnitRange,C<:AbstractUnitRange}
    rowcomp::Vector{Int}
    colcomp::Vector{Int}
    ncomp::Int
    rowax::R
    colax::C
end

"""
    support_components(A) -> sc::SupportComponents

Return the connected components of `A`'s bipartite support graph. The matrix is
read through [`foreach_support`](@ref).

See also: [`SupportComponents`](@ref).
"""
function support_components(A::AbstractMatrix)
    rowcomp, colcomp, ncomp, _, _ = _support_components(A)
    return SupportComponents(rowcomp, colcomp, ncomp, axes(A, 1), axes(A, 2))
end

"""
    ncomponents(sc::SupportComponents) -> Int

Number of connected components, so component ids run `1:ncomponents(sc)`.
"""
ncomponents(sc::SupportComponents) = sc.ncomp

"""
    rowcomponent(sc::SupportComponents, i) -> Int

Component id of row `i`, or `0` if that row has empty support. `i` is the
matrix's own row index.
"""
rowcomponent(sc::SupportComponents, i) = sc.rowcomp[i - first(sc.rowax) + 1]

"""
    colcomponent(sc::SupportComponents, j) -> Int

Component id of column `j`, or `0` if that column has empty support. `j` is the
matrix's own column index.
"""
colcomponent(sc::SupportComponents, j) = sc.colcomp[j - first(sc.colax) + 1]

# Number of entries `foreach_support` reports, i.e. the size of the stored support.
function _nsupport(A::AbstractMatrix)
    n = Ref(0)
    foreach_support((_, _, _) -> (n[] += 1), A)
    return n[]
end

# Support gathered as a flat edge list in 1-based position space, for solver backends
# whose models are indexed by position rather than by `A`'s own indices: `ei[e]` and
# `ej[e]` are positions within `axes(A, 1)` and `axes(A, 2)`, and `elog[e]` is
# `log(abs(A[i,j]))`, the form every model here uses. Building a model from this costs
# O(nnz) rather than the O(length(A)) of materializing the matrix in position space.
function _edge_list(A::AbstractMatrix, ::Type{T}) where T
    or, oc = first(axes(A, 1)) - 1, first(axes(A, 2)) - 1
    ei, ej, elog = Int[], Int[], T[]
    foreach_support(A) do i, j, v
        push!(ei, i - or); push!(ej, j - oc); push!(elog, log(T(v)))
    end
    return ei, ej, elog
end

# Symmetric counterpart: each off-diagonal pair is entered in both orientations and the
# diagonal once, so the list is the full-grid reading of `A` (see `foreach_support_sym`'s
# *Objective weighting*). A caller wanting the triangle instead takes the `ei[e] <= ej[e]`
# half, which is the same constraint set and the other objective convention.
function _sym_edge_list(A::AbstractMatrix, ::Type{T}) where T
    o = first(axes(A, 1)) - 1
    ei, ej, elog = Int[], Int[], T[]
    foreach_support_sym(A) do i, j, v
        lv = log(T(v))
        push!(ei, i - o); push!(ej, j - o); push!(elog, lv)
        i == j || (push!(ei, j - o); push!(ej, i - o); push!(elog, lv))
    end
    return ei, ej, elog
end

# Number of edges incident on each of `n` positions, counting one endpoint per edge.
function _degrees(endpoints, n)
    d = zeros(Int, n)
    for p in endpoints
        d[p] += 1
    end
    return d
end

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
