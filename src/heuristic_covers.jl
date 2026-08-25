# Fast O(mn) hard-cover heuristics `symcover` and `cover`, together with the
# tightening and initialization routines they (and the soft covers) build on.

# ============================================================
# Public interface
# ============================================================

"""
    a = symcover(ϕ, A; maxiter=3)
    a = symcover(A; maxiter=3)

Given a square matrix `A` assumed to be symmetric, return a vector `a`
representing a symmetric hard cover of `A`: `a[i] * a[j] >= abs(A[i, j])` for
all `i`, `j`.

The method initializes from per-row geometric means, covers the most-violated
entries first, then applies `maxiter` tightening iterations.

`ϕ` is accepted for API compatibility but is currently ignored.
For a cover that provably minimizes a given `ϕ`, use [`symcover_min`](@ref).

See also: [`symcover!`](@ref), [`symcover_min`](@ref), [`soft_symcover`](@ref), [`cover`](@ref).

# Examples

```jldoctest
julia> A = [4 1; 1 4];

julia> a = symcover(A)
2-element Vector{Float64}:
 2.0
 2.0

julia> a * a'   # covers |A|: a[i]*a[j] >= abs(A[i, j])
2×2 Matrix{Float64}:
 4.0  4.0
 4.0  4.0
```
"""
symcover(ϕ::AbstractCoverPenalty, A::AbstractMatrix; kwargs...) = symcover(A; kwargs...)

function symcover(A::AbstractMatrix; kwargs...)
    axes(A, 2) == axes(A, 1) || throw(ArgumentError("symcover requires a square matrix"))
    T = float(real(eltype(A)))
    a = similar(Array{T}, axes(A, 1))
    return symcover!(a, A; kwargs...)
end

"""
    a = symcover!(ϕ, a, A; maxiter=3)
    a = symcover!(a, A; maxiter=3)

Mutating counterpart of [`symcover`](@ref): writes the symmetric hard cover
into `a` and returns it, rather than allocating a new vector. `eachindex(a)`
must match `axes(A, 1)` (and `A` must be square). `ϕ` has the same meaning as in
[`symcover`](@ref), and is likewise ignored by the current heuristic covers.

See also: [`symcover`](@ref).
"""
symcover!(ϕ::AbstractCoverPenalty, a::AbstractVector, A::AbstractMatrix; kwargs...) = symcover!(a, A; kwargs...)

function symcover!(a::AbstractVector, A::AbstractMatrix; kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover! requires a square matrix"))
    require_abs_symmetric(A, :symcover!)
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(ax))"))
    return _symcover!(a, A; kwargs...)
end

function _symcover!(a::AbstractVector, A::AbstractMatrix; maxiter::Int=3)
    T = float(real(eltype(a)))
    if _use_dense_grid(A, T)
        _symcover_dense!(a, A, T, maxiter)
    else
        sup = flat_support_sym(A, T)
        unconstrained_min!(AbsLog{2}(), a, sup)
        boost_feasible!(a, sup)
        tighten_cover!(a, sup; maxiter)
    end
    # Certify against `A` after log-domain tightening.
    return _certify_cover!(a, A, :symcover)
end

"""
    a, b = cover(ϕ, A; maxiter=3)
    a, b = cover(A; maxiter=3)

Given a matrix `A`, return vectors `a` and `b` such that
`a[i] * b[j] >= abs(A[i, j])` for all `i`, `j`. The method initializes from row
and column geometric means, covers the most-violated entries first, then applies
`maxiter` tightening iterations.

The factors use the per-component balance convention described by
[`cover_min`](@ref).

`ϕ` is accepted for API compatibility but is currently ignored.
For a cover that provably minimizes a given `ϕ`, use [`cover_min`](@ref).

See also: [`cover!`](@ref), [`cover_min`](@ref), [`symcover`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [1 2 3; 6 5 4];

julia> a, b = cover(A)
([1.2544610775677627, 3.475905976749231], [1.7261686708831454, 1.621762761307448, 2.3914651906272066])

julia> a * b'
2×3 Matrix{Float64}:
 2.16541  2.03444  3.0
 6.0      5.63709  8.31251
```
"""
cover(ϕ::AbstractCoverPenalty, A::AbstractMatrix; kwargs...) = cover(A; kwargs...)

function cover(A::AbstractMatrix; kwargs...)
    T = float(real(eltype(A)))
    a = similar(Array{T}, axes(A, 1))
    b = similar(Array{T}, axes(A, 2))
    return cover!(a, b, A; kwargs...)
end

# Adjoint/Transpose wrappers for cover.
function cover(A::Adjoint; kwargs...)
    a, b = cover(parent(A); kwargs...)
    return b, a
end
function cover(A::Transpose; kwargs...)
    a, b = cover(parent(A); kwargs...)
    return b, a
end

"""
    a, b = cover!(ϕ, a, b, A; maxiter=3)
    a, b = cover!(a, b, A; maxiter=3)

Mutating counterpart of [`cover`](@ref): writes the hard cover into `a` and
`b` and returns them, rather than allocating new vectors. `eachindex(a)` must
match `axes(A, 1)` and `eachindex(b)` must match `axes(A, 2)`. `ϕ` has the same
meaning as in [`cover`](@ref), and is likewise ignored by the current heuristic
covers.

See also: [`cover`](@ref).
"""
cover!(ϕ::AbstractCoverPenalty, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    cover!(a, b, A; kwargs...)

function cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    return _cover!(a, b, A; kwargs...)
end

function _cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; maxiter::Int=3)
    T = float(promote_type(eltype(a), eltype(b)))
    if _use_dense_grid(A, T)
        _cover_dense!(a, b, A, T, maxiter)
    else
        sup = flat_support(A, T)
        unconstrained_min!(AbsLog{2}(), a, b, sup)
        boost_feasible!(a, b, sup)
        tighten_cover!(a, b, sup; maxiter)
        # Apply the package's balance convention, then restore coverage lost to rounding.
        _balance_cover!(a, b, A)
        inflate_feasible!(a, b, sup)
    end
    return _certify_cover!(a, b, A, :cover)
end

# Adjoint/Transpose wrappers for cover!.
function cover!(a::AbstractVector, b::AbstractVector, A::Adjoint; kwargs...)
    cover!(b, a, parent(A); kwargs...)
    return a, b
end
function cover!(a::AbstractVector, b::AbstractVector, A::Transpose; kwargs...)
    cover!(b, a, parent(A); kwargs...)
    return a, b
end

# ============================================================
# Internal helpers
# ============================================================
# Matrix support flattened in traversal order as row, column, and `log|A_ij|`
# arrays.
struct FlatSupport{Ti<:Integer,Tj<:Integer,T}
    is::Vector{Ti}
    js::Vector{Tj}
    lv::Vector{T}
end

# Use `Int32` when it contains the axis; otherwise preserve the axis index type.
function _flat_index_type(ax)
    I = eltype(ax)
    I <: Integer || return I
    isempty(ax) && return Int32
    return (typemin(Int32) <= first(ax) && last(ax) <= typemax(Int32)) ? Int32 : I
end

# Storage-specific upper bounds for `sizehint!`; zero means unknown.
_support_sizehint(::AbstractMatrix) = 0
_support_sizehint_sym(::AbstractMatrix) = 0

# The outer methods select concrete index types for the traversal.
flat_support_sym(A::AbstractMatrix, ::Type{T}) where T =
    _flat_support_sym(A, T, _flat_index_type(axes(A, 1)))

function _flat_support_sym(A::AbstractMatrix, ::Type{T}, ::Type{Ti}) where {T,Ti}
    is, js, lv = Ti[], Ti[], T[]
    hint = _support_sizehint_sym(A)
    if hint > 0
        sizehint!(is, hint); sizehint!(js, hint); sizehint!(lv, hint)
    end
    foreach_support_sym(A) do i, j, v
        push!(is, i); push!(js, j); push!(lv, T(v))
    end
    _fastlog!(lv)   # one vectorized pass over the collected magnitudes
    return FlatSupport(is, js, lv)
end

flat_support(A::AbstractMatrix, ::Type{T}) where T =
    _flat_support(A, T, _flat_index_type(axes(A, 1)), _flat_index_type(axes(A, 2)))

function _flat_support(A::AbstractMatrix, ::Type{T}, ::Type{Ti}, ::Type{Tj}) where {T,Ti,Tj}
    is, js, lv = Ti[], Tj[], T[]
    hint = _support_sizehint(A)
    if hint > 0
        sizehint!(is, hint); sizehint!(js, hint); sizehint!(lv, hint)
    end
    foreach_support(A) do i, j, v
        push!(is, i); push!(js, j); push!(lv, T(v))
    end
    _fastlog!(lv)   # one vectorized pass over the collected magnitudes
    return FlatSupport(is, js, lv)
end

# Select violated entries in traversal order without branching.
function _flat_violated(sup::FlatSupport{Ti,Tj,T}, la, lb, nviol::Int) where {Ti,Tj,T}
    is, js, lv = sup.is, sup.js, sup.lv
    entries = Vector{Tuple{Ti,Tj,T}}(undef, nviol + 1)
    k = 1
    for p in eachindex(is, js, lv)
        i, j, lvp = is[p], js[p], lv[p]
        entries[k] = (i, j, lvp)
        k += ifelse(lvp - la[i] - lb[j] > zero(T), 1, 0)
    end
    resize!(entries, nviol)
    return entries
end

# Apply the row/column balance convention independently to each support
# component. Rounding the shift to a power of two preserves cover products
# exactly, at the cost of balancing only within a factor of `sqrt(2)`.
function _balance_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    rowcomp, colcomp, ncomp, nzrow, nzcol = _support_components(A)
    return _balance_cover!(a, b, rowcomp, colcomp, ncomp, nzrow, nzcol)
end

# Balance from precomputed component labels and support counts.
function _balance_cover!(a::AbstractVector, b::AbstractVector, rowcomp::Vector{Int},
                         colcomp::Vector{Int}, ncomp::Int, nzrow::Vector{Int},
                         nzcol::Vector{Int})
    T = float(promote_type(eltype(a), eltype(b)))
    iszero(ncomp) && return a, b
    Lα = zeros(T, ncomp)
    Lβ = zeros(T, ncomp)
    nnz = zeros(Int, ncomp)
    # Weight each scale by its support count.
    for (p, i) in enumerate(eachindex(a))
        c = rowcomp[p]
        c == 0 && continue
        Lα[c] += nzrow[p] * log2(T(a[i]))
        nnz[c] += nzrow[p]
    end
    for (q, j) in enumerate(eachindex(b))
        c = colcomp[q]
        c == 0 && continue
        Lβ[c] += nzcol[q] * log2(T(b[j]))
    end
    # An integer base-2 exponent makes the rescaling exact.
    gamma = [exp2(round((Lβ[c] - Lα[c]) / (2 * nnz[c]))) for c in 1:ncomp]
    for (p, i) in enumerate(eachindex(a))
        c = rowcomp[p]
        c == 0 && continue
        a[i] *= gamma[c]
    end
    for (q, j) in enumerate(eachindex(b))
        c = colcomp[q]
        c == 0 && continue
        b[j] /= gamma[c]
    end
    return a, b
end


# Analytical minimizer of the unconstrained `AbsLog{2}` symmetric objective
#   ∑_{i,j: A[i,j]≠0} (log(a[i]*a[j]) - log|A[i,j]|)²
# Returns row support counts. The Sherman-Morrison approximation is exact on
# complete support.
function unconstrained_min!(::AbsLog{2}, a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(string(axes(A))), axes(a)=$(string(axes(a)))"))
    loga = fill!(similar(a), zero(T))
    nza  = zeros(Int, ax)
    foreach_support_sym(A) do i, j, v
        lAij = log(T(v))
        loga[i] += lAij
        nza[i]  += 1
        if i != j
            loga[j] += lAij
            nza[j]  += 1
        end
    end
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in ax
        # exp can underflow for extreme dynamic range; a zero scale on a
        # supported row would make the boost's log-deficits infinite, so
        # clamp to the smallest normal positive value.
        a[i] = iszero(nza[i]) ? zero(T) : max(exp(loga[i] / nza[i] - halfmu), floatmin(T))
    end
    return nza
end

# The symmetric objective over a flattened support: `sup` must have been built
# by `flat_support_sym` over a matrix whose axes match `eachindex(a)`.
function unconstrained_min!(::AbsLog{2}, a::AbstractVector{T}, sup::FlatSupport) where T
    is, js, lv = sup.is, sup.js, sup.lv
    loga = fill!(similar(a), zero(T))
    nza  = zeros(Int, eachindex(a))
    for k in eachindex(is, js, lv)
        i, j, lAij = is[k], js[k], lv[k]
        loga[i] += lAij
        nza[i]  += 1
        if i != j
            loga[j] += lAij
            nza[j]  += 1
        end
    end
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in eachindex(a)
        # exp can underflow for extreme dynamic range; a zero scale on a
        # supported row would make the boost's log-deficits infinite, so
        # clamp to the smallest normal positive value.
        a[i] = iszero(nza[i]) ? zero(T) : max(exp(loga[i] / nza[i] - halfmu), floatmin(T))
    end
    return nza
end

function unconstrained_min!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    T = float(promote_type(eltype(a), eltype(b)))
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, b, A)` requires row indices of `A` to match `a`, got axes(A, 1)=$(string(axes(A, 1))), axes(a)=$(string(axes(a)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, b, A)` requires column indices of `A` to match `b`, got axes(A, 2)=$(string(axes(A, 2))), axes(b)=$(string(axes(b)))"))
    loga = fill!(similar(a, T), zero(T))
    logb = fill!(similar(b, T), zero(T))
    nza  = zeros(Int, axes(A, 1))
    nzb  = zeros(Int, axes(A, 2))
    foreach_support(A) do i, j, v
        lAij = log(T(v))
        loga[i] += lAij
        logb[j] += lAij
        nza[i]  += 1
        nzb[j]  += 1
    end
    # Each stored entry contributes lAij to loga exactly once and increments
    # nza exactly once, so these sums equal the per-entry running totals.
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in axes(A, 1)
        # exp can underflow for extreme dynamic range; a zero scale on a
        # supported row would make the boost's log-deficits infinite, so
        # clamp to the smallest normal positive value.
        a[i] = iszero(nza[i]) ? zero(T) : max(exp(loga[i] / nza[i] - halfmu), floatmin(T))
    end
    for j in axes(A, 2)
        b[j] = iszero(nzb[j]) ? zero(T) : max(exp(logb[j] / nzb[j] - halfmu), floatmin(T))
    end
    return nza, nzb
end

# The asymmetric objective over a flattened support: `sup` must have been
# built by `flat_support` over a matrix whose row and column axes match
# `eachindex(a)` and `eachindex(b)`.
function unconstrained_min!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, sup::FlatSupport)
    T = float(promote_type(eltype(a), eltype(b)))
    is, js, lv = sup.is, sup.js, sup.lv
    loga = fill!(similar(a, T), zero(T))
    logb = fill!(similar(b, T), zero(T))
    nza  = zeros(Int, eachindex(a))
    nzb  = zeros(Int, eachindex(b))
    for k in eachindex(is, js, lv)
        i, j, lAij = is[k], js[k], lv[k]
        loga[i] += lAij
        logb[j] += lAij
        nza[i]  += 1
        nzb[j]  += 1
    end
    # Each stored entry contributes lAij to loga exactly once and increments
    # nza exactly once, so these sums equal the per-entry running totals.
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in eachindex(a)
        # exp can underflow for extreme dynamic range; a zero scale on a
        # supported row would make the boost's log-deficits infinite, so
        # clamp to the smallest normal positive value.
        a[i] = iszero(nza[i]) ? zero(T) : max(exp(loga[i] / nza[i] - halfmu), floatmin(T))
    end
    for j in eachindex(b)
        b[j] = iszero(nzb[j]) ? zero(T) : max(exp(logb[j] / nzb[j] - halfmu), floatmin(T))
    end
    return nza, nzb
end

# Feasible cover starting from the diagonal alone, resolved by
# `boost_feasible_seq!`. Unlike `boost_feasible!`, a zero entry of `a` going
# into that call means "not yet resolved", not "permanently unsupported" —
# every diagonal-zero row is deferred until an off-diagonal neighbor supplies
# a scale for it (see `boost_feasible_seq!`).
function init_feasible_diag!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`init_feasible_diag!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(string(axes(A))), axes(a)=$(string(axes(a))))"))
    # A diagonal entry the traversal skips is zero, which is the "not yet resolved"
    # value `boost_feasible_seq!` expects.
    fill!(a, zero(T))
    foreach_support_sym(A) do i, j, v
        i == j && (a[i] = sqrt(T(v)))
    end
    return boost_feasible_seq!(a, A)
end

# Shrink `x` by `exp(lr/2)` in log space, clamping supported scales at
# `floatmin` on underflow.
function _tighten_shrink(x, lr)
    T = float(promote_type(typeof(x), typeof(lr)))
    y = T(x) / exp(T(lr) / 2)
    if iszero(y) && !iszero(x)
        y = max(exp(log(T(x)) - T(lr) / 2), floatmin(T))
    end
    return y
end

function tighten_cover!(a::AbstractVector{T}, A::AbstractMatrix; maxiter::Int=3) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    lratio = similar(a)
    la = similar(a)
    for _ in 1:maxiter
        map!(log, la, a)   # log(0) = -Inf marks zero scales; see below
        fill!(lratio, T(Inf))
        # A is assumed symmetric-valued; entries not visited by `foreach_support_sym`
        # (zero, or the redundant triangle) leave lratio at +Inf, a no-op below. Working
        # in log-ratio (rather than a[i]*a[j]/v) keeps the comparison finite even when
        # the linear-space product would overflow for extreme dynamic range; a zero
        # scale gives lr = -Inf, marking the row uncoverable by any finite rescale.
        foreach_support_sym(A) do i, j, v
            lr = la[i] + la[j] - log(T(v))
            lratio[i] = min(lratio[i], lr)
            i == j || (lratio[j] = min(lratio[j], lr))
        end
        for i in eachindex(a)
            lr = lratio[i]
            # lr == -Inf marks a row whose cover product vanishes on some entry; no
            # finite rescale covers it, so leave the scale unchanged. lr == +Inf marks
            # a row untouched this pass (a[i] is already 0 there), likewise a no-op.
            isinf(lr) || (a[i] = _tighten_shrink(a[i], lr))
        end
    end
    return a
end

# Symmetric tightening over a flattened support; the log-ratio convention is
# that of the matrix method. The `ifelse` minimum rejects NaN (an infinite
# entry against a zero scale), leaving such rows at the +Inf no-op, and lets
# the loop run branch-free; a diagonal entry updates its row twice, which the
# minimum absorbs.
function tighten_cover!(a::AbstractVector{T}, sup::FlatSupport; maxiter::Int=3) where T
    is, js, lv = sup.is, sup.js, sup.lv
    lratio = similar(a)
    la = similar(a)
    for _ in 1:maxiter
        map!(log, la, a)   # log(0) = -Inf marks zero scales; see the matrix method
        fill!(lratio, T(Inf))
        for k in eachindex(is, js, lv)
            i, j = is[k], js[k]
            lr = la[i] + la[j] - lv[k]
            lratio[i] = ifelse(lr < lratio[i], lr, lratio[i])
            lratio[j] = ifelse(lr < lratio[j], lr, lratio[j])
        end
        for i in eachindex(a)
            lr = lratio[i]
            isinf(lr) || (a[i] = _tighten_shrink(a[i], lr))
        end
    end
    return a
end

function tighten_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; maxiter::Int=3)
    T = float(promote_type(eltype(a), eltype(b)))
    eachindex(a) == axes(A, 1) || throw(DimensionMismatch("indices of a must match row-indexing of A"))
    eachindex(b) == axes(A, 2) || throw(DimensionMismatch("indices of b must match column-indexing of A"))
    lratioa = fill(T(Inf), eachindex(a))
    lratiob = fill(T(Inf), eachindex(b))
    la, lb = similar(a, T), similar(b, T)
    for _ in 1:maxiter
        map!(log, la, a)   # log(0) = -Inf marks zero scales; see below
        map!(log, lb, b)
        fill!(lratioa, T(Inf))
        fill!(lratiob, T(Inf))
        # Working in log-ratio (rather than a[i]*b[j]/v) keeps the comparison
        # finite even when the linear-space product would overflow for extreme
        # dynamic range; a zero scale gives lr = -Inf, marking the row or column
        # uncoverable by any finite rescale.
        foreach_support(A) do i, j, v
            lr = la[i] + lb[j] - log(T(v))
            lratioa[i] = min(lratioa[i], lr)
            lratiob[j] = min(lratiob[j], lr)
        end
        for i in eachindex(a)
            lr = lratioa[i]
            # lr == -Inf marks a row whose cover product vanishes on some entry;
            # no finite rescale covers it, so leave the scale unchanged.
            isinf(lr) || (a[i] = _tighten_shrink(a[i], lr))
        end
        for j in eachindex(b)
            lr = lratiob[j]
            # lr == -Inf marks a column whose cover product vanishes on some
            # entry; no finite rescale covers it, so leave the scale unchanged.
            isinf(lr) || (b[j] = _tighten_shrink(b[j], lr))
        end
    end
    return a, b
end

# Asymmetric tightening over a flattened support; see the symmetric flat
# method for the `ifelse`-minimum convention.
function tighten_cover!(a::AbstractVector, b::AbstractVector, sup::FlatSupport; maxiter::Int=3)
    T = float(promote_type(eltype(a), eltype(b)))
    is, js, lv = sup.is, sup.js, sup.lv
    lratioa = fill(T(Inf), eachindex(a))
    lratiob = fill(T(Inf), eachindex(b))
    la, lb = similar(a, T), similar(b, T)
    for _ in 1:maxiter
        map!(log, la, a)   # log(0) = -Inf marks zero scales; see the matrix method
        map!(log, lb, b)
        fill!(lratioa, T(Inf))
        fill!(lratiob, T(Inf))
        for k in eachindex(is, js, lv)
            i, j = is[k], js[k]
            lr = la[i] + lb[j] - lv[k]
            lratioa[i] = ifelse(lr < lratioa[i], lr, lratioa[i])
            lratiob[j] = ifelse(lr < lratiob[j], lr, lratiob[j])
        end
        for i in eachindex(a)
            lr = lratioa[i]
            isinf(lr) || (a[i] = _tighten_shrink(a[i], lr))
        end
        for j in eachindex(b)
            lr = lratiob[j]
            isinf(lr) || (b[j] = _tighten_shrink(b[j], lr))
        end
    end
    return a, b
end

# Adjoint/Transpose wrappers for tighten_cover!.
function tighten_cover!(a::AbstractVector, b::AbstractVector, A::Adjoint; kwargs...)
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end
function tighten_cover!(a::AbstractVector, b::AbstractVector, A::Transpose; kwargs...)
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end

# Approximate greedy max-deficit boost. Deficits only decrease, so entries move
# to lower buckets. Log-deficit buckets preserve covariance except for ties.
const BOOST_BUCKET_WIDTH = log(2) / 4   # quality indistinguishable from exact greedy; only bucket count grows as w shrinks

# Visit deficit buckets from highest to lowest. Original entries are stored
# contiguously; entries demoted from higher buckets use per-bucket stacks.
function bucket_boost!(deficit::F, apply!::G, entries::AbstractVector, ::Type{T}, zmax::T) where {F,G,T}
    zmax > zero(T) || return
    w = T(BOOST_BUCKET_WIDTH)
    B = max(1, ceil(Int, zmax / w))
    # `z` is a positive log difference no greater than `zmax`; its spacing keeps
    # `z / w` from underflowing, so the ceiling remains in `1:B`.
    bucketof(z) = unsafe_trunc(Int, ceil(z / w))
    ptr = zeros(Int, B + 1)
    for entry in entries
        z = deficit(entry)
        z > zero(T) || continue
        ptr[bucketof(z)+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    cursor = ptr[1:end-1]              # next free slot of each level
    sorted = similar(entries, ptr[end] - 1)
    for k in reverse(eachindex(entries))
        entry = entries[k]
        z = deficit(entry)
        z > zero(T) || continue
        b = bucketof(z)
        sorted[cursor[b]] = entry
        cursor[b] += 1
    end
    # Demoted entries, as a stack per level over one shared array.
    dhead = zeros(Int, B)
    dentry = similar(entries, 0)
    dnext = Int[]
    demote!(entry, b2) = (push!(dentry, entry); push!(dnext, dhead[b2]); dhead[b2] = length(dentry))
    function visit!(entry, b)
        z = deficit(entry)
        z > zero(T) || return
        b2 = bucketof(z)
        b2 < b ? demote!(entry, b2) : apply!(entry, z)
        return
    end
    for b in B:-1:1
        e = dhead[b]
        while e != 0
            enext = dnext[e]              # save before a further demotion appends
            visit!(dentry[e], b)
            e = enext
        end
        for s in ptr[b]:ptr[b+1]-1
            visit!(sorted[s], b)
        end
    end
    return
end

# Symmetric-contract feasibility boost: scale `a` in place so that
# `a[i]*a[j] >= |A[i,j]|`, up to the round-off of the log-domain updates,
# for every entry visited by `foreach_support_sym`
# (the diagonal included, so no separate clamp step is needed). Requires a
# start with strictly positive scale on every supported row (the geometric-mean
# init from `unconstrained_min!` guarantees this).
function boost_feasible!(a::AbstractVector{T}, A::AbstractMatrix) where T
    IdxT = eltype(eachindex(a))
    # `la` caches log.(a) and is updated alongside `a`, so deficits cost no log
    # calls; log(0) = -Inf on unsupported rows is never read (every entry's
    # endpoints pass the positive-scale check below). Growing log-scales
    # directly (rather than multiplying by exp(z/2)) stays finite even when
    # exp(z/2) alone would overflow.
    la = map(log, a)
    # `entries` holds only entries already violated at this starting point:
    # bucket_boost! never revisits an entry once satisfied, so a still-slack
    # entry need not be stored at all. A zero scale on either endpoint makes
    # its deficit +Inf, so such entries are always violated and the fail-fast
    # check below always runs on them. Two passes (count then fill) allocate
    # `entries` once at its exact size, instead of the repeated grow-and-copy
    # of building it with `push!`.
    nviol = Ref(0)
    zmax = Ref(zero(T))
    foreach_support_sym(A) do i, j, v
        z = log(T(v)) - la[i] - la[j]
        if z > zero(T)
            nviol[] += 1
            zmax[] = max(zmax[], z)
        end
    end
    entries = Vector{Tuple{IdxT,IdxT,T}}(undef, nviol[])
    nfill = Ref(0)
    foreach_support_sym(A) do i, j, v
        lv = log(T(v))
        z = lv - la[i] - la[j]
        if z > zero(T)
            (iszero(a[i]) || iszero(a[j])) &&
                throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row"))
            nfill[] += 1
            entries[nfill[]] = (i, j, lv)
        end
    end
    deficit((i, j, lv)) = lv - la[i] - la[j]
    function apply!((i, j, lv), z)
        h = z / 2
        la[i] += h; a[i] = exp(la[i])
        i == j || (la[j] += h; a[j] = exp(la[j]))
    end
    bucket_boost!(deficit, apply!, entries, T, zmax[])
    return a
end

# Symmetric boost over a flattened support. As in the matrix method, only
# entries already violated at the start are stored; the count pass runs
# branchlessly (about half a fresh start's entries violate, so a data-dependent
# branch would mispredict constantly), and `_flat_violated` selects them the
# same way. A zero scale on a supported row makes some deficit +Inf, which the
# `isfinite` check below turns into the matrix method's error.
function boost_feasible!(a::AbstractVector{T}, sup::FlatSupport) where T
    is, js, lv = sup.is, sup.js, sup.lv
    # `la` caches log.(a) and is updated alongside `a`; see the matrix method.
    la = map(log, a)
    nviol = 0
    zmax = zero(T)
    for k in eachindex(is, js, lv)
        z = lv[k] - la[is[k]] - la[js[k]]
        nviol += ifelse(z > zero(T), 1, 0)
        zmax = ifelse(z > zmax, z, zmax)
    end
    isfinite(zmax) ||
        throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row"))
    entries = _flat_violated(sup, la, la, nviol)
    deficit((i, j, lvk)) = lvk - la[i] - la[j]
    function apply!((i, j, lvk), z)
        h = z / 2
        la[i] += h; a[i] = exp(la[i])
        i == j || (la[j] += h; a[j] = exp(la[j]))
    end
    bucket_boost!(deficit, apply!, entries, T, zmax)
    return a
end

# Asymmetric feasibility boost: scale `a`, `b` in place so that
# `a[i]*b[j] >= |A[i,j]|`, up to the round-off of the log-domain updates,
# for every entry visited by `foreach_support`. The
# diagonal is treated as an ordinary entry. Requires a start with strictly
# positive scale on every supported row of `a` and column of `b`.
function boost_feasible!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    T = float(promote_type(eltype(a), eltype(b)))
    IdxA, IdxB = eltype(eachindex(a)), eltype(eachindex(b))
    # `la`/`lb` cache log.(a)/log.(b) and are updated alongside `a`/`b`; see
    # the symmetric method.
    la, lb = map(log, a), map(log, b)
    # `entries` holds only entries already violated at this starting point;
    # see the symmetric method for why this is safe and why the fail-fast
    # check always fires on a zero-scale endpoint. Two passes (count then
    # fill) allocate `entries` once at its exact size, instead of the
    # repeated grow-and-copy of building it with `push!`.
    nviol = Ref(0)
    zmax = Ref(zero(T))
    foreach_support(A) do i, j, v
        z = log(T(v)) - la[i] - lb[j]
        if z > zero(T)
            nviol[] += 1
            zmax[] = max(zmax[], z)
        end
    end
    entries = Vector{Tuple{IdxA,IdxB,T}}(undef, nviol[])
    nfill = Ref(0)
    foreach_support(A) do i, j, v
        lv = log(T(v))
        z = lv - la[i] - lb[j]
        if z > zero(T)
            (iszero(a[i]) || iszero(b[j])) &&
                throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row/column"))
            nfill[] += 1
            entries[nfill[]] = (i, j, lv)
        end
    end
    deficit((i, j, lv)) = lv - la[i] - lb[j]
    function apply!((i, j, lv), z)
        h = z / 2
        la[i] += h; a[i] = exp(la[i])
        lb[j] += h; b[j] = exp(lb[j])
    end
    bucket_boost!(deficit, apply!, entries, T, zmax[])
    return a, b
end

# Asymmetric boost over a flattened support; see the symmetric flat method.
function boost_feasible!(a::AbstractVector, b::AbstractVector, sup::FlatSupport)
    T = float(promote_type(eltype(a), eltype(b)))
    is, js, lv = sup.is, sup.js, sup.lv
    # `la`/`lb` cache log.(a)/log.(b) and are updated alongside `a`/`b`; see
    # the matrix methods.
    la, lb = map(log, a), map(log, b)
    nviol = 0
    zmax = zero(T)
    for k in eachindex(is, js, lv)
        z = lv[k] - la[is[k]] - lb[js[k]]
        nviol += ifelse(z > zero(T), 1, 0)
        zmax = ifelse(z > zmax, z, zmax)
    end
    isfinite(zmax) ||
        throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row/column"))
    entries = _flat_violated(sup, la, lb, nviol)
    deficit((i, j, lvk)) = lvk - la[i] - lb[j]
    function apply!((i, j, lvk), z)
        h = z / 2
        la[i] += h; a[i] = exp(la[i])
        lb[j] += h; b[j] = exp(lb[j])
    end
    bucket_boost!(deficit, apply!, entries, T, zmax)
    return a, b
end

# Sequential nearest-neighbor feasibility propagation by diagonal offset.
function boost_feasible_seq!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`boost_feasible_seq!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(string(axes(A))), axes(a)=$(string(axes(a))))"))
    I = eltype(ax)

    # The support gathered as off-diagonal pairs, ordered by increasing offset and
    # then by row, which is the propagation order the heuristic is defined by.
    pairs = Tuple{I,I,T}[]
    foreach_support_sym(A) do i, j, v
        i == j || push!(pairs, (i, j, T(v)))
    end
    sort!(pairs; by = ((k, l, _),) -> (l - k, k))

    deferred = Tuple{I,I,T}[]
    for (k, l, Akl) in pairs
        ak, al = a[k], a[l]
        if !iszero(ak) && !iszero(al)
            aprod = ak * al
            if aprod < Akl
                s = sqrt(Akl / aprod)
                a[k] *= s; a[l] *= s
            end
        elseif !iszero(ak)
            a[l] = Akl / ak
        elseif !iszero(al)
            a[k] = Akl / al
        else
            push!(deferred, (k, l, Akl))
        end
    end

    # Each vertex is assigned at most once, so worklist resolution is linear in
    # the number of deferred pairs.
    if !isempty(deferred)
        o = first(ax) - 1
        inc = [Int[] for _ in eachindex(ax)]   # deferred pairs touching each vertex
        for (e, (k, l, _)) in enumerate(deferred)
            push!(inc[k-o], e)
            push!(inc[l-o], e)
        end
        done = falses(length(deferred))
        queue = collect(eachindex(deferred))
        qi = firstindex(queue)
        while qi <= lastindex(queue)
            e = queue[qi]
            qi += 1
            done[e] && continue
            k, l, v = deferred[e]
            ak, al = a[k], a[l]
            if iszero(ak) && iszero(al)
                continue
            elseif iszero(al)
                a[l] = v / ak
                done[e] = true
                append!(queue, inc[l-o])
            elseif iszero(ak)
                a[k] = v / al
                done[e] = true
                append!(queue, inc[k-o])
            else
                aprod = ak * al
                if aprod < v
                    s = sqrt(v / aprod)
                    a[k] *= s; a[l] *= s
                end
                done[e] = true
            end
        end
        # Split components that no diagonal scale reached, preserving order.
        for (e, (k, l, v)) in enumerate(deferred)
            done[e] && continue
            ak, al = a[k], a[l]
            if iszero(ak) && iszero(al)
                a[k] = a[l] = sqrt(v)
            elseif iszero(ak)
                a[k] = v / al
            elseif iszero(al)
                a[l] = v / ak
            else
                aprod = ak * al
                if aprod < v
                    s = sqrt(v / aprod)
                    a[k] *= s; a[l] *= s
                end
            end
        end
    end

    return a
end

# Apply the smallest uniform inflation that covers `A`. This preserves the
# starting point's shape and works in log space to avoid overflow.
function inflate_feasible!(a::AbstractVector{T}, A::AbstractMatrix) where T
    la = map(log, a)
    tref = Ref(zero(T))
    foreach_support_sym(A) do i, j, v
        tref[] = max(tref[], (log(T(v)) - la[i] - la[j]) / 2)
    end
    t = tref[]
    # A supported row with zero scale gives la = -Inf, hence t = +Inf.
    isfinite(t) ||
        throw(ArgumentError("inflate_feasible! requires a start with positive scale on every supported row"))
    iszero(t) && return a
    for i in eachindex(a)
        iszero(a[i]) || (a[i] = exp(la[i] + t))
    end
    return a
end

# Asymmetric counterpart: multiply every scale of `a` and `b` by the same factor
# until `a[i]*b[j] >= |A[i,j]|` for every entry visited by `foreach_support`.
# Requires a start with strictly positive scale on every supported row and column.
# The shift is covariant under an independent row/column rescaling `D_r*A*D_c`,
# and is accumulated in the log domain for the reasons given in the symmetric method.
function inflate_feasible!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    T = float(promote_type(eltype(a), eltype(b)))
    la, lb = map(log, a), map(log, b)
    tref = Ref(zero(T))
    foreach_support(A) do i, j, v
        tref[] = max(tref[], (log(T(v)) - la[i] - lb[j]) / 2)
    end
    t = tref[]
    # A supported row or column with zero scale gives la (or lb) = -Inf, hence t = +Inf.
    isfinite(t) ||
        throw(ArgumentError("inflate_feasible! requires a start with positive scale on every supported row/column"))
    iszero(t) && return a, b
    for i in eachindex(a)
        iszero(a[i]) || (a[i] = exp(la[i] + t))
    end
    for j in eachindex(b)
        iszero(b[j]) || (b[j] = exp(lb[j] + t))
    end
    return a, b
end

# Asymmetric uniform inflation over a flattened support; the shift convention
# is that of the matrix method.
function inflate_feasible!(a::AbstractVector, b::AbstractVector, sup::FlatSupport)
    T = float(promote_type(eltype(a), eltype(b)))
    is, js, lv = sup.is, sup.js, sup.lv
    la, lb = map(log, a), map(log, b)
    t = zero(T)
    for k in eachindex(is, js, lv)
        u = (lv[k] - la[is[k]] - lb[js[k]]) / 2
        t = ifelse(u > t, u, t)
    end
    # A supported row or column with zero scale gives la (or lb) = -Inf, hence t = +Inf.
    isfinite(t) ||
        throw(ArgumentError("inflate_feasible! requires a start with positive scale on every supported row/column"))
    iszero(t) && return a, b
    for i in eachindex(a)
        iszero(a[i]) || (a[i] = exp(la[i] + t))
    end
    for j in eachindex(b)
        iszero(b[j]) || (b[j] = exp(lb[j] + t))
    end
    return a, b
end
