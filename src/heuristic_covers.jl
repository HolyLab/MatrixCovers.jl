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

The initialization is the AbsLog{2} unconstrained minimum (geometric mean of
nonzero entries per row). It is then boosted to feasibility by a greedy
max-deficit rule (the most-violated entries are covered first), and `maxiter`
iterations of the tightening algorithm (Algorithm 1 of the manuscript) are
applied.

`ϕ` names the penalty the caller would like the cover to do well on.  Currently,
the heuristic covers ignore `ϕ`, although this behavior may change in future versions.
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
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$ax"))
    unconstrained_min!(AbsLog{2}(), a, A)
    boost_feasible!(a, A)
    return tighten_cover!(a, A; kwargs...)
end

"""
    a, b = cover(ϕ, A; maxiter=3)
    a, b = cover(A; maxiter=3)

Given a matrix `A`, return vectors `a` and `b` such that `a[i] * b[j] >= abs(A[i, j])`
for all `i`, `j`. The initialization is the AbsLog{2}
unconstrained minimum (geometric mean of nonzero entries per row/column). It is
then boosted to feasibility by a greedy max-deficit rule (the most-violated
entries are covered first), and `maxiter` tightening iterations are applied.

Only the products `a[i] * b[j]` are determined by the problem: `a -> c*a`, `b -> b/c`
leaves every one of them unchanged. The split is fixed by the balance convention
`∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` (`nzaᵢ`, `nzbⱼ` = nonzero counts of row `i`,
column `j`), imposed within each connected component of the support (the gauge acts
independently on each), as it is throughout the package; see [`cover_min`](@ref).

`ϕ` names the penalty the caller would like the cover to do well on. Currently,
the heuristic covers ignore `ϕ`, although this behavior may change in future versions.
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
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$(axes(A, 1))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(eachindex(b)), axes(A, 2)=$(axes(A, 2))"))
    unconstrained_min!(AbsLog{2}(), a, b, A)
    boost_feasible!(a, b, A)
    tighten_cover!(a, b, A; kwargs...)
    # The boost and the tightening raise and shrink rows and columns independently, so they
    # leave the gauge wherever they happen to land it. Pin it, so the split reported between
    # `a` and `b` is the package's convention rather than a residue of the passes above.
    # The shift is exact in the log-scales but rounds in the products, which the tightening
    # has driven onto the coverage boundary; the uniform inflation restores exact coverage,
    # and preserves the balance just set because ∑ nzaᵢ = ∑ nzbⱼ.
    _balance_cover!(a, b, A)
    return inflate_feasible!(a, b, A)
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
# Shift `(a, b)` along the gauge `a -> c*a`, `b -> b/c`, which leaves every product
# `a[i]*b[j]` — and hence every objective and every coverage constraint — untouched,
# onto the balance convention `∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` that every asymmetric
# cover in the package reports its result in. Summing `log a[i]` over the support counts
# row `i` exactly `nzaᵢ` times, which is those weighted sums. Nothing else pins the gauge:
# the objective cannot see it, so without a convention the split between `a` and `b` would
# be an artifact of whichever pass last touched them.
#
# The gauge acts independently on each connected component of the bipartite
# support graph (`_support_components`), so the convention is imposed per
# component: within every component, the row-side and column-side weighted log
# sums agree to within the rounding described below. This makes the split a
# well-defined function of the support and the products — block-diagonal
# assembly commutes with balancing — rather than pinning only the global scalar
# and leaving the per-component splits to whichever solver internals ran last.
# Rows and columns with empty support belong to no component and are left
# untouched.
#
# The shift is rounded to a whole power of two before it is applied. Scaling `a`
# by `2^k` and `b` by `2^-k` is exact in binary floating point, so every product
# `a[i]*b[j]` is preserved bit for bit and no cover is perturbed into
# infeasibility by the act of pinning its gauge. The balance is therefore met to
# within a factor of `√2` rather than exactly — a bound on the residual
# imbalance, traded for exactness of the quantity that carries the meaning.
#
# Two points differing only by the gauge land on the same point here, so a refiner given
# either start cannot tell them apart. The uniform inflation to feasibility raises every
# supported scale of `a` and `b` alike, and within each component
# `∑ nzaᵢ = ∑ nzbⱼ = nnz`, so it preserves the balance it finds.
function _balance_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    T = float(promote_type(eltype(a), eltype(b)))
    rowcomp, colcomp, ncomp = _support_components(A)
    iszero(ncomp) && return a, b
    Lα = zeros(T, ncomp)
    Lβ = zeros(T, ncomp)
    nnz = zeros(Int, ncomp)
    or = first(axes(A, 1)) - 1
    oc = first(axes(A, 2)) - 1
    foreach_support(A) do i, j, v
        c = rowcomp[i-or]
        Lα[c] += log2(T(a[i]))
        Lβ[c] += log2(T(b[j]))
        nnz[c] += 1
    end
    # `Lα`, `Lβ` are log2 sums, so the shift is already an exponent: rounding it
    # to an integer is what makes the rescaling below exact.
    gamma = [exp2(round((Lβ[c] - Lα[c]) / (2 * nnz[c]))) for c in 1:ncomp]
    for i in eachindex(a)
        c = rowcomp[i-or]
        c == 0 && continue
        a[i] *= gamma[c]
    end
    for j in eachindex(b)
        c = colcomp[j-oc]
        c == 0 && continue
        b[j] /= gamma[c]
    end
    return a, b
end


# Compute the analytical minimizer of the unconstrained AbsLog{2} symmetric objective
#   ∑_{i,j: A[i,j]≠0} (log(a[i]*a[j]) - log|A[i,j]|)²
# Fills `a` in-place and returns nza[i] = number of nonzero entries in row i.
# For efficiency, uses a Sherman-Morrison approximation for the pattern of nonzeros. (It's exact when there are no zeros.)
# This is the "rank-1 solution" described in manuscript section 5.2.
function unconstrained_min!(::AbsLog{2}, a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a))"))
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

function unconstrained_min!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    T = float(promote_type(eltype(a), eltype(b)))
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, b, A)` requires row indices of `A` to match `a`, got axes(A, 1)=$(axes(A, 1)), axes(a)=$(axes(a))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, b, A)` requires column indices of `A` to match `b`, got axes(A, 2)=$(axes(A, 2)), axes(b)=$(axes(b))"))
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

# Feasible cover starting from the diagonal alone, resolved by
# `boost_feasible_seq!`. Unlike `boost_feasible!`, a zero entry of `a` going
# into that call means "not yet resolved", not "permanently unsupported" —
# every diagonal-zero row is deferred until an off-diagonal neighbor supplies
# a scale for it (see `boost_feasible_seq!`).
function init_feasible_diag!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`init_feasible_diag!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a)))"))
    # A diagonal entry the traversal skips is zero, which is the "not yet resolved"
    # value `boost_feasible_seq!` expects.
    fill!(a, zero(T))
    foreach_support_sym(A) do i, j, v
        i == j && (a[i] = sqrt(T(v)))
    end
    return boost_feasible_seq!(a, A)
end

# Shrink x by exp(lr/2) (lr = log of the cover-to-entry ratio for the tightest
# entry touching x). The direct quotient degenerates to exact zero when
# exp(lr/2) overflows or the division underflows; recomputing in log space
# recovers any representable result, and the floatmin clamp handles genuine
# underflow: x is supported (nonzero going in), and an exact zero would make
# every entry through it permanently uncoverable, whereas floatmin keeps it
# representable and, being the smallest normal positive magnitude, changes the
# resulting cover products negligibly.
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

# Adjoint/Transpose wrappers for tighten_cover!.
function tighten_cover!(a::AbstractVector, b::AbstractVector, A::Adjoint; kwargs...)
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end
function tighten_cover!(a::AbstractVector, b::AbstractVector, A::Transpose; kwargs...)
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end

# Feasibility boost by approximate greedy max-deficit. `entries` holds the
# support entries; `deficit(e)` returns the log-deficit z = log|A[i,j]| minus
# the log of the current cover product, > 0 iff violated; `apply!(e, z)` grows
# the scales so the entry becomes exactly covered. Deficits only shrink as
# scales grow, so entries only move to lower buckets: total work is
# O(#entries + moves), moves per entry bounded by the bucket count. Bucket
# edges are anchored at 0 in log-deficit (a scale-invariant quantity), so
# processing order — hence the result — is covariant under diagonal rescaling
# of A, up to within-bucket ties. Working in log-deficit keeps every quantity
# finite for finite nonzero entries and positive scales, however extreme the
# dynamic range.
const BOOST_BUCKET_WIDTH = log(2) / 4   # quality indistinguishable from exact greedy; only bucket count grows as w shrinks

# Buckets are a flat singly-linked bucket queue (as in bucket-queue Dijkstra),
# not one growable Vector{Int} per bucket: `head[b]` is the top-of-stack index
# into `entries`, `nxt[k]` the next index below it in whatever bucket k
# currently occupies. Push/pop are O(1) pointer updates with no reallocation,
# and `deficit(entries[k])` is cheap enough (a couple of array reads and a
# subtraction) that recomputing it on each of the two passes below costs less
# than caching it in a separate array would.
function bucket_boost!(deficit::F, apply!::G, entries, ::Type{T}) where {F,G,T}
    n = length(entries)
    zmax = zero(T)
    for entry in entries
        z = deficit(entry)
        z > zero(T) || continue
        zmax = max(zmax, z)
    end
    zmax > zero(T) || return
    w = T(BOOST_BUCKET_WIDTH)
    B = max(1, ceil(Int, zmax / w))
    bucketof(z) = clamp(ceil(Int, z / w), 1, B)
    head = zeros(Int, B)
    nxt = zeros(Int, n)
    for k in eachindex(entries)
        z = deficit(entries[k])
        z > zero(T) || continue
        b = bucketof(z)
        nxt[k] = head[b]
        head[b] = k
    end
    for b in B:-1:1
        k = head[b]
        while k != 0
            knext = nxt[k]   # save before a possible demotion overwrites nxt[k]
            e = entries[k]
            z = deficit(e)
            if z > zero(T)
                b2 = bucketof(z)
                if b2 < b
                    nxt[k] = head[b2]
                    head[b2] = k          # deficit shrank: demote, revisit later
                else
                    apply!(e, z)
                end
            end
            k = knext
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
    foreach_support_sym(A) do i, j, v
        z = log(T(v)) - la[i] - la[j]
        z > zero(T) && (nviol[] += 1)
    end
    entries = Vector{Tuple{IdxT,IdxT,T}}(undef, nviol[])
    k = Ref(0)
    foreach_support_sym(A) do i, j, v
        lv = log(T(v))
        z = lv - la[i] - la[j]
        if z > zero(T)
            (iszero(a[i]) || iszero(a[j])) &&
                throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row"))
            k[] += 1
            entries[k[]] = (i, j, lv)
        end
    end
    deficit((i, j, lv)) = lv - la[i] - la[j]
    function apply!((i, j, lv), z)
        h = z / 2
        la[i] += h; a[i] = exp(la[i])
        i == j || (la[j] += h; a[j] = exp(la[j]))
    end
    bucket_boost!(deficit, apply!, entries, T)
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
    foreach_support(A) do i, j, v
        z = log(T(v)) - la[i] - lb[j]
        z > zero(T) && (nviol[] += 1)
    end
    entries = Vector{Tuple{IdxA,IdxB,T}}(undef, nviol[])
    k = Ref(0)
    foreach_support(A) do i, j, v
        lv = log(T(v))
        z = lv - la[i] - lb[j]
        if z > zero(T)
            (iszero(a[i]) || iszero(b[j])) &&
                throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row/column"))
            k[] += 1
            entries[k[]] = (i, j, lv)
        end
    end
    deficit((i, j, lv)) = lv - la[i] - lb[j]
    function apply!((i, j, lv), z)
        h = z / 2
        la[i] += h; a[i] = exp(la[i])
        lb[j] += h; b[j] = exp(lb[j])
    end
    bucket_boost!(deficit, apply!, entries, T)
    return a, b
end

# Feasibility by sequential nearest-neighbor propagation with deferral,
# processing off-diagonal pairs in order of increasing offset
# j = 1, …, n-1 (each nonzero A[k, k+j] requires a[k]*a[k+j] ≥ |A[k, k+j]|).
# `a[k] == 0` means "not yet resolved" rather than "unsupported": entries with
# both endpoints still zero are deferred until a later offset (or another
# deferred entry) supplies a scale for one of them, and any left unresolved
# after every offset is processed are equal-split as a[k]=a[l]=√|A[k,l]|.
#
# When both a[k] and a[l] are already nonzero but a[k]*a[l] < |A[k,l]|, both
# are scaled by the square root of the ratio √(|A[k,l]|/(a[k]*a[l])). Equal
# scaling is of course ad-hoc; while it might be better to do something tuned
# to a particular penalty function, that would risk making the algorithm
# O(n^3) (we'd likely need to revisit earlier offsets), and earlier decisions
# might be reversed by later ones anyway. For something intended as an
# initialization, a heuristic guaranteed to be O(n^2) seems reasonable.
function boost_feasible_seq!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`boost_feasible_seq!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a)))"))
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

    # Resolve deferred constraints: re-scan until no more progress, then equal-split.
    while !isempty(deferred)
        changed = false
        filter!(deferred) do (k, l, v)
            ak, al = a[k], a[l]
            if !iszero(ak) && !iszero(al)
                aprod = ak * al
                if aprod < v
                    s = sqrt(v / aprod)
                    a[k] *= s; a[l] *= s
                end
            elseif !iszero(ak)
                a[l] = v / ak
            elseif !iszero(al)
                a[k] = v / al
            else
                return true   # still unresolvable; keep in list
            end
            changed = true
            return false      # resolved; drop from list
        end
        changed && continue
        # No progress: all remaining have both indices zero.
        # Process in order so earlier equal-splits can inform later ones in the same pass.
        for (k, l, v) in deferred
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
        break
    end

    return a
end

# Feasibility by the smallest uniform inflation: multiply every scale by the same
# factor until `a[i]*a[j] >= |A[i,j]|` for every entry visited by
# `foreach_support_sym`. Requires a start with strictly positive scale on every
# supported row (the geometric-mean init from `unconstrained_min!` guarantees this).
#
# Unlike `boost_feasible!`, which raises only the rows touching violated entries,
# this leaves the shape of the starting point untouched and moves it bodily to the
# feasibility boundary. The two reach different basins of the non-convex AbsLinear
# objective, which is why `soft_symcover` offers both as starts. The shift depends
# on `A` only through the log-deficits at the starting point, which are invariant
# under a diagonal rescaling `D*A*D`, so the result is scale-covariant. Growing the
# log-scales directly (rather than multiplying by `exp(t)`) stays finite even when
# `exp(t)` alone would overflow.
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
