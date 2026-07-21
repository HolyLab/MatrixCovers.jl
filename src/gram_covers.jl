# Symmetric covers of a (weighted) Gram matrix `A'*W*A`, computed directly from
# an asymmetric cover of `A` — without ever forming the Gram matrix.

# ============================================================
# Public interface
# ============================================================

"""
    s = gramcover(a, b, A)
    s = gramcover(a, b, A, w::AbstractVector)
    s = gramcover(a, b, A, W::AbstractMatrix)
    s = gramcover(a, b, sc::SupportComponents)
    s = gramcover(a, b, sc::SupportComponents, w::AbstractVector)
    s = gramcover(a, b, sc::SupportComponents, W::AbstractMatrix)

Given an asymmetric cover `a[i]*b[j] >= abs(A[i,j])` — from [`cover`](@ref),
[`cover_min`](@ref), or any other solver producing such a pair — return a
symmetric cover `s` of a (weighted) Gram matrix of `A`, without forming that
Gram matrix: `s[j]*s[k] >= abs(G[j,k])` for every `j`, `k`, where `G = A'*A`
for the two-argument form, `G = A'*Diagonal(w)*A` for the vector-weighted
form, and `G = A'*W*A` for the general form. Only `abs.(W)` enters the bound,
so `W` need not be positive definite, positive semidefinite, or even
symmetric; passing `W::Diagonal` is equivalent to passing `W.diag` as `w`.

`(a, b)` covering `A` is a precondition, not verified here — use
[`iscover`](@ref)`(a, b, A)` to check it beforehand. `gramcover` composes with
any asymmetric cover this package produces: [`cover`](@ref), [`cover_min`](@ref),
[`soft_cover`](@ref), and their mutating and `_min` forms.

Only the connected components of `A`'s support enter the two- and
vector-weighted forms, so a caller holding an
[`support_components`](@ref)`(A)` result may pass it in place of `A`; the
matrix forms are exactly that call followed by the `sc` form. `sc.rowax` and
`sc.colax` then play the roles of `axes(A, 1)` and `axes(A, 2)` in the axis
requirements on `a`, `b`, and `w`/`W`.

# Extended help

For `G[j,k] = Σ_{i,i'} A[i,j]*W[i,i']*A[i',k]`, the triangle inequality against
`a[i]*b[j] >= abs(A[i,j])` gives
`abs(G[j,k]) <= (Σ_{i,i'} a[i]*abs(W[i,i'])*a[i']) * b[j]*b[k]`.
Partitioning the rows and columns of `A` into the connected components of its
bipartite support graph, columns in different components share no supported
row, so for `W` diagonal the sum needed is exactly the one over the rows of
`j`'s own component:

    s[j] = sqrt(Σ_{i ∈ rows(comp(j))} abs(w[i])*a[i]^2) * b[j]

(the unweighted form is this with `w[i] = 1`). A nonzero off-diagonal
`W[i,i']` can couple rows from two different components; components joined by a
chain of such couplings form a group. Within a group, writing
`M[p,q] = Σ_{i ∈ rows(p), i' ∈ rows(q)} a[i]*abs(W[i,i'])*a[i']` for the block
sum over components `p` and `q`, `abs(G[j,k]) <= M[p,q]*b[j]*b[k]` for `j ∈ p`
and `k ∈ q`. Take `M` symmetric, as it is whenever `abs.(W)` is; the general
case needs one substitution, made in the remark below. Then

    s[j] = sqrt(Σ_q M[p,q] * sqrt(M[p,p]/M[q,q])) * b[j],  j ∈ p

meets every one of those bounds: the `q` term of the sum for `s[j]` and the `p`
term of the one for `s[k]` already multiply to `M[p,q]*M[q,p] = M[p,q]^2`, and
no term is negative. For a component that no `W` entry couples to another, the
group is a single component and this reduces to the diagonal-`W` formula above.
`G[j,k]` is exactly zero across distinct groups, and columns with no support get
`s[j] = 0`.

Rescaling `a -> γ*a`, `b -> b/γ` within any support component — independently
per component — leaves `s` unchanged. Unlike `b` alone, `s` is therefore safe to
use as an absolute scale, e.g. a Levenberg-Marquardt damping term
`λ*Diagonal(s.^2)`: a caller should maintain the dimensionless `λ` against `s`,
not against a quantity that depends on which gauge the cover solver happened to
return. What pins the relative scale of coupled components is the ratio
`M[p,p]/M[q,q]`, extended to a component whose own block vanishes (`abs.(W)` zero
throughout it, though `W` couples it to a sibling) by propagating along the
coupling. A group in which *every* component's own block vanishes is the one
exception, and it is a genuine degeneracy rather than a shortcoming of the
formula: the gauge acts on the surviving off-diagonal data as
`M[p,q] -> γ[p]*γ[q]*M[p,q]`, which fixes each *product* `s[j]*s[k]` across two
components but nothing about how it divides between them. Such a group falls back
to the uniform total `sqrt(Σ_{p,q} M[p,q])`, which covers, but there `s` depends
on the factorization and not on the products alone.

With more than one component and no coupling between them, `s[j] <= norm(a)*b[j]`,
strictly tighter whenever another component carries weight — the naive global
bound obtained by ignoring the block structure entirely. Coupled components admit
no such uniform comparison: fixing the gauge redistributes tightness among them,
resulting in an `s` that depends only on the products at the cost of individual entries
that a gauge-dependent global sum can beat.

When `abs.(W)` is not symmetric, neither is `G`, and since `s[j]*s[k]` is a
single number bounding both `abs(G[j,k])` and `abs(G[k,j])`, `M[p,q]` is
replaced throughout by `max(M[p,q], M[q,p])`; nothing else changes, the gauge
included, since that replacement is itself symmetric.

When a positive-semidefinite `W` is available only as an operator — `W[i,i]`
readable, `W[i,i']` for `i != i'` not — `abs(W[i,i']) <= sqrt(W[i,i]*W[i',i'])`
yields the looser diagonal-only bound
`s[j] = (Σ_{i ∈ rows(comp(j))} sqrt(W[i,i])*a[i]) * b[j]`, computable by hand
from `diag(W)`. The methods here always compute the tighter entrywise form
above, which requires `W`'s entries.

The accumulation feeding each `sqrt` is inflated to guarantee coverage despite
naive-summation roundoff, without ever forming `G` to check it.

See also: [`gramcover!`](@ref), [`symcover`](@ref), [`cover`](@ref), [`iscover`](@ref).

# Examples

```jldoctest
julia> J = [4 1; 1 3];

julia> a, b = cover(J);

julia> s = gramcover(a, b, J);

julia> all(s * s' .>= abs.(J' * J))
true

julia> w = [1.0, -2.0];

julia> sw = gramcover(a, b, J, w);   # covers J'*Diagonal(w)*J

julia> all(sw * sw' .>= abs.(J' * (w .* J)))
true
```
"""
gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix) =
    gramcover(a, b, support_components(A))

gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, w::AbstractVector) =
    gramcover(a, b, support_components(A), w)

gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::Diagonal) =
    gramcover(a, b, A, W.diag)

gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::AbstractMatrix) =
    gramcover(a, b, support_components(A), W)

function gramcover(a::AbstractVector, b::AbstractVector, sc::SupportComponents)
    T = _gc_eltype(a, b)
    s = similar(Array{T}, sc.colax)
    return gramcover!(s, a, b, sc)
end

function gramcover(a::AbstractVector, b::AbstractVector, sc::SupportComponents, w::AbstractVector)
    T = _gc_eltype(a, b, w)
    s = similar(Array{T}, sc.colax)
    return gramcover!(s, a, b, sc, w)
end

gramcover(a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::Diagonal) =
    gramcover(a, b, sc, W.diag)

function gramcover(a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::AbstractMatrix)
    T = _gc_eltype(a, b, W)
    s = similar(Array{T}, sc.colax)
    return gramcover!(s, a, b, sc, W)
end

"""
    s = gramcover!(s, a, b, A)
    s = gramcover!(s, a, b, A, w::AbstractVector)
    s = gramcover!(s, a, b, A, W::AbstractMatrix)
    s = gramcover!(s, a, b, sc::SupportComponents)
    s = gramcover!(s, a, b, sc::SupportComponents, w::AbstractVector)
    s = gramcover!(s, a, b, sc::SupportComponents, W::AbstractMatrix)

Mutating counterpart of [`gramcover`](@ref): writes the symmetric cover of the
(weighted) Gram matrix into `s` and returns it, rather than allocating a new
vector. `eachindex(s)` must match `axes(A, 2)` — `sc.colax` for the
[`SupportComponents`](@ref) forms — in addition to the axis requirements
[`gramcover`](@ref) places on `a`, `b`, and `w`/`W`.

See also: [`gramcover`](@ref).
"""
gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix) =
    gramcover!(s, a, b, support_components(A))

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, w::AbstractVector) =
    gramcover!(s, a, b, support_components(A), w)

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::Diagonal) =
    gramcover!(s, a, b, A, W.diag)

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::AbstractMatrix) =
    gramcover!(s, a, b, support_components(A), W)

function gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, sc::SupportComponents)
    _check_gramcover_ab(a, b, sc)
    _check_gramcover_s(s, sc)
    m = zeros(typeof(_gc_term(a)), ncomponents(sc))
    n = zeros(Int, ncomponents(sc))
    for i in sc.rowax
        c = rowcomponent(sc, i)
        iszero(c) && continue
        m[c] += a[i] * a[i]
        n[c] += 1
    end
    oc = first(sc.colax) - 1
    return _write_gramcover!(s, b, sc.colcomp, oc, m, n)
end

function gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, sc::SupportComponents, w::AbstractVector)
    _check_gramcover_ab(a, b, sc)
    _check_gramcover_s(s, sc)
    eachindex(w) == sc.rowax ||
        throw(DimensionMismatch("`w` holds one weight per support row: eachindex(w) must be $(sc.rowax), got $(eachindex(w))"))
    m = zeros(typeof(_gc_term(a, w)), ncomponents(sc))
    n = zeros(Int, ncomponents(sc))
    for i in sc.rowax
        c = rowcomponent(sc, i)
        iszero(c) && continue
        m[c] += abs(w[i]) * a[i] * a[i]
        n[c] += 1
    end
    oc = first(sc.colax) - 1
    return _write_gramcover!(s, b, sc.colcomp, oc, m, n)
end

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::Diagonal) =
    gramcover!(s, a, b, sc, W.diag)

function gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::AbstractMatrix)
    _check_gramcover_ab(a, b, sc)
    _check_gramcover_s(s, sc)
    axes(W) == (sc.rowax, sc.rowax) ||
        throw(DimensionMismatch("`W` couples support rows, so it must be square on the row axis: axes(W) must be $((sc.rowax, sc.rowax)), got $(axes(W))"))
    ncomp = ncomponents(sc)

    # Union-find over the component ids of `sc`: merge two of them whenever a
    # nonzero `W[i,i']` couples a supported row of one to a supported row of the
    # other, a coupling `A`'s own support graph does not carry but the product
    # `A'*W*A` does. Rows with no support (component id 0) contribute nothing to
    # `A'*W*A` regardless of `W`, so they are skipped.
    parent = collect(1:ncomp)
    function find(p)
        while parent[p] != p
            parent[p] = parent[parent[p]]   # path halving
            p = parent[p]
        end
        return p
    end
    merged = false
    for i in sc.rowax
        ci = rowcomponent(sc, i)
        iszero(ci) && continue
        for ip in sc.rowax
            cip = rowcomponent(sc, ip)
            (iszero(cip) || ci == cip) && continue
            iszero(abs(W[i, ip])) && continue
            ri, rip = find(ci), find(cip)
            ri == rip && continue
            parent[ri] = rip
            merged = true
        end
    end

    # With every component still its own group, each needs only its own block sum,
    # and the group bookkeeping below would allocate per component to say so.
    if !merged
        m = zeros(typeof(_gc_term(a, W)), ncomp)
        n = zeros(Int, ncomp)
        for i in sc.rowax
            ci = rowcomponent(sc, i)
            iszero(ci) && continue
            for ip in sc.rowax
                rowcomponent(sc, ip) == ci || continue
                m[ci] += a[i] * abs(W[i, ip]) * a[ip]
                n[ci] += 1
            end
        end
        return _write_gramcover!(s, b, sc.colcomp, first(sc.colax) - 1, m, n)
    end

    # Components merged into a common root, listed at that root; `local_idx` is a
    # component's position within its group, indexing the block sums below.
    members = [Int[] for _ in 1:ncomp]
    for c in 1:ncomp
        push!(members[find(c)], c)
    end
    local_idx = zeros(Int, ncomp)
    for r in 1:ncomp, (p, c) in enumerate(members[r])
        local_idx[c] = p
    end

    # Block sums `M[r][p,q] = Σ_{i ∈ comp p, i' ∈ comp q} a[i]*abs(W[i,i'])*a[i']`
    # over the components merged into root `r`; `nterm` counts the terms of each,
    # for the roundoff inflation in `_gc_group_scales!`.
    T = typeof(_gc_term(a, W))
    M = Vector{Matrix{T}}(undef, ncomp)
    nterm = Vector{Matrix{Int}}(undef, ncomp)
    for r in 1:ncomp
        k = length(members[r])
        iszero(k) && continue
        M[r] = zeros(T, k, k)
        nterm[r] = zeros(Int, k, k)
    end
    for i in sc.rowax
        ci = rowcomponent(sc, i)
        iszero(ci) && continue
        r = find(ci)
        p = local_idx[ci]
        for ip in sc.rowax
            cip = rowcomponent(sc, ip)
            iszero(cip) && continue
            find(cip) == r || continue
            q = local_idx[cip]
            M[r][p, q] += a[i] * abs(W[i, ip]) * a[ip]
            nterm[r][p, q] += 1
        end
    end

    sq = _gc_group_scales!(members, M, nterm)
    oc = first(sc.colax) - 1
    return _write_gramcover_sq!(s, b, sc.colcomp, oc, sq)
end

# ============================================================
# Internal helpers
# ============================================================

function _check_gramcover_ab(a::AbstractVector, b::AbstractVector, sc::SupportComponents)
    eachindex(a) == sc.rowax ||
        throw(DimensionMismatch("`a` holds one scale per support row: eachindex(a) must be $(sc.rowax), got $(eachindex(a))"))
    eachindex(b) == sc.colax ||
        throw(DimensionMismatch("`b` holds one scale per support column: eachindex(b) must be $(sc.colax), got $(eachindex(b))"))
    return nothing
end

function _check_gramcover_s(s::AbstractVector, sc::SupportComponents)
    eachindex(s) == sc.colax ||
        throw(DimensionMismatch("`s` holds one Gram scale per support column: eachindex(s) must be $(sc.colax), got $(eachindex(s))"))
    return nothing
end

# The per-term type of the sum accumulated into `m[c]`, computed from `zero`
# values so it tracks the units and precision of `a` (and `w`/`W`) without
# hardcoding a numeric type.
_gc_term(a::AbstractVector) = zero(eltype(a)) * zero(eltype(a))
_gc_term(a::AbstractVector, w::AbstractVector) = abs(zero(eltype(w))) * zero(eltype(a)) * zero(eltype(a))
_gc_term(a::AbstractVector, W::AbstractMatrix) = zero(eltype(a)) * abs(zero(eltype(W))) * zero(eltype(a))

# The element type `gramcover` allocates `s` at: the type of `sqrt` of a
# `_gc_term` times a `b`-scale, matching what `_write_gramcover!` actually computes.
_gc_eltype(a, b, args...) = typeof(sqrt(_gc_term(a, args...)) * zero(eltype(b)))

# Turn per-component sums `m[c]`, each accumulated from `n[c]` nonnegative terms,
# into per-component scales: the uncoupled case, where component `c`'s cover is
# just `sqrt(m[c])`.
#
# Naive summation of `n` nonnegative terms computes `fl(Σ) >= Σ/(1+γ)` with
# `γ = n*ulp/(1-n*ulp)`, `ulp = eps(scalarT)/2`; inflating `sqrt(m[c])` by
# `1 + (n[c]+3)*eps(scalarT)` absorbs that shortfall together with the roundoff of
# the `sqrt` itself and of the multiply by `b[j]`, so the cover holds despite
# `A'*W*A` never being formed to check it.
function _write_gramcover!(s::AbstractVector, b::AbstractVector, colcomp::Vector{Int}, oc, m::AbstractVector, n::AbstractVector{Int})
    scalarT = scalar_type(eltype(m))
    sq = [sqrt(m[c]) * (1 + (n[c] + 3) * eps(scalarT)) for c in eachindex(m)]
    return _write_gramcover_sq!(s, b, colcomp, oc, sq)
end

# Per-component scales `sq[c]` from the block sums `M[r][p,q]` of each merged
# group: `s[j] = sq[c]*b[j]` covers `A'*W*A` iff `sq[p]*sq[q] >= max(M[p,q], M[q,p])`
# for every pair of components `p`, `q` in one group (the `max` because `W` need
# not be symmetric, so the `(j,k)` and `(k,j)` entries of the Gram matrix are
# bounded by different block sums).
#
# Writing `d[p] = M[p,p]`, the choice
#
#     sq[p] = sqrt(Σ_q max(M[p,q], M[q,p]) * sqrt(d[p]/d[q]))
#
# satisfies that: the `q` term of `sq[p]^2` and the `p` term of `sq[q]^2` alone
# multiply to `max(M[p,q], M[q,p])^2`, and every term is nonnegative.
#
# The proof needs only that `d` be strictly positive, so `d` is purely a choice of
# gauge, and it is the choice that makes `s` independent of the gauge `a -> γ*a`,
# `b -> b/γ` applied independently per component. Under that gauge
# `M[p,q] -> γ[p]*γ[q]*M[p,q]`, so any `d` with `d[p] -> γ[p]^2*d[p]` makes each
# term `M[p,q]*sqrt(d[p]/d[q])` scale by `γ[p]^2`, leaving `sq[p]*b[j]` unchanged.
# A uniform group total `sqrt(Σ_{p,q} M[p,q])`, which mixes blocks that scale by
# different powers of `γ`, does not have that property.
#
# `d[p] = M[p,p]` transforms that way, and is the choice wherever it is nonzero.
# Where it is zero — `abs.(W)` vanishing throughout a component even though `W`
# couples it to a sibling — `d[p] = M[p,q]^2/d[q]` for an already-assigned
# neighbor `q` transforms the same way, and `M[p,q] > 0` for every coupled pair,
# so propagating outward from the nonzero diagonals covers the whole group.
#
# A group whose diagonal vanishes entirely leaves no pivot to propagate from, and
# no choice of `d` would help: the gauge sends `M[p,q] -> γ[p]*γ[q]*M[p,q]`, so
# with only off-diagonal blocks surviving it fixes the products `sq[p]*sq[q]` but
# not the split between them. That group falls back to the uniform total.
function _gc_group_scales!(members::Vector{Vector{Int}}, M::Vector{<:Matrix}, nterm::Vector{Matrix{Int}})
    T = eltype(eltype(M))
    scalarT = scalar_type(T)
    sq = [sqrt(zero(T)) for _ in 1:length(members)]
    for r in eachindex(members)
        mems = members[r]
        isempty(mems) && continue
        Mr, nr = M[r], nterm[r]
        k = length(mems)
        if k == 1
            sq[mems[1]] = sqrt(Mr[1, 1]) * (1 + (nr[1, 1] + 3) * eps(scalarT))
            continue
        end
        Ms = [max(Mr[p, q], Mr[q, p]) for p in 1:k, q in 1:k]   # `W` need not be symmetric
        d = _gc_gauge(Ms)
        if d === nothing
            v = sqrt(sum(Mr)) * (1 + (sum(nr) + 3) * eps(scalarT))
            for p in 1:k
                sq[mems[p]] = v
            end
            continue
        end
        for p in 1:k
            acc = zero(T)
            nt = 0
            for q in 1:k
                acc += Ms[p, q] * sqrt(d[p] / d[q])
                nt += max(nr[p, q], nr[q, p])
            end
            # `nt` terms of naive summation, plus a division and a `sqrt` per term,
            # plus the outer `sqrt` and the multiply by `b[j]`.
            sq[mems[p]] = sqrt(acc) * (1 + (nt + 3k + 3) * eps(scalarT))
        end
    end
    return sq
end

# Strictly positive gauge `d` for one group's symmetrized block sums `Ms`: the
# diagonal where it is nonzero, propagated as `Ms[p,q]^2/d[q]` from an assigned
# neighbor `q` where it is not. Returns `nothing` if the diagonal is entirely zero.
# The propagation takes the lowest-indexed assigned neighbor, so `d` depends on
# `Ms` and the component ordering alone — never on the gauge it is fixing.
function _gc_gauge(Ms::Matrix)
    k = size(Ms, 1)
    d = [Ms[p, p] for p in 1:k]
    any(!iszero, d) || return nothing
    while any(iszero, d)
        progress = false
        for p in 1:k
            iszero(d[p]) || continue
            q = findfirst(q -> !iszero(d[q]) && !iszero(Ms[p, q]), 1:k)
            q === nothing && continue
            d[p] = Ms[p, q] * Ms[p, q] / d[q]
            progress = true
        end
        # Every pair the union-find merged has `Ms[p,q] > 0`, so the group is
        # connected and a sweep that assigns nothing cannot happen; guard anyway
        # rather than spin.
        progress || error("gramcover: group with $(count(iszero, d)) unreachable component(s); please report this with the inputs")
    end
    return d
end

# Write `s[j] = sq[c]*b[j]` for `j` in component `c` (`colcomp[j - oc]`), and
# `s[j] = 0` for a column with no support. `sq` carries different units than the
# sums it came from (e.g. sums in `s` give `sq` in `s^(1/2)`), so its element type
# is inferred from the computation rather than taken from those sums.
function _write_gramcover_sq!(s::AbstractVector{T}, b::AbstractVector, colcomp::Vector{Int}, oc, sq::AbstractVector) where T
    for j in eachindex(s)
        c = colcomp[j-oc]
        s[j] = iszero(c) ? zero(T) : sq[c] * b[j]
    end
    return s
end
