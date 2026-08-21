# Symmetric covers of a (weighted) Gram matrix `A'*W*A`, computed directly from
# an asymmetric cover of `A` — without ever forming the Gram matrix.

# Default for the general-`W` methods.
const GRAMCOVER_DEGENERATE = :error

# ============================================================
# Public interface
# ============================================================

"""
    s = gramcover(a, b, A)
    s = gramcover(a, b, A, w::AbstractVector)
    s = gramcover(a, b, A, W::AbstractMatrix; degenerate=:$(GRAMCOVER_DEGENERATE))
    s = gramcover(a, b, sc::SupportComponents)
    s = gramcover(a, b, sc::SupportComponents, w::AbstractVector)
    s = gramcover(a, b, sc::SupportComponents, W::AbstractMatrix; degenerate=:$(GRAMCOVER_DEGENERATE))

Given an asymmetric cover `a[i]*b[j] >= abs(A[i,j])`, return a symmetric cover
`s` of a weighted Gram matrix without forming it: `s[j]*s[k] >= abs(G[j,k])`,
where `G` is `A'*A`, `A'*Diagonal(w)*A`, or `A'*W*A`. Only `abs.(W)` enters the
bound, so `W` need not be symmetric or positive semidefinite. Passing
`W::Diagonal` is equivalent to passing `W.diag`.

`(a, b)` must cover `A`; use [`iscover`](@ref)`(a, b, A)` to check it.

The methods accepting [`SupportComponents`](@ref) reuse a previous
[`support_components`](@ref)`(A)` computation. For these methods, `sc.rowax` and
`sc.colax` replace `axes(A, 1)` and `axes(A, 2)` in the axis requirements.

Equivalent componentwise rescalings of `(a, b)` produce the same `s`. Some
coupling patterns make this impossible; the general form then throws an
`ArgumentError`. Set `degenerate=:uniform` to return a gauge-dependent cover.

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
and `k ∈ q`. When `abs.(W)` is not symmetric neither is `G`, and since
`s[j]*s[k]` is a single number bounding both `abs(G[j,k])` and `abs(G[k,j])`, the
block sums enter only through their symmetrization
`Ms[p,q] = max(M[p,q], M[q,p])`. What remains is to divide each bound between its
two components: any `σ` with `σ[p]*σ[q] >= Ms[p,q]` for every `p`, `q` yields

    s[j] = σ[p]*b[j],  j ∈ p

This is a symmetric cover of the `k×k` matrix `Ms`, where `k` is the number of
components in the group. The implementation computes
[`symcover_min`](@ref)`(AbsLog{2}(), Ms)`. For a component that no `W` entry
couples to another, this reduces to the diagonal-`W` formula. Entries of `G`
across distinct groups vanish, and unsupported columns get `s[j] = 0`.

Rescaling `a -> γ*a`, `b -> b/γ` within each support component leaves `s`
unchanged. Under this rescaling, `Ms[p,q] -> γ[p]*γ[q]*Ms[p,q]` and the minimal
cover changes as `σ[p] -> γ[p]*σ[p]`; therefore `σ[p]*b[j]` is invariant. This
makes `s` suitable as an absolute scale, such as in a Levenberg-Marquardt term
`λ*Diagonal(s.^2)`.

The invariant cover does not exist when a connected part of `Ms`'s support graph
has an edge, no loop, and is bipartite. Scaling one color class by `t` and the
other by `1/t` leaves `Ms` unchanged but changes the individual `σ` values.
`gramcover` throws an `ArgumentError` in this case. With
`degenerate=:uniform`, it instead uses
`σ[p] = sqrt(Σ_{p,q} Ms[p,q])`, which depends on the gauge of `(a, b)`. A loop
or odd cycle removes this degeneracy.

For uncoupled components, `s[j] <= norm(a)*b[j]`, with a strict inequality when
another component carries weight. There is no uniform comparison for coupled
components because changing the gauge redistributes tightness among them.

When a positive-semidefinite `W` is available only as an operator — `W[i,i]`
readable, `W[i,i']` for `i != i'` not — `abs(W[i,i']) <= sqrt(W[i,i]*W[i',i'])`
yields the looser diagonal-only bound
`s[j] = (Σ_{i ∈ rows(comp(j))} sqrt(W[i,i])*a[i]) * b[j]`, computable by hand
from `diag(W)`. The methods here always compute the tighter entrywise form
above, which requires `W`'s entries.

The minimal-cover solve has size `k`, the number of support components coupled
by `W`. Constructing `Ms` also costs `O(k^2)` and usually dominates the solve.

Roundoff margins on the block sums and a final feasibility check preserve the
cover in floating-point arithmetic.

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

# Accept and validate `degenerate` consistently for every matrix weight.
gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::Diagonal; degenerate::Symbol=GRAMCOVER_DEGENERATE) =
    (_gc_check_degenerate(degenerate); gramcover(a, b, A, W.diag))

gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::AbstractMatrix; kwargs...) =
    gramcover(a, b, support_components(A), W; kwargs...)

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

gramcover(a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::Diagonal; degenerate::Symbol=GRAMCOVER_DEGENERATE) =
    (_gc_check_degenerate(degenerate); gramcover(a, b, sc, W.diag))

function gramcover(a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::AbstractMatrix; kwargs...)
    T = _gc_eltype(a, b, W)
    s = similar(Array{T}, sc.colax)
    return gramcover!(s, a, b, sc, W; kwargs...)
end

"""
    s = gramcover!(s, a, b, A)
    s = gramcover!(s, a, b, A, w::AbstractVector)
    s = gramcover!(s, a, b, A, W::AbstractMatrix; degenerate=:$(GRAMCOVER_DEGENERATE))
    s = gramcover!(s, a, b, sc::SupportComponents)
    s = gramcover!(s, a, b, sc::SupportComponents, w::AbstractVector)
    s = gramcover!(s, a, b, sc::SupportComponents, W::AbstractMatrix; degenerate=:$(GRAMCOVER_DEGENERATE))

Mutating counterpart of [`gramcover`](@ref): writes the symmetric cover of the
(weighted) Gram matrix into `s` and returns it, rather than allocating a new
vector. `eachindex(s)` must match `axes(A, 2)` — `sc.colax` for the
[`SupportComponents`](@ref) forms — in addition to the axis requirements
[`gramcover`](@ref) places on `a`, `b`, and `w`/`W`, and shares its
`degenerate` keyword.

See also: [`gramcover`](@ref).
"""
gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix) =
    gramcover!(s, a, b, support_components(A))

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, w::AbstractVector) =
    gramcover!(s, a, b, support_components(A), w)

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::Diagonal; degenerate::Symbol=GRAMCOVER_DEGENERATE) =
    (_gc_check_degenerate(degenerate); gramcover!(s, a, b, A, W.diag))

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::AbstractMatrix; kwargs...) =
    gramcover!(s, a, b, support_components(A), W; kwargs...)

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
        throw(DimensionMismatch("`w` holds one weight per support row: eachindex(w) must be $(string(sc.rowax)), got $(string(eachindex(w)))"))
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

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::Diagonal; degenerate::Symbol=GRAMCOVER_DEGENERATE) =
    (_gc_check_degenerate(degenerate); gramcover!(s, a, b, sc, W.diag))

function gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, sc::SupportComponents, W::AbstractMatrix; degenerate::Symbol=GRAMCOVER_DEGENERATE)
    _gc_check_degenerate(degenerate)
    _check_gramcover_ab(a, b, sc)
    _check_gramcover_s(s, sc)
    axes(W) == (sc.rowax, sc.rowax) ||
        throw(DimensionMismatch("`W` couples support rows, so it must be square on the row axis: axes(W) must be $(string((sc.rowax, sc.rowax))), got $(string(axes(W)))"))
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
    # for the roundoff inflation in `_gc_group_scales`.
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

    sq = _gc_group_scales(members, M, nterm, degenerate)
    oc = first(sc.colax) - 1
    return _write_gramcover_sq!(s, b, sc.colcomp, oc, sq)
end

# ============================================================
# Internal helpers
# ============================================================

function _check_gramcover_ab(a::AbstractVector, b::AbstractVector, sc::SupportComponents)
    eachindex(a) == sc.rowax ||
        throw(DimensionMismatch("`a` holds one scale per support row: eachindex(a) must be $(string(sc.rowax)), got $(string(eachindex(a)))"))
    eachindex(b) == sc.colax ||
        throw(DimensionMismatch("`b` holds one scale per support column: eachindex(b) must be $(string(sc.colax)), got $(string(eachindex(b)))"))
    return nothing
end

function _gc_check_degenerate(degenerate::Symbol)
    degenerate in (:error, :uniform) ||
        throw(ArgumentError("`degenerate` must be :error or :uniform, got :$degenerate"))
    return degenerate
end

function _check_gramcover_s(s::AbstractVector, sc::SupportComponents)
    eachindex(s) == sc.colax ||
        throw(DimensionMismatch("`s` holds one Gram scale per support column: eachindex(s) must be $(string(sc.colax)), got $(string(eachindex(s)))"))
    return nothing
end

# Infer the accumulator type while preserving units and precision.
_gc_term(a::AbstractVector) = zero(eltype(a)) * zero(eltype(a))
_gc_term(a::AbstractVector, w::AbstractVector) = abs(zero(eltype(w))) * zero(eltype(a)) * zero(eltype(a))
_gc_term(a::AbstractVector, W::AbstractMatrix) = zero(eltype(a)) * abs(zero(eltype(W))) * zero(eltype(a))

# Element type of the output scale.
_gc_eltype(a, b, args...) = typeof(sqrt(_gc_term(a, args...)) * zero(eltype(b)))

# Convert uncoupled component sums to scales.
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

# Compute a minimal symmetric cover of each symmetrized block matrix. Minimality
# makes the result invariant under componentwise rescaling of `(a, b)`.
function _gc_group_scales(members::Vector{Vector{Int}}, M::Vector{<:Matrix}, nterm::Vector{Matrix{Int}}, degenerate::Symbol)
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
        # Account for summation and multiplication roundoff in each block.
        Ms = [max(Mr[p, q], Mr[q, p]) * (1 + (max(nr[p, q], nr[q, p]) + 1) * eps(scalarT))
              for p in 1:k, q in 1:k]
        if _loopless_bipartite(Ms)
            degenerate === :uniform || throw(ArgumentError(
                "gramcover: $k coupled support components have a loopless bipartite support graph, so no gauge-invariant cover exists; pass `degenerate=:uniform` to allow a gauge-dependent cover"))
            v = sqrt(sum(Ms)) * (1 + (k * k + 3) * eps(scalarT))
            for p in 1:k
                sq[mems[p]] = v
            end
            continue
        end
        σ = _gc_inflate_to_cover(symcover_min(AbsLog{2}(), Ms), Ms, scalarT)
        for p in 1:k
            # The extra factor is for the multiply by `b[j]` in `_write_gramcover_sq!`.
            sq[mems[p]] = σ[p] * (1 + 3 * eps(scalarT))
        end
    end
    return sq
end

# Uniformly inflate `σ` until it covers `Ms` in floating-point arithmetic. The
# retry guards against a downward-rounded inflation factor.
function _gc_inflate_to_cover(σ::AbstractVector, Ms::AbstractMatrix, ::Type{scalarT}) where {scalarT}
    for _ in 1:8
        ρ = one(scalarT)
        for q in axes(Ms, 2), p in axes(Ms, 1)
            iszero(Ms[p, q]) && continue
            pq = σ[p] * σ[q]
            iszero(pq) &&
                throw(ErrorException("gramcover: zero scale on a coupled component; please report this with the inputs"))
            ρ = max(ρ, scalarT(Ms[p, q] / pq))
        end
        ρ <= 1 && return σ
        σ = σ .* (sqrt(ρ) * (1 + 4 * eps(scalarT)))
    end
    throw(ErrorException("gramcover: group cover did not reach feasibility; please report this with the inputs"))
end

# True when a nontrivial connected component of `Ms`'s support graph is
# loopless and bipartite.
#
# Zero-weight components are isolated and do not create a degeneracy.
function _loopless_bipartite(Ms::AbstractMatrix)
    k = size(Ms, 1)
    color = zeros(Int8, k)
    stack = Int[]
    for root in 1:k
        iszero(color[root]) || continue
        color[root] = 1
        push!(stack, root)
        hasedge = hasloop = false
        bipartite = true
        while !isempty(stack)
            p = pop!(stack)
            iszero(Ms[p, p]) || (hasloop = true)
            for q in 1:k
                (q == p || iszero(Ms[p, q])) && continue
                hasedge = true
                if iszero(color[q])
                    color[q] = -color[p]
                    push!(stack, q)
                elseif color[q] == color[p]
                    bipartite = false
                end
            end
        end
        hasedge && !hasloop && bipartite && return true
    end
    return false
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
