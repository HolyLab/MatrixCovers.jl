# Symmetric covers of a (weighted) Gram matrix `A'*W*A`, computed directly from
# an asymmetric cover of `A` — without ever forming the Gram matrix.

# ============================================================
# Public interface
# ============================================================

"""
    s = gramcover(a, b, A)
    s = gramcover(a, b, A, w::AbstractVector)
    s = gramcover(a, b, A, W::AbstractMatrix)

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
`W[i,i']` can couple rows from two different components, which are merged
(via a union-find over the components) before the analogous sum is taken over
each merged component `c`:

    s[j] = sqrt(Σ_{i,i' ∈ rows(c)} a[i]*abs(W[i,i'])*a[i']) * b[j],  j ∈ c

Every term is nonnegative, so the merged-component sum dominates every
sub-block sum it contains, and `G[j,k]` is exactly zero across distinct merged
components. Columns with no support get `s[j] = 0`.

Rescaling `a -> γ*a`, `b -> b/γ` within any support component leaves `s`
unchanged — unlike `b` alone, `s` is safe to use as an absolute scale, e.g. a
Levenberg-Marquardt damping term `λ*Diagonal(s.^2)`: a caller should maintain
the dimensionless `λ` against `s`, not against a quantity that depends on
which gauge the cover solver happened to return.

With more than one component, `s[j] <= norm(a)*b[j]`, strictly tighter
whenever another component carries weight — the naive global bound obtained
by ignoring the block structure entirely.

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
function gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    T = _gc_eltype(a, b)
    s = similar(Array{T}, axes(A, 2))
    return gramcover!(s, a, b, A)
end

function gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, w::AbstractVector)
    T = _gc_eltype(a, b, w)
    s = similar(Array{T}, axes(A, 2))
    return gramcover!(s, a, b, A, w)
end

gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::Diagonal) =
    gramcover(a, b, A, W.diag)

function gramcover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::AbstractMatrix)
    T = _gc_eltype(a, b, W)
    s = similar(Array{T}, axes(A, 2))
    return gramcover!(s, a, b, A, W)
end

"""
    s = gramcover!(s, a, b, A)
    s = gramcover!(s, a, b, A, w::AbstractVector)
    s = gramcover!(s, a, b, A, W::AbstractMatrix)

Mutating counterpart of [`gramcover`](@ref): writes the symmetric cover of the
(weighted) Gram matrix into `s` and returns it, rather than allocating a new
vector. `eachindex(s)` must match `axes(A, 2)`, in addition to the axis
requirements [`gramcover`](@ref) places on `a`, `b`, and `w`/`W`.

See also: [`gramcover`](@ref).
"""
function gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    _check_gramcover_ab(a, b, A)
    _check_gramcover_s(s, A)
    rowcomp, colcomp, ncomp = _support_components(A)
    m = zeros(typeof(_gc_term(a)), ncomp)
    n = zeros(Int, ncomp)
    or = first(axes(A, 1)) - 1
    for i in axes(A, 1)
        c = rowcomp[i-or]
        iszero(c) && continue
        m[c] += a[i] * a[i]
        n[c] += 1
    end
    oc = first(axes(A, 2)) - 1
    return _write_gramcover!(s, b, colcomp, oc, m, n)
end

function gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, w::AbstractVector)
    _check_gramcover_ab(a, b, A)
    _check_gramcover_s(s, A)
    eachindex(w) == axes(A, 1) ||
        throw(DimensionMismatch("indices of `w` must match row-indexing of `A`, got eachindex(w)=$(eachindex(w)), axes(A, 1)=$(axes(A, 1))"))
    rowcomp, colcomp, ncomp = _support_components(A)
    m = zeros(typeof(_gc_term(a, w)), ncomp)
    n = zeros(Int, ncomp)
    or = first(axes(A, 1)) - 1
    for i in axes(A, 1)
        c = rowcomp[i-or]
        iszero(c) && continue
        m[c] += abs(w[i]) * a[i] * a[i]
        n[c] += 1
    end
    oc = first(axes(A, 2)) - 1
    return _write_gramcover!(s, b, colcomp, oc, m, n)
end

gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::Diagonal) =
    gramcover!(s, a, b, A, W.diag)

function gramcover!(s::AbstractVector, a::AbstractVector, b::AbstractVector, A::AbstractMatrix, W::AbstractMatrix)
    _check_gramcover_ab(a, b, A)
    _check_gramcover_s(s, A)
    axes(W) == (axes(A, 1), axes(A, 1)) ||
        throw(DimensionMismatch("axes of `W` must equal (axes(A, 1), axes(A, 1)), got axes(W)=$(axes(W)), axes(A, 1)=$(axes(A, 1))"))
    rowcomp, colcomp, ncomp = _support_components(A)
    or = first(axes(A, 1)) - 1

    # Union-find over the ids `_support_components` assigned: merge two of them
    # whenever a nonzero `W[i,i']` couples a supported row of one to a supported
    # row of the other, a coupling `A`'s own support graph does not carry but the
    # product `A'*W*A` does. Rows with no support (rowcomp == 0) contribute nothing
    # to `A'*W*A` regardless of `W`, so they are skipped.
    parent = collect(1:ncomp)
    function find(p)
        while parent[p] != p
            parent[p] = parent[parent[p]]   # path halving
            p = parent[p]
        end
        return p
    end
    for i in axes(A, 1)
        ci = rowcomp[i-or]
        iszero(ci) && continue
        for ip in axes(A, 1)
            cip = rowcomp[ip-or]
            (iszero(cip) || ci == cip) && continue
            iszero(abs(W[i, ip])) && continue
            ri, rip = find(ci), find(cip)
            ri == rip || (parent[ri] = rip)
        end
    end

    # Accumulate `m[c] = Σ_{i,i' merged into c} a[i]*abs(W[i,i'])*a[i']` at each
    # merged component's root; `n[c]` tracks the term count for the roundoff
    # inflation below.
    m = zeros(typeof(_gc_term(a, W)), ncomp)
    n = zeros(Int, ncomp)
    for i in axes(A, 1)
        ci = rowcomp[i-or]
        iszero(ci) && continue
        ri = find(ci)
        for ip in axes(A, 1)
            cip = rowcomp[ip-or]
            iszero(cip) && continue
            find(cip) == ri || continue
            m[ri] += a[i] * abs(W[i, ip]) * a[ip]
            n[ri] += 1
        end
    end
    # `colcomp` reports the original (pre-merge) component id, so roll each
    # merged total down to every id that was merged into it.
    for c in 1:ncomp
        r = find(c)
        r == c && continue
        m[c] = m[r]
        n[c] = n[r]
    end

    oc = first(axes(A, 2)) - 1
    return _write_gramcover!(s, b, colcomp, oc, m, n)
end

# ============================================================
# Internal helpers
# ============================================================

function _check_gramcover_ab(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    eachindex(a) == axes(A, 1) ||
        throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$(axes(A, 1))"))
    eachindex(b) == axes(A, 2) ||
        throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(eachindex(b)), axes(A, 2)=$(axes(A, 2))"))
    return nothing
end

function _check_gramcover_s(s::AbstractVector, A::AbstractMatrix)
    eachindex(s) == axes(A, 2) ||
        throw(DimensionMismatch("indices of `s` must match column-indexing of `A`, got eachindex(s)=$(eachindex(s)), axes(A, 2)=$(axes(A, 2))"))
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

# Turn per-(merged-)component sums `m[c]`, each accumulated from `n[c]`
# nonnegative terms, into `s`: `s[j] = sqrt(m[c])*b[j]` for `j` in component
# `c` (`colcomp[j - oc]`), `s[j] = 0` for a column with no support.
function _write_gramcover!(s::AbstractVector{T}, b::AbstractVector, colcomp::Vector{Int}, oc, m::AbstractVector, n::AbstractVector{Int}) where T
    scalarT = scalar_type(eltype(m))
    # `sqrt(m[c])` carries different units than `m[c]` itself (e.g. `m[c]` in `s`
    # gives `sqrt(m[c])` in `s^(1/2)`), so `sq`'s element type is inferred from the
    # computation rather than taken from `similar(m)`.
    #
    # Naive summation of `n[c]` nonnegative terms computes `fl(Σ) >= Σ/(1+γ)` with
    # `γ = n*ulp/(1-n*ulp)`, `ulp = eps(scalarT)/2`; inflating `sqrt(m[c])` by
    # `1 + (n[c]+3)*eps(scalarT)` absorbs that shortfall together with the roundoff
    # of the `sqrt` itself and of the multiply by `b[j]` below, so the cover this
    # writes holds despite never forming `A'*W*A` to check it.
    sq = [sqrt(m[c]) * (1 + (n[c] + 3) * eps(scalarT)) for c in eachindex(m)]
    for j in eachindex(s)
        c = colcomp[j-oc]
        s[j] = iszero(c) ? zero(T) : sq[c] * b[j]
    end
    return s
end
