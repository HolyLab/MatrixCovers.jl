# Symmetric covers of a (weighted) Gram matrix `A'*W*A`, computed from `A` and
# `W` without forming the Gram matrix, and the cover of `A` implied by a fixed
# row scale.

# ============================================================
# Public interface
# ============================================================

"""
    s = gramcover(A)
    s = gramcover(A, w::AbstractVector)
    s = gramcover(A, W::Diagonal)
    s = gramcover(A, W::Cholesky)
    s = gramcover(A, W::AbstractMatrix; v=<symmetric cover of abs.(W)>)

Return a vector `s` whose outer product `s*s'` covers the Gram matrix
`G = A'*W*A`: `s[j]*s[k] >= abs(G[j,k])` for all `j`, `k`. `G` is never formed.
`gramcover(A)` covers `A'*A`, and passing `w` is equivalent to passing
`Diagonal(w)`. Columns of `A` with no stored nonzero get `s[j] = 0`.

No cover of `A` is required. Every method is scale-covariant: replacing `A` by
`D1*A*D2` and `W` by `inv(D1)*W*inv(D1)` (for positive diagonal `D1`, `D2`)
multiplies `s` by `diag(D2)`.

- **Diagonal weight** (`w`, `Diagonal(w)`, or none):
  `s[j] = sqrt(Σ_i abs(w[i]) * abs(A[i,j])^2)`. The weights may have any sign.
  When `w .>= 0` this is `sqrt(G[j,j])`, which is the minimal symmetric cover of
  `G`. One pass over the stored nonzeros of `A`.
- **`Cholesky` weight**: for `W = L*L'`, `s[j] = norm(L'*A[:,j]) = sqrt(G[j,j])`,
  the minimal symmetric cover of the positive semidefinite `G`. Passing the
  factorization is the caller's assertion that `W` is positive semidefinite.
  Cost is that of the dense product `L'*A`, O(m²n) for `A` of size m×n.
- **General weight** `W`: `s[j] = Σ_i v[i] * abs(A[i,j])`, where `v .>= 0`
  satisfies `abs(W[i,k]) <= v[i]*v[k]` for all `i`, `k`. `W` may be indefinite
  or asymmetric. This bound is looser than the other two: for diagonal `W` with
  `w .>= 0` it exceeds `sqrt(G[j,j])` by up to a factor `sqrt(n_j)`, where `n_j`
  is the number of nonzeros in column `j`. Prefer the `Diagonal` or `Cholesky`
  method whenever `W` has that structure. The default `v` is
  `symcover(max.(abs.(W), abs.(transpose(W))))`; a supplied `v` is checked for
  these conditions. Cost is that of computing `v` plus one pass over `A`.

`W` must be square with both axes equal to `axes(A, 1)`, and `s` has axes
`(axes(A, 2),)`. The bounds hold in floating-point arithmetic: `s` is inflated
by a relative `O(n_j * eps)` to absorb the rounding of the sums. For the
`Cholesky` method the bound is relative to the computed factor.

See also: [`gramcover!`](@ref), [`symcover`](@ref), [`cover`](@ref), [`iscover`](@ref).

# Examples

```jldoctest
julia> using LinearAlgebra

julia> A = [4.0 1.0 0.0; 1.0 3.0 2.0; 0.0 2.0 5.0; 1.0 0.0 1.0];

julia> w = [1.0, -2.0, 0.5, 3.0];   # mixed signs

julia> s = gramcover(A, w);

julia> all(s * s' .>= abs.(A' * Diagonal(w) * A))
true

julia> W = [2.0 1.0 0.0 0.0; 1.0 2.0 1.0 0.0; 0.0 1.0 2.0 1.0; 0.0 0.0 1.0 2.0];

julia> s = gramcover(A, cholesky(W));

julia> all(s * s' .>= abs.(A' * W * A))
true
```
"""
function gramcover(A::AbstractMatrix)
    s = similar(Array{_gc_eltype(A)}, axes(A, 2))
    return gramcover!(s, A)
end

function gramcover(A::AbstractMatrix, w::AbstractVector)
    s = similar(Array{_gc_eltype(A, w)}, axes(A, 2))
    return gramcover!(s, A, w)
end

gramcover(A::AbstractMatrix, W::Diagonal) = gramcover(A, W.diag)

function gramcover(A::AbstractMatrix, W::Cholesky)
    s = similar(Array{_gc_eltype(A, W)}, axes(A, 2))
    return gramcover!(s, A, W)
end

function gramcover(A::AbstractMatrix, W::AbstractMatrix; v::AbstractVector=_gc_default_v(A, W))
    s = similar(Array{_gc_eltype(A, W, v)}, axes(A, 2))
    return gramcover!(s, A, W; v)
end

"""
    s = gramcover!(s, A)
    s = gramcover!(s, A, w::AbstractVector)
    s = gramcover!(s, A, W::Diagonal)
    s = gramcover!(s, A, W::Cholesky)
    s = gramcover!(s, A, W::AbstractMatrix; v=<symmetric cover of abs.(W)>)

Mutating counterpart of [`gramcover`](@ref): writes the symmetric cover of the
(weighted) Gram matrix into `s` and returns it. `eachindex(s)` must equal
`axes(A, 2)`.

See also: [`gramcover`](@ref).
"""
function gramcover!(s::AbstractVector, A::AbstractMatrix)
    _check_gramcover_s(s, A)
    m, n = _gc_accumulators(A, _gc_term(A))
    foreach_support(A) do _, j, x
        m[j] += x * x
        n[j] += 1
    end
    return _write_sqrt!(s, m, n)
end

function gramcover!(s::AbstractVector, A::AbstractMatrix, w::AbstractVector)
    _check_gramcover_s(s, A)
    eachindex(w) == axes(A, 1) ||
        throw(DimensionMismatch("`w` holds one weight per row of `A`: eachindex(w) must be $(string(axes(A, 1))), got $(string(eachindex(w)))"))
    m, n = _gc_accumulators(A, _gc_term(A, w))
    foreach_support(A) do i, j, x
        m[j] += abs(w[i]) * x * x
        n[j] += 1
    end
    return _write_sqrt!(s, m, n)
end

gramcover!(s::AbstractVector, A::AbstractMatrix, W::Diagonal) = gramcover!(s, A, W.diag)

function gramcover!(s::AbstractVector, A::AbstractMatrix, W::Cholesky)
    _check_gramcover_s(s, A)
    # A factorization has one-based axes, so `A` must too along its rows.
    Base.OneTo(size(W, 1)) == axes(A, 1) ||
        throw(DimensionMismatch("`W` must be square on the rows of `A`: axes(A, 1) must be $(string(Base.OneTo(size(W, 1)))), got $(string(axes(A, 1)))"))
    # `U'*U == W`, so column `k` of `U*A` has squared norm `A[:,k]'*W*A[:,k]`.
    T = promote_type(eltype(W), float(eltype(A)))
    B = copyto!(Matrix{T}(undef, size(A)), A)
    lmul!(W.U, B)
    # Each entry of `B` is a length-`size(A, 1)` dot product; as for the sums in
    # `_write_sqrt!`, the factor absorbs their rounding, the norm's, and its own.
    scalarT = scalar_type(eltype(s))
    inflate = 1 + (size(A, 1) + 3) * eps(scalarT)
    for (k, j) in zip(axes(B, 2), axes(A, 2))
        s[j] = norm(view(B, :, k)) * inflate
    end
    return s
end

function gramcover!(s::AbstractVector, A::AbstractMatrix, W::AbstractMatrix; v::AbstractVector=_gc_default_v(A, W))
    _check_gramcover_s(s, A)
    _check_gramcover_W(W, A)
    eachindex(v) == axes(A, 1) ||
        throw(DimensionMismatch("`v` holds one scale per row of `A`: eachindex(v) must be $(string(axes(A, 1))), got $(string(eachindex(v)))"))
    for i in eachindex(v)
        v[i] >= zero(v[i]) ||
            throw(ArgumentError("`v` must be nonnegative, got v[$(string(i))] = $(string(v[i]))"))
    end
    iscover(v, v, W) ||
        throw(ArgumentError("`v` must satisfy abs(W[i,k]) <= v[i]*v[k] for all i, k"))
    m, n = _gc_accumulators(A, _gc_linterm(A, v))
    foreach_support(A) do i, j, x
        m[j] += v[i] * x
        n[j] += 1
    end
    # Inflate so the cover holds in floating point: naive summation of `n[j]`
    # nonnegative products falls short of the exact sum by at most a relative
    # `(n[j] + 1) * eps/2`, and the multiply by the factor adds one rounding.
    scalarT = scalar_type(eltype(m))
    for j in eachindex(s)
        s[j] = m[j] * (1 + (n[j] + 3) * eps(scalarT))
    end
    return s
end

"""
    a, b = cover(A, a::AbstractVector)

Return the tightest cover `(a, b)` of `A` with the row scale `a` held fixed:
`b[j] = max_i abs(A[i,j]) / a[i]`, the column maxima of `Diagonal(1 ./ a) * A`.
The returned `a` is the argument itself. Columns with no stored nonzero get
`b[j] = 0`.

`a` must be positive on every row that has a stored nonzero and nonnegative
elsewhere; `eachindex(a)` must equal `axes(A, 1)`. For each supported column,
some `i` attains `a[i]*b[j] ≈ abs(A[i,j])` (up to one rounding), and
`a[i]*b[j] >= abs(A[i,j])` holds exactly in floating point.

See also: [`cover`](@ref), [`gramcover`](@ref), [`iscover`](@ref).

# Examples

```jldoctest
julia> A = [1.0 2.0; 4.0 1.0];

julia> a, b = cover(A, [1.0, 2.0]);

julia> b
2-element Vector{Float64}:
 2.0
 2.0

julia> iscover(a, b, A)
true
```
"""
function cover(A::AbstractMatrix, a::AbstractVector)
    eachindex(a) == axes(A, 1) ||
        throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    for i in eachindex(a)
        a[i] >= zero(a[i]) ||
            throw(ArgumentError("the row scale `a` must be nonnegative, got a[$(string(i))] = $(string(a[i]))"))
    end
    Tb = typeof(abs(oneunit(eltype(A))) / oneunit(eltype(a)))
    b = similar(Array{Tb}, axes(A, 2))
    fill!(b, zero(Tb))
    scalarT = scalar_type(Tb)
    foreach_support(A) do i, j, x
        ai = a[i]
        ai > zero(ai) ||
            throw(ArgumentError("the row scale `a` must be positive on every row that has a stored nonzero, got a[$(string(i))] = $(string(ai)) with A[$(string(i)),$(string(j))] nonzero"))
        r = x / ai
        # One rounding in the division can leave `ai*r` just short of `x`; the
        # bump restores `ai*r >= x` in floating point.
        ai * r >= x || (r *= 1 + 4 * eps(scalarT))
        r > b[j] && (b[j] = r)
    end
    return a, b
end

# ============================================================
# Internal helpers
# ============================================================

function _check_gramcover_s(s::AbstractVector, A::AbstractMatrix)
    eachindex(s) == axes(A, 2) ||
        throw(DimensionMismatch("`s` holds one Gram scale per column of `A`: eachindex(s) must be $(string(axes(A, 2))), got $(string(eachindex(s)))"))
    return nothing
end

function _check_gramcover_W(W::AbstractMatrix, A::AbstractMatrix)
    axes(W) == (axes(A, 1), axes(A, 1)) ||
        throw(DimensionMismatch("`W` must be square on the rows of `A`: axes(W) must be $(string((axes(A, 1), axes(A, 1)))), got $(string(axes(W)))"))
    return nothing
end

# A symmetric cover of `abs.(W)` that also covers `abs.(transpose(W))`, so it
# bounds every entry of an asymmetric `W`.
function _gc_default_v(A::AbstractMatrix, W::AbstractMatrix)
    _check_gramcover_W(W, A)
    return symcover(max.(abs.(W), abs.(transpose(W))))
end

# Per-column accumulators for the sums and for the number of terms in each.
function _gc_accumulators(A::AbstractMatrix, term)
    m = similar(Array{typeof(term)}, axes(A, 2))
    fill!(m, term)
    n = similar(Array{Int}, axes(A, 2))
    fill!(n, 0)
    return m, n
end

# Accumulator types for the sums, preserving units and precision.
_gc_term(A::AbstractMatrix) = float(abs(zero(eltype(A))))^2
_gc_term(A::AbstractMatrix, w::AbstractVector) = abs(zero(eltype(w))) * _gc_term(A)
_gc_linterm(A::AbstractMatrix, v::AbstractVector) = zero(eltype(v)) * float(abs(zero(eltype(A))))

# Element type of the output scale.
_gc_eltype(A::AbstractMatrix) = typeof(sqrt(_gc_term(A)))
_gc_eltype(A::AbstractMatrix, w::AbstractVector) = typeof(sqrt(_gc_term(A, w)))
_gc_eltype(A::AbstractMatrix, W::Cholesky) = typeof(sqrt(abs(zero(eltype(W)))) * float(abs(zero(eltype(A)))))
_gc_eltype(A::AbstractMatrix, W::AbstractMatrix, v::AbstractVector) = typeof(_gc_linterm(A, v))

# Write `s[j] = sqrt(m[j])`, inflated so the cover holds in floating point without
# forming `A'*W*A`: naive summation of `n[j]` nonnegative terms, each formed with
# up to two roundings, falls short of the exact sum by at most a relative
# `(n[j] + 1) * eps/2`; the `sqrt`, the factor, and the multiply by it each add
# one more rounding.
function _write_sqrt!(s::AbstractVector, m::AbstractVector, n::AbstractVector{Int})
    scalarT = scalar_type(eltype(m))
    for j in _eachindex(s, m, n)
        s[j] = sqrt(m[j]) * (1 + (n[j] + 3) * eps(scalarT))
    end
    return s
end
