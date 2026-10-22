# Cover predicates.

"""
    iscover(a, b, A; rtol=0, atol=0)
    iscover(a, A; rtol=0, atol=0)

Test whether `a` and `b` cover `A`, that is, whether `a[i]*b[j] >= abs(A[i,j])`
for every entry. The two-argument form tests the symmetric cover `a*a'`, and
requires `A` to be square.

`rtol` and `atol` allow for small violations, testing

    a[i]*b[j] >= abs(A[i,j])*(1 - rtol) - atol

Both tolerances default to zero. A nonzero `atol` breaks scale invariance.

The results of `cover`, `symcover`, and the native `AbsLog{2}` minimal covers
pass this test at zero tolerance. Initializers and extension solvers may
require a nonzero `rtol`. Soft covers are not hard covers and in general fail
it.

`a` and `b` must be nonnegative; a negative scale raises an `ArgumentError`.
Zero is allowed for unsupported rows and columns. `eachindex(a)` must match
`axes(A, 1)` and `eachindex(b)` must match `axes(A, 2)`.

See also: [`cover_objective`](@ref), [`cover`](@ref), [`symcover`](@ref).

# Examples

```jldoctest
julia> A = [1.0 2.0; 3.0 4.0];

julia> a, b = cover(A);

julia> iscover(a, b, A)
true

julia> iscover([1.0, 1.0], [1.0, 1.0], A)   # a*b' = ones, which does not reach A[2,2]
false
```
"""
function iscover(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; rtol=0, atol=0)
    eachindex(a) == axes(A, 1) ||
        throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    eachindex(b) == axes(A, 2) ||
        throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    _require_nonneg(a, "a")
    _require_nonneg(b, "b")
    # Zero entries impose no constraint and are skipped by the traversal.
    covered = Ref(true)
    foreach_support(A) do i, j, v
        covered[] &= _iscovered(a[i] * b[j], v, rtol, atol)
    end
    return covered[]
end

function iscover(a::AbstractVector, A::AbstractMatrix; kwargs...)
    axes(A, 1) == axes(A, 2) ||
        throw(DimensionMismatch("iscover(a, A) requires a square matrix, got axes $(string(axes(A)))"))
    return iscover(a, a, A; kwargs...)
end

"""
    max_coverage_ratio(a, b, A)
    max_coverage_ratio(a, A)

Return the largest coverage ratio `abs(A[i,j])/(a[i]*b[j])` over the nonzero
entries of `A`. The two-argument form uses the symmetric cover `a*a'`, and
requires `A` to be square.

`a` and `b` cover `A` exactly when the result `r` is at most one, so this is the
quantitative form of [`iscover`](@ref). Scaling the product `a*b'` by `r` makes
it a cover that is tight at the worst entry, whether `r` is above or below one;
in the symmetric form that scaling is `sqrt(r)*a`.

Zero entries of `A` impose no constraint and are skipped. A nonzero entry over
a zero product gives `Inf`, and a matrix with no nonzero entries gives zero.
The ratio is unchanged when `A`, `a`, and `b` are rescaled together, as under a
change of units.

`a` and `b` must be nonnegative, and their indices must match those of `A`, as
for [`iscover`](@ref).

See also: [`iscover`](@ref), [`cover_objective`](@ref).

# Examples

```jldoctest
julia> A = [1.0 2.0; 3.0 4.0];

julia> max_coverage_ratio([1.0, 1.0], [1.0, 2.0], A)   # A[2,1] = 3 against a[2]*b[1] = 1
3.0

julia> max_coverage_ratio([2.0, 2.0], A)
1.0
```
"""
function max_coverage_ratio(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    eachindex(a) == axes(A, 1) ||
        throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    eachindex(b) == axes(A, 2) ||
        throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    _require_nonneg(a, "a"; fname=:max_coverage_ratio)
    _require_nonneg(b, "b"; fname=:max_coverage_ratio)
    T = promote_type(objective_type(A), objective_type(a), objective_type(b))
    worst = Ref(zero(T))
    foreach_support(A) do i, j, v
        ab = a[i] * b[j]
        r = iszero(ab) ? T(Inf) : T(v / ab)
        worst[] = max(worst[], r)
    end
    return worst[]
end

function max_coverage_ratio(a::AbstractVector, A::AbstractMatrix)
    axes(A, 1) == axes(A, 2) ||
        throw(DimensionMismatch("max_coverage_ratio(a, A) requires a square matrix, got axes $(string(axes(A)))"))
    return max_coverage_ratio(a, a, A)
end

_iscovered(p, v, rtol, atol) = iszero(atol) ? p >= v * (1 - rtol) : p >= v * (1 - rtol) - atol

function _require_nonneg(x::AbstractVector, name::String; fname::Symbol=:iscover)
    for i in eachindex(x)
        # Ask the value for zero because dimensional abstract element types may
        # not define `zero(eltype(x))`. This also rejects NaN.
        x[i] >= zero(x[i]) ||
            throw(ArgumentError("$fname requires nonnegative scales, got $name[$(string(i))] = $(string(x[i]))"))
    end
    return nothing
end

# Log-domain solvers can lose coverage to rounding. Measure the largest
# linear-arithmetic shortfall and apply a uniform inflation without changing the
# balance convention.
const CERTIFY_SWEEPS = 4

function _certify_cover!(a::AbstractVector, A::AbstractMatrix, fname::Symbol)
    T = scalar_type(eltype(a))
    for _ in 1:CERTIFY_SWEEPS
        r = _worst_shortfall(a, A, T, fname)
        r > one(T) || return a
        _inflate_nonzero!(a, _certify_factor(r))
    end
    throw(ArgumentError("$fname could not certify a cover of `A` within $CERTIFY_SWEEPS inflation sweeps"))
end

function _certify_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, fname::Symbol)
    T = scalar_type(promote_type(eltype(a), eltype(b)))
    for _ in 1:CERTIFY_SWEEPS
        r = _worst_shortfall(a, b, A, T, fname)
        r > one(T) || return a, b
        s = _certify_factor(r)
        _inflate_nonzero!(a, s)
        _inflate_nonzero!(b, s)
    end
    throw(ArgumentError("$fname could not certify a cover of `A` within $CERTIFY_SWEEPS inflation sweeps"))
end

# How far short of `v` the cover product `p` falls, as the factor `p` must grow
# by. A vanishing or non-finite product cannot be lifted onto a positive entry.
function _shortfall(p, v, i, j, fname::Symbol)
    r = v / p
    isfinite(r) && return r
    isfinite(v) ||
        throw(ArgumentError("$fname requires finite entries, got abs(A[$(string(i)), $(string(j))]) = $(string(v))"))
    throw(ArgumentError("$fname requires a positive, finite cover product on every supported entry, got $(string(p)) at ($(string(i)), $(string(j))); this usually means the cover factors exceed the exponent range of their floating-point type, and converting `A` to a type with a wider range, such as `BigFloat`, may help"))
end

# Throw an informative error when supported log-scale factors cannot be
# exponentiated in `T`. `lα[k]` is supported when `hasα(k)`; `lβ === nothing`
# checks a symmetric cover. With `gauge=true` the factors may still be
# shifted by `(t, -t)` before they are exponentiated, so the test is whether
# some shift fits both into range; otherwise the values must fit as given.
# `hint` is appended to the error message.
function _check_representable(lα::AbstractVector, hasα, lβ, hasβ, ::Type{T}, fname::Symbol;
                              gauge::Bool, hint::String="") where {T}
    lo, hi = log(floatmin(T)), log(floatmax(T))
    loα, hiα = _supported_extrema(lα, hasα)
    loβ, hiβ = lβ === nothing ? (loα, hiα) : _supported_extrema(lβ, hasβ)
    # NaN log scales are a solver failure, not a range limitation.
    any(isnan, (loα, hiα, loβ, hiβ)) && return nothing
    ok = if gauge
        max(lo - loα, hiβ - hi) <= min(hi - hiα, loβ - lo)
    else
        lo <= loα && hiα <= hi && lo <= loβ && hiβ <= hi
    end
    ok && return nothing
    dec(x) = string(round(x / log(T(10)); digits=1))
    # Row and column spans are what must fit when a common rescaling is still free.
    range = if gauge
        "the row factors span $(dec(hiα - loα)) and the column factors $(dec(hiβ - loβ)) decades, but $(string(T)) spans $(dec(hi - lo))"
    elseif lβ === nothing
        "the factors range from 10^$(dec(loα)) to 10^$(dec(hiα)), but $(string(T)) spans 10^$(dec(lo)) to 10^$(dec(hi))"
    else
        "the row factors range from 10^$(dec(loα)) to 10^$(dec(hiα)) and the column factors from 10^$(dec(loβ)) to 10^$(dec(hiβ)), but $(string(T)) spans 10^$(dec(lo)) to 10^$(dec(hi))"
    end
    throw(ArgumentError("$fname: the scale factors computed for `A` are not representable in $(string(T)): $range. Cover factors of nonsymmetric band matrices can vary geometrically along the band (see `cover_min`). Converting `A` to a type with a wider exponent range, such as `BigFloat`, avoids the overflow$hint."))
end

function _supported_extrema(lx::AbstractVector, has)
    T = eltype(lx)
    l, h = T(Inf), T(-Inf)
    for k in eachindex(lx)
        has(k) || continue
        x = lx[k]
        isnan(x) && return (x, x)
        l = min(l, x)
        h = max(h, x)
    end
    return l, h
end

# Round each factor's share of the required inflation upward.
_certify_factor(r::T) where {T} = max(nextfloat(sqrt(r)), nextfloat(one(T)))

function _inflate_nonzero!(x::AbstractVector, s)
    for i in eachindex(x)
        iszero(x[i]) || (x[i] *= s)
    end
    return x
end

# Worst factor by which a cover product must grow to reach its entry, or `one`
# when the cover already holds everywhere.
function _worst_shortfall(a::AbstractVector, A::AbstractMatrix, ::Type{T}, fname::Symbol) where {T}
    worst = Ref(one(T))
    foreach_support_sym(A) do i, j, v
        p = a[i] * a[j]
        p >= v && return
        worst[] = max(worst[], convert(T, _shortfall(p, v, i, j, fname)))
    end
    return worst[]
end

function _worst_shortfall(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                          ::Type{T}, fname::Symbol) where {T}
    worst = Ref(one(T))
    foreach_support(A) do i, j, v
        p = a[i] * b[j]
        p >= v && return
        worst[] = max(worst[], convert(T, _shortfall(p, v, i, j, fname)))
    end
    return worst[]
end

# Direct dense-storage implementations avoid callback state.
function _worst_shortfall(a::AbstractVector, A::StridedMatrix, ::Type{T}, fname::Symbol) where {T}
    worst = one(T)
    ax = axes(A, 1)
    for j in ax
        aj = a[j]
        for i in first(ax):j
            v = abs(A[i, j])
            iszero(v) && continue
            p = a[i] * aj
            p >= v && continue
            worst = max(worst, convert(T, _shortfall(p, v, i, j, fname)))
        end
    end
    return worst
end

function _worst_shortfall(a::AbstractVector, b::AbstractVector, A::StridedMatrix,
                          ::Type{T}, fname::Symbol) where {T}
    worst = one(T)
    for j in axes(A, 2)
        bj = b[j]
        for i in axes(A, 1)
            v = abs(A[i, j])
            iszero(v) && continue
            p = a[i] * bj
            p >= v && continue
            worst = max(worst, convert(T, _shortfall(p, v, i, j, fname)))
        end
    end
    return worst
end
