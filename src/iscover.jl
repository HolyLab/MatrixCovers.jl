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

`cover`, `symcover`, and native `AbsLog{2}` minimal covers certify their results
at zero tolerance. Initializers and extension solvers may require a nonzero
`rtol`.

`a` and `b` must be nonnegative; a negative scale raises an `ArgumentError`.
Zero is allowed for unsupported rows and columns.

`eachindex(a)` must match `axes(A, 1)` and `eachindex(b)` must match `axes(A,
2)`.

See also: [`cover_objective`](@ref), [`cover`](@ref), [`symcover`](@ref).

# Examples

```jldoctest
julia> A = [1.0 2.0; 3.0 4.0];

julia> a, b = cover(A);

julia> iscover(a, b, A; rtol=8eps())
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

_iscovered(p, v, rtol, atol) = iszero(atol) ? p >= v * (1 - rtol) : p >= v * (1 - rtol) - atol

function _require_nonneg(x::AbstractVector, name::String)
    for i in eachindex(x)
        # Ask the value for zero because dimensional abstract element types may
        # not define `zero(eltype(x))`. This also rejects NaN.
        x[i] >= zero(x[i]) ||
            throw(ArgumentError("iscover requires nonnegative scales, got $name[$(string(i))] = $(string(x[i]))"))
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
    isfinite(r) ||
        throw(ArgumentError("$fname requires a positive, finite cover product on every supported entry, got $(string(p)) at ($(string(i)), $(string(j)))"))
    return r
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
