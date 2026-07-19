# The cover predicate. Kept beside the traversal it is built on: checking coverage is
# exactly a walk over the support, since an entry that is zero constrains nothing.

"""
    iscover(a, b, A; rtol=0, atol=0)
    iscover(a, A; rtol=0, atol=0)

Test whether `a` and `b` cover `A`, that is, whether `a[i]*b[j] >= abs(A[i,j])`
for every entry. The two-argument form tests the symmetric cover `a*a'`, and
requires `A` to be square.

`rtol` and `atol` allow for small violations, testing

    a[i]*b[j] >= abs(A[i,j])*(1 - rtol) - atol

The default for both tolerances is zero (no slack, test that the cover condition
holds); note that `atol != 0` breaks scale-invariance.

`a` and `b` must be nonnegative; a negative scale raises an `ArgumentError`.
Zero is allowed, and is what every solver here returns for a row or column with
no support.

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
        throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$(axes(A, 1))"))
    eachindex(b) == axes(A, 2) ||
        throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(eachindex(b)), axes(A, 2)=$(axes(A, 2))"))
    _require_nonneg(a, "a")
    _require_nonneg(b, "b")
    # Zero entries of `A` are skipped by `foreach_support`, and need no check: they demand
    # `a[i]*b[j] >= 0`, which nonnegative scales satisfy outright.
    covered = Ref(true)
    foreach_support(A) do i, j, v
        covered[] &= _iscovered(a[i] * b[j], v, rtol, atol)
    end
    return covered[]
end

function iscover(a::AbstractVector, A::AbstractMatrix; kwargs...)
    axes(A, 1) == axes(A, 2) ||
        throw(DimensionMismatch("iscover(a, A) requires a square matrix, got axes $(axes(A))"))
    return iscover(a, a, A; kwargs...)
end

_iscovered(p, v, rtol, atol) = iszero(atol) ? p >= v * (1 - rtol) : p >= v * (1 - rtol) - atol

function _require_nonneg(x::AbstractVector, name::String)
    for i in eachindex(x)
        # `zero(x[i])`, not `zero(eltype(x))`: a dimensional scale carries its units in the
        # value, and `zero(Quantity{Float64})` is undefined. Also rejects NaN, which fails
        # every comparison.
        x[i] >= zero(x[i]) ||
            throw(ArgumentError("iscover requires nonnegative scales, got $name[$i] = $(x[i])"))
    end
    return nothing
end
