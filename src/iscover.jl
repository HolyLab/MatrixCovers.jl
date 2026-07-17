# The cover predicate. Kept beside the traversal it is built on: checking coverage is
# exactly a walk over the support, since an entry that is zero constrains nothing.

"""
    iscover(a, b, A; rtol=0, atol=0)
    iscover(a, A; rtol=0, atol=0)

Test whether `a` and `b` cover `A`, that is, whether `a[i]*b[j] >= abs(A[i,j])` for every
entry. The two-argument form tests the symmetric cover `a*a'`, and requires `A` to be
square.

This is the inequality the whole package is organized around. [`cover`](@ref),
[`symcover`](@ref), and the `*_min` solvers all satisfy it by construction — up to the
roundoff noted below — so the predicate earns its keep mainly on covers that carry no such
guarantee: those from [`soft_cover`](@ref) and [`soft_symcover`](@ref), which penalize
under-coverage rather than forbidding it, and covers a caller has adjusted by hand.

`rtol` and `atol` supply the slack the producing algorithm warrants, testing
`a[i]*b[j] >= abs(A[i,j])*(1 - rtol) - atol`. Neither defaults to any slack at all, so the
bare call is the exact inequality. Use `rtol` for roundoff that scales with entry magnitude
(the log-domain arithmetic behind the heuristics warrants a few multiples of `eps`), and
`atol` for the convergence tolerance of an iterative solver. `atol` is subtracted only when
it is nonzero: `abs(A[i,j]) - atol` is undefined when the two carry different units, so an
`rtol`-only check never forms it. That makes `rtol` the only one of the two meaningful for a
dimensional `A`, whose entries need not share units — no single scalar `atol` is
commensurate with all of them.

`a` and `b` must be nonnegative; a negative scale raises an `ArgumentError`. Zero is
allowed, and is what every solver here returns for a row or column with no support. The
requirement is not cosmetic: only nonnegativity makes it sound to skip the zero entries of
`A`, which is what lets this run in time proportional to the support rather than to
`length(A)`.

`eachindex(a)` must match `axes(A, 1)` and `eachindex(b)` must match `axes(A, 2)`.

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
