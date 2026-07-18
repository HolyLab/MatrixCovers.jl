# ============================================================
# φ types
# ============================================================

"""
    AbstractCoverPenalty <: Function

Supertype of the penalty functions `ϕ` that score a cover, and the type of the
first argument of most of this package's API. The built-in subtypes are
[`AbsLog`](@ref) and [`AbsLinear`](@ref).

A penalty is a function of the single ratio `r = |A[i,j]| / (a[i]*b[j])`, and
[`cover_objective`](@ref) sums it over the entries of `A`. Because `ϕ` sees only
that ratio, and every diagonal rescaling of `A` leaves it fixed, any objective
built from a penalty is automatically scale-invariant.

# Extending

A subtype must be callable on a nonnegative real:

    (::MyPenalty)(r::Real)

`r` ranges over `[0, Inf]`. Both endpoints occur and neither may error: `r = 0`
whenever `A[i,j]` is zero, and `cover_objective` passes `typemax` for an entry
left uncovered by a zero scale. Penalties are conventionally singleton structs.

That call is the whole contract, and it buys exactly one thing:
[`cover_objective`](@ref) works for any subtype. **The solvers do not.** Every
solver in this package dispatches on a concrete built-in penalty — `AbsLog{2}`
is solved natively, the `AbsLinear` penalties through JuMP — so a custom subtype
passed to [`symcover_min`](@ref), [`soft_symcover`](@ref), or any other solver
raises a `MethodError`. Scoring covers with your own penalty is supported;
minimizing it is not.
"""
abstract type AbstractCoverPenalty<:Function end

"""
    AbsLog{p}

Penalty type for

    φ(r) = |log(r)|^p  if r > 0
           0           if r = 0

The discontinuity at r=0 prevents zero entries in A from sending the objective
value to infinity.

This leads to convex optimization problems in log space. `AbsLog{1}` typically
has a flat minimum-basin in which members of an entire family of solutions are
equally good. `AbsLog{2}`, except in degenerate cases like `[0 1; 1 0]`, has a
unique minimum.

See also: [`AbsLinear`](@ref).
"""
struct AbsLog{p} <: AbstractCoverPenalty end

"""
    AbsLinear{p}

Penalty type for `φ(r) = |1 - r|^p`. Unlike [`AbsLog`](@ref), this penalty is
continuous at `r = 0` (`φ(0) = 1`), so zero entries in `A` naturally contribute a
constant penalty.

The resulting optimization problems are non-convex and may have multiple local
minima.
"""
struct AbsLinear{p} <: AbstractCoverPenalty end

(::AbsLog{p})(r::Real) where p = iszero(r) ? zero(float(r)) : abs(log(r))^p
(::AbsLinear{p})(r::Real) where p = abs(oneunit(r) - r)^p

# ============================================================
# cover_objective
# ============================================================

"""
    MatrixCovers.scalar_type(T)

The plain floating-point type underlying the element type `T`, with any units
removed. [`cover_objective`](@ref) sums the ratios `|A[i,j]| / (a[i]*b[j])`,
which are dimensionless because a cover requires
`unit(A[i,j]) == unit(a[i])*unit(b[j])`, so the score is an ordinary number
whatever the operands carry.

This cannot be expressed as `float(real(T))`: a matrix whose entries carry
different units has an abstract `eltype`, for which `real` and `oneunit` are
undefined. A unit-carrying element type therefore needs its own method.
"""
scalar_type(::Type{T}) where {T<:Number} = float(real(T))

# The accumulator type for a cover objective over `x`. `eltype` answers it when
# concrete; a matrix whose entries carry different units can have an element type
# as abstract as `Quantity`, which names no numeric type at all, so there the
# elements themselves are consulted. The extra pass is O(length(x)) against an
# objective that is already O(length(A)).
function objective_type(x)
    T = eltype(x)
    isconcretetype(T) && return scalar_type(T)
    return mapfoldl(v -> scalar_type(typeof(v)), promote_type, x; init=Bool)
end

"""
    cover_objective(ϕ, a, b, A)
    cover_objective(ϕ, a, A)

Compute the cover objective `∑_{i,j} ϕ(|A[i,j]| / (a[i] * b[j]))` for the given
penalty function `ϕ`. The two-argument form is for symmetric matrices where the cover
is `a*a'`.

The sum runs over the full grid in both forms, so in the symmetric form each
off-diagonal pair contributes twice and each diagonal entry once. This weighting
is what the `sym` solvers minimize, so the score reported here is the quantity
they optimized; code that reads a symmetric matrix through
[`foreach_support_sym`](@ref), which reports each pair once, must apply the
factor of 2 itself to match.

Zero entries of `A` are handled according to `ϕ`:
- `AbsLog{p}`: zero entries contribute 0 (φ(0) = 0 by convention).
- `AbsLinear{p}`: zero entries contribute 1 (φ(0) = |1-0|^p = 1).

See also:
- Penalty types (options for `ϕ`): [`AbsLog`](@ref), [`AbsLinear`](@ref).
- Solvers: [`symcover`](@ref), [`cover`](@ref), [`soft_symcover`](@ref), [`soft_cover`](@ref).
"""
function cover_objective(ϕ, a, b, A)
    T = promote_type(objective_type(A), objective_type(a), objective_type(b))
    s = zero(T)
    for j in eachindex(b)
        bj = b[j]
        for i in eachindex(a)
            ab = a[i] * bj
            Aij = abs(A[i, j])
            # 0/0 → 0 (no cover constraint); nonzero/0 → Inf (violated cover)
            r = iszero(ab) ? (iszero(Aij) ? zero(T) : typemax(T)) : T(Aij / ab)
            s += T(ϕ(r))
        end
    end
    return s
end
cover_objective(ϕ, a, A) = cover_objective(ϕ, a, a, A)

# Adjoint/Transpose dispatch: covering A' or transpose(A) swaps the row/column scales.
cover_objective(ϕ, a, b, A::Adjoint)   = cover_objective(ϕ, b, a, parent(A))
cover_objective(ϕ, a, b, A::Transpose) = cover_objective(ϕ, b, a, parent(A))
