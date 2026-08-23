# ============================================================
# φ types
# ============================================================

"""
    AbstractCoverPenalty <: Function

Supertype of cover penalties. Built-in subtypes are [`AbsLog`](@ref) and
[`AbsLinear`](@ref).

[`cover_objective`](@ref) applies the penalty to
`r = |A[i,j]|/(a[i]*b[j])` and sums over `A`.

# Extending

A subtype must be callable on a nonnegative real:

    (::MyPenalty)(r::Real)

The method must accept `r = 0` and `r = typemax(...)`. Penalties are usually
singleton structs.

[`cover_objective`](@ref) works for any subtype, but solvers support only
specific built-in penalties: `AbsLog{2}` natively and `AbsLinear` through JuMP.
Passing a custom subtype to a solver raises a `MethodError`.
"""
abstract type AbstractCoverPenalty<:Function end

"""
    AbsLog{p}

Penalty type for

    φ(r) = |log(r)|^p  if r > 0
           0           if r = 0

The `r=0` convention keeps zero entries finite. The objective is convex in log
space; `AbsLog{1}` may have multiple minima.

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

Return the unitless floating-point type underlying `T`. Unit-carrying element
types should specialize this method.
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

Compute `∑ ϕ(|A[i,j]|/(a[i]*b[j]))`. The shorter form uses the symmetric cover
`a*a'`.

Both forms use full-grid weighting: symmetric off-diagonal pairs contribute
twice and diagonal entries once.

Zero entries of `A` are handled according to `ϕ`:
- `AbsLog{p}`: zero entries contribute 0 (φ(0) = 0 by convention).
- `AbsLinear{p}`: zero entries contribute 1 (φ(0) = |1-0|^p = 1).

`eachindex(a)` must match `axes(A, 1)` and `eachindex(b)` must match `axes(A, 2)`.

`A` is read through [`foreach_support`](@ref).

See also:
- Penalty types (options for `ϕ`): [`AbsLog`](@ref), [`AbsLinear`](@ref).
- Solvers: [`symcover`](@ref), [`cover`](@ref), [`soft_symcover`](@ref), [`soft_cover`](@ref).
"""
function cover_objective(ϕ, a, b, A)
    eachindex(a) == axes(A, 1) ||
        throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    eachindex(b) == axes(A, 2) ||
        throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    T = promote_type(objective_type(A), objective_type(a), objective_type(b))
    s = Ref(zero(T))
    nsupport = Ref(0)
    foreach_support(A) do i, j, v
        ab = a[i] * b[j]
        # A nonzero entry over a zero scale is uncovered; `typemax` is the ratio's
        # stand-in for the infinite excess.
        r = iszero(ab) ? typemax(T) : T(v / ab)
        s[] += T(ϕ(r))
        nsupport[] += 1
    end
    # Every entry outside the support has `r = 0` whatever its scales, including a
    # zero entry over a zero scale, which constrains nothing. They therefore share
    # one penalty value, and only their count is needed: zero for `AbsLog`, but a
    # nonzero constant for `AbsLinear`, which is continuous at `r = 0`.
    # The guard prevents `0 * Inf` for penalties that are infinite at zero.
    nzero = length(a) * length(b) - nsupport[]
    return iszero(nzero) ? s[] : s[] + nzero * T(ϕ(zero(T)))
end
cover_objective(ϕ, a, A) = cover_objective(ϕ, a, a, A)

# Adjoint/Transpose dispatch: covering A' or transpose(A) swaps the row/column scales.
cover_objective(ϕ, a, b, A::Adjoint)   = cover_objective(ϕ, b, a, parent(A))
cover_objective(ϕ, a, b, A::Transpose) = cover_objective(ϕ, b, a, parent(A))
