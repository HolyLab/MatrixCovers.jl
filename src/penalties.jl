# ============================================================
# Penalty types
# ============================================================

"""
    AbstractCoverPenalty <: Function

Supertype of cover penalties. Built-in subtypes are [`AbsLog`](@ref),
[`AbsLinear`](@ref), and [`PowerMean`](@ref).

A penalty is a function `ρ(z)` of the log slack

    z = log(a[i]*b[j]) - log|A[i,j]| = -log(r),    r = |A[i,j]|/(a[i]*b[j]),

of a nonzero entry of `A`. A hard cover has `z ≥ 0` on every nonzero entry,
`z = 0` marks a tight entry, and `z < 0` an entry the cover violates.
[`cover_objective`](@ref) sums `ρ(z)` over the nonzero entries of `A`.

# Extending

A subtype must be callable on a real log slack:

    (::MyPenalty)(z::Real)

The method must accept `z = -Inf`, which [`cover_objective`](@ref) passes for a
nonzero entry over a zero product `a[i]*b[j]`, and `z = Inf`, which it passes
when the ratio underflows to zero. Penalties are usually singleton structs.

[`cover_objective`](@ref) works for any subtype, but solvers support only
specific built-in penalties:

- hard covers ([`symcover_min`](@ref), [`cover_min`](@ref)): `AbsLog{2}`
  natively, `AbsLog{1}` and `AbsLinear` through JuMP;
- soft covers ([`soft_symcover`](@ref), [`soft_cover`](@ref)): `PowerMean` and
  `AbsLog{2}` natively, `AbsLog{1}` and `AbsLinear` through JuMP.

Passing a custom subtype to a solver raises a `MethodError`.
"""
abstract type AbstractCoverPenalty<:Function end

"""
    AbsLog{p}

The ℓ_p penalty on the log slack,

    ρ(z) = |z|^p.

`AbsLog{1}` gives the ℓ₁ objective, a linear program, and `AbsLog{2}` the ℓ₂
objective, whose minimizer among hard covers is the ℓ₂-minimal cover. The
objective is convex in the logarithms of the scales;
`AbsLog{1}` may have multiple minima. `ρ(±Inf) = Inf`.

See also: [`AbsLinear`](@ref).
"""
struct AbsLog{p} <: AbstractCoverPenalty end

"""
    AbsLinear{p}

Penalty type for

    ρ(z) = |1 - exp(-z)|^p,

which is `|1 - r|^p` in the ratio `r = exp(-z)`. It is bounded for slack
entries, approaching `ρ(Inf) = 1`, and grows exponentially for violated ones,
with `ρ(-Inf) = Inf`.

The resulting optimization problems are not convex and may have multiple local
minima.
"""
struct AbsLinear{p} <: AbstractCoverPenalty end

"""
    PowerMean{p}

Penalty type for

    ρ(z) = (exp(-p*z) - 1 + p*z) / p,

with `p > 0`. In the ratio `r = exp(-z)`, this is `(r^p - 1 - p*log(r)) / p`.
`ρ(±Inf) = Inf`.

`PowerMean{2}` is the default penalty for soft covers. For entries modeled as
independent normal variables with standard deviations `a[i]*b[j]`, it gives the
maximum-likelihood scales.

The objective is strictly convex in the log scales, so the soft cover it defines
has unique products `a[i]*b[j]` on the support of `A`, and a symmetric `A` has a
symmetric minimizer. At the minimum, the `p`-power mean of the ratios
`r = |A[i,j]|/(a[i]*b[j])` over the nonzero entries of each row and each column
equals one: `∑_j r[i,j]^p = n_i` and `∑_i r[i,j]^p = m_j`, where `n_i` and `m_j`
count the nonzeros of row `i` and column `j`. Consequently no ratio exceeds
`min(n_i, m_j)^(1/p)`.

See also: [`soft_cover`](@ref), [`soft_symcover`](@ref).
"""
struct PowerMean{p} <: AbstractCoverPenalty
    function PowerMean{p}() where p
        (p isa Real && p > 0) ||
            throw(ArgumentError("PowerMean{p} requires a real exponent p > 0, got p = $(repr(p))"))
        return new{p}()
    end
end

(::AbsLog{p})(z::Real) where p = abs(float(z))^p
(::AbsLinear{p})(z::Real) where p = abs(expm1(-float(z)))^p
# `expm1(-t) + t` avoids the cancellation in `exp(-t) - 1 + t` near `z = 0`. At
# `z = -Inf` that sum is `Inf - Inf`.
function (::PowerMean{p})(z::Real) where p
    zf = float(z)
    zf == -Inf && return -zf
    t = p * zf
    return (expm1(-t) + t) / p
end

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
    cover_objective(ρ, a, b, A)
    cover_objective(ρ, a, A)

Compute `∑ ρ(z[i,j])` over the nonzero entries of `A`, where
`z[i,j] = log(a[i]*b[j]) - log|A[i,j]|` is the log slack. The shorter form uses
the symmetric cover `a*a'`.

Both forms sum over every nonzero entry of `A`, so in the symmetric form each
off-diagonal pair contributes twice and each diagonal entry once. Zero entries
of `A` constrain nothing and are excluded from the sum; `ρ` is never evaluated
for them. A nonzero entry over a zero product `a[i]*b[j]` has `z = -Inf`.

`eachindex(a)` must match `axes(A, 1)` and `eachindex(b)` must match `axes(A, 2)`.

The nonzero entries of `A` are enumerated by [`foreach_support`](@ref).

See also:
- Penalty types (options for `ρ`): [`AbsLog`](@ref), [`AbsLinear`](@ref), [`PowerMean`](@ref).
- Solvers that minimize it: [`symcover_min`](@ref), [`cover_min`](@ref), [`soft_symcover`](@ref), [`soft_cover`](@ref).
"""
function cover_objective(ρ, a, b, A)
    eachindex(a) == axes(A, 1) ||
        throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    eachindex(b) == axes(A, 2) ||
        throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    T = promote_type(objective_type(A), objective_type(a), objective_type(b))
    s = Ref(zero(T))
    foreach_support(A) do i, j, v
        ab = a[i] * b[j]
        # The slack comes from the unitless ratio, so scales and entries may carry
        # units. A nonzero entry over a zero scale is uncovered by an infinite amount.
        z = iszero(ab) ? -T(Inf) : -log(T(v / ab))
        s[] += T(ρ(z))
    end
    return s[]
end
cover_objective(ρ, a, A) = cover_objective(ρ, a, a, A)

# Adjoint/Transpose dispatch: covering A' or transpose(A) swaps the row/column scales.
cover_objective(ρ, a, b, A::Adjoint)   = cover_objective(ρ, b, a, parent(A))
cover_objective(ρ, a, b, A::Transpose) = cover_objective(ρ, b, a, parent(A))
