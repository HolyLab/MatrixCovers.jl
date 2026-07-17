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
    cover_objective(ϕ, a, b, A)
    cover_objective(ϕ, a, A)

Compute the cover objective `∑_{i,j} ϕ(|A[i,j]| / (a[i] * b[j]))` for the given
penalty function `ϕ`. The two-argument form is for symmetric matrices where the cover
is `a*a'`.

Zero entries of `A` are handled according to `ϕ`:
- `AbsLog{p}`: zero entries contribute 0 (φ(0) = 0 by convention).
- `AbsLinear{p}`: zero entries contribute 1 (φ(0) = |1-0|^p = 1).

See also:
- Penalty types (options for `ϕ`): [`AbsLog`](@ref), [`AbsLinear`](@ref).
- Solvers: [`symcover`](@ref), [`cover`](@ref), [`soft_symcover`](@ref), [`soft_cover`](@ref).
"""
function cover_objective(ϕ, a, b, A)
    T = float(promote_type(eltype(a), eltype(b), real(eltype(A))))
    s = zero(T)
    for j in eachindex(b)
        bj = T(b[j])
        for i in eachindex(a)
            ai = T(a[i])
            Aij = T(abs(A[i, j]))
            ab = ai * bj
            # 0/0 → 0 (no cover constraint); nonzero/0 → Inf (violated cover)
            r = iszero(ab) ? (iszero(Aij) ? zero(T) : typemax(T)) : Aij / ab
            s += T(ϕ(r))
        end
    end
    return s
end
cover_objective(ϕ, a, A) = cover_objective(ϕ, a, a, A)

# Adjoint/Transpose dispatch: covering A' or transpose(A) swaps the row/column scales.
cover_objective(ϕ, a, b, A::Adjoint)   = cover_objective(ϕ, b, a, parent(A))
cover_objective(ϕ, a, b, A::Transpose) = cover_objective(ϕ, b, a, parent(A))
