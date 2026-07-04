# ============================================================
# φ types
# ============================================================

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
struct AbsLog{p} end

"""
    AbsLinear{p}

Penalty type for `φ(r) = |1 - r|^p`. Unlike [`AbsLog`](@ref), this penalty is
continuous at `r = 0` (`φ(0) = 1`), so zero entries in `A` naturally contribute a
constant penalty.

The resulting optimization problems are non-convex and may have multiple local
minima.
"""
struct AbsLinear{p} end

(::AbsLog{p})(r::Real) where p = iszero(r) ? zero(float(r)) : abs(log(r))^p
(::AbsLinear{p})(r::Real) where p = abs(one(r) - r)^p

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
- `AbsLog{p}`: zero entries contribute 0 (φ(0) = 0 by convention; see manuscript section 2).
- `AbsLinear{p}`: zero entries contribute 1 (φ(0) = |1-0|^p = 1).

See also:
- Penalty types (options for `ϕ`): [`AbsLog`](@ref), [`AbsLinear`](@ref).
- Solvers: [`symcover`](@ref), [`cover`](@ref), [`soft_symcover`](@ref), [`soft_cover`](@ref).
"""
function cover_objective(ϕ, a, b, A)
    T = float(promote_type(eltype(a), eltype(b), eltype(A)))
    s = zero(T)
    for j in eachindex(b)
        bj = T(b[j])
        for i in eachindex(a)
            ai = T(a[i])
            Aij = abs(T(A[i, j]))
            ab = ai * bj
            # 0/0 → 0 (no cover constraint); nonzero/0 → Inf (violated cover)
            r = iszero(ab) ? (iszero(Aij) ? zero(T) : typemax(T)) : Aij / ab
            s += T(ϕ(r))
        end
    end
    return s
end
cover_objective(ϕ, a, A) = cover_objective(ϕ, a, a, A)

# ============================================================
# Internal helpers
# ============================================================

# Compute the analytical minimizer of the unconstrained AbsLog{2} symmetric objective
#   ∑_{i,j: A[i,j]≠0} (log(a[i]*a[j]) - log|A[i,j]|)²
# Fills `a` in-place and returns nza[i] = number of nonzero entries in row i.
# For efficiency, uses a Sherman-Morrison approximation for the pattern of nonzeros. (It's exact when there are no zeros.)
# This is the "rank-1 solution" described in manuscript section 5.2.
function unconstrained_min!(::AbsLog{2}, a::AbstractVector{T}, A::AbstractMatrix; logcache=nothing) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a))"))
    logcache === nothing || axes(logcache) == (ax, ax) ||
        throw(DimensionMismatch("`logcache` must have the same axes as `A` (got $(axes(logcache)) vs $((ax, ax)))"))
    loga = fill!(similar(a), zero(T))
    nza  = zeros(Int, ax)
    # When `logcache` is provided, store log|A[i,j]| for every nonzero upper-triangle
    # entry so a caller can reuse it (zero entries are skipped and never read back).
    for j in ax
        for i in first(ax):j
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            lAij = log(Aij)
            logcache === nothing || (logcache[i, j] = lAij)
            loga[i] += lAij
            nza[i]  += 1
            if i != j
                loga[j] += lAij
                nza[j]  += 1
            end
        end
    end
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in ax
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
    end
    return nza
end

function unconstrained_min!(::AbsLog{2}, a::AbstractVector{T}, b::AbstractVector{T}, A::AbstractMatrix) where T
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, b, A)` requires row indices of `A` to match `a`, got axes(A, 1)=$(axes(A, 1)), axes(a)=$(axes(a))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, b, A)` requires column indices of `A` to match `b`, got axes(A, 2)=$(axes(A, 2)), axes(b)=$(axes(b))"))
    loga = fill!(similar(a), zero(T))
    logb = fill!(similar(b), zero(T))
    nza  = zeros(Int, axes(A, 1))
    nzb  = zeros(Int, axes(A, 2))
    logmu   = zero(T)
    nztotal = 0
    for j in axes(A, 2)
        for i in axes(A, 1)
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            lAij = log(Aij)
            loga[i] += lAij
            logb[j] += lAij
            nza[i]  += 1
            nzb[j]  += 1
            logmu   += lAij
            nztotal += 1
        end
    end
    halfmu = iszero(nztotal) ? zero(T) : T(logmu / (2 * nztotal))
    for i in axes(A, 1)
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
    end
    for j in axes(A, 2)
        b[j] = iszero(nzb[j]) ? zero(T) : exp(logb[j] / nzb[j] - halfmu)
    end
    return nza, nzb
end

# Build a feasible hard cover by processing diagonals in order of increasing
# offset. When both a[k] and a[l] are already nonzero but a[k]*a[l] < |A[k,l]|,
# both a[k] and a[l] are scaled by the square root of the ratio,
#               √(|A[k,l]| / (a[k]*a[l]))
# Equal scaling is of course ad-hoc; while it might be better to do something
# tuned to a particular penalty function, that would risk making the algorithm
# O(n^3) (we'd likely need to revisit the previous diagonals), and earlier
# decisions might be reversed by later ones anyway. For something intended as
# an initialization, using a heuristic that is guaranteed to be O(n^2) seems
# like a reasonable choice.

# When both a[k] and a[l] are zero at a nonzero A[k,l], deferral is used: the
# constraint is held pending until a later diagonal provides scale for one of
# the two indices. If both remain zero after all diagonals are processed, the
# constraint is resolved by a[k]=a[l]=√|A[k,l]|.
function init_feasible!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    n  = length(ax)
    f  = first(ax)

    # Diagonal: a[k] = √|A[k,k]|
    for k in ax
        a[k] = sqrt(T(abs(A[k, k])))
    end

    # Off-diagonals in order of increasing offset j = 1, …, n-1.
    # Each nonzero A[k, k+j] requires a[k]*a[k+j] ≥ |A[k, k+j]|.
    deferred = Tuple{Int,Int,T}[]
    for j in 1:n-1
        for ik in 0:n-j-1
            k   = f + ik
            l   = k + j
            Akl = T(abs(A[k, l]))
            iszero(Akl) && continue
            ak, al = a[k], a[l]
            if !iszero(ak) && !iszero(al)
                aprod = ak * al
                if aprod < Akl
                    s = sqrt(Akl / aprod)
                    a[k] *= s; a[l] *= s
                end
            elseif !iszero(ak)
                a[l] = Akl / ak
            elseif !iszero(al)
                a[k] = Akl / al
            else
                push!(deferred, (k, l, Akl))
            end
        end
    end

    # Resolve deferred constraints: re-scan until no more progress, then equal-split.
    while !isempty(deferred)
        changed = false
        filter!(deferred) do (k, l, v)
            ak, al = a[k], a[l]
            if !iszero(ak) && !iszero(al)
                aprod = ak * al
                if aprod < v
                    s = sqrt(v / aprod)
                    a[k] *= s; a[l] *= s
                end
            elseif !iszero(ak)
                a[l] = v / ak
            elseif !iszero(al)
                a[k] = v / al
            else
                return true   # still unresolvable; keep in list
            end
            changed = true
            return false      # resolved; drop from list
        end
        changed && continue
        # No progress: all remaining have both indices zero.
        # Process in order so earlier equal-splits can inform later ones in the same pass.
        for (k, l, v) in deferred
            ak, al = a[k], a[l]
            if iszero(ak) && iszero(al)
                a[k] = a[l] = sqrt(v)
            elseif iszero(ak)
                a[k] = v / al
            elseif iszero(al)
                a[l] = v / ak
            else
                aprod = ak * al
                if aprod < v
                    s = sqrt(v / aprod)
                    a[k] *= s; a[l] *= s
                end
            end
        end
        break
    end

    return a
end

# Compute the eigenvector of the AbsLog{2} symmetric Hessian with the greatest curvature.
# The Hessian H ≈ Diagonal(nza) + nza*nza'/sum(nza) (Sherman-Morrison rank-1 form).
# The characteristic equation (section 5.3.1 of the manuscript) is:
#   ∑_i n_i² / (λ - n_i) = ∑_i n_i   (N = ∑ n_i)
# solved by Newton's method.  The eigenvector components are v_i ∝ n_i / (λ - n_i).
# Accepts any positive-valued weight vector (not just integer counts).
function _abslog2_greatest_curvature_eigvec(nza::AbstractVector{<:Real})
    N = float(sum(nza))
    iszero(N) && return zeros(Float64, length(nza))
    λ = 2.0 * maximum(nza)  # initial guess: above all n_i so all (λ - n_i) > 0
    for _ in 1:20           # generous bound; quadratic convergence exits early
        s1 = 0.0
        s2 = 0.0
        for i in eachindex(nza)
            ni = nza[i]
            iszero(ni) && continue
            d = λ - ni
            s1 += ni^2 / d
            s2 += ni^2 / d^2
        end
        abs(s1 - N) < 1e-12 * N && break
        iszero(s2) && break
        λ += (s1 - N) / s2
    end
    # `λ` is reassigned in the loop, so a comprehension capturing it would box it
    # (Core.Box), losing type stability; copy to a binding that is never reassigned.
    λ★ = λ
    v = [iszero(nza[i]) ? 0.0 : nza[i] / (λ★ - nza[i]) for i in eachindex(nza)]
    nv = sqrt(sum(abs2, v))
    return iszero(nv) ? v : v ./ nv
end

# AbsLinear symmetric initialization. Start at the AbsLog{2} unconstrained
# minimum, then move along the eigenvector of greatest Hessian curvature by the
# smallest nonnegative distance that makes the cover feasible
# (a[i]*a[j] >= |A[i,j]| for every entry). Moving along this direction keeps the
# step scale-covariant and reaches feasibility with the least perturbation of
# the unconstrained minimum. Fills `a` in place and returns it.
function _symcover_abslinear_init!(a::AbstractVector{T}, A::AbstractMatrix; cache=nothing) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`_symcover_abslinear_init!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a)))"))
    nza = unconstrained_min!(AbsLog{2}(), a, A; logcache=cache)
    v   = _abslog2_greatest_curvature_eigvec(nza)
    # Feasibility requires a[i]*a[j] >= |A[i,j]|; move along v by the least t
    # achieving it, using deficit = log|A[i,j]| - log a[i] - log a[j]. Whenever
    # |A[i,j]|>0 both a[i],a[j]>0 (a zero entry of `a` means an all-zero row), so
    # log a[i] is well defined.
    #
    # The logarithm and division are needed only for entries that raise the
    # running maximum t_feas: an entry can beat it only if its deficit exceeds
    # s*t_feas >= 2*vmin*t_feas (s = v[i]+v[j], vmin the least participating
    # eigenvector entry), i.e. only if |A[i,j]| > exp(2*vmin*t_feas)*a[i]*a[j].
    # Maintaining that threshold reduces the scan to one multiply-and-compare
    # per entry, with log/divide costs only along the increasing-max chain.
    # (Rounding of exp/log can misjudge the filter by ~1 ulp of t_feas; t_feas
    # is only accurate to rounding anyway.) For chain entries, `cache` supplies
    # log|A[i,j]| when present so it need not be recomputed.
    lα = similar(a)
    vmin = T(Inf)
    for i in ax
        if iszero(a[i])
            lα[i] = zero(T)
        else
            lα[i] = log(a[i])
            vmin = min(vmin, T(v[i]))
        end
    end
    t_feas = zero(T)
    thresh = one(T)
    for j in ax
        aj, lαj = a[j], lα[j]
        for i in first(ax):j
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            if thresh < T(Inf)
                # a rounded-to-Inf product means the true product exceeds
                # floatmax >= Aij, so skipping remains conservative
                Aij <= thresh * (a[i] * aj) && continue
            else
                # exp overflowed: fall back to skipping only feasible entries
                Aij <= a[i] * aj && continue
            end
            s = T(v[i]) + T(v[j])
            iszero(s) && continue
            lAij = cache === nothing ? log(Aij) : cache[i, j]
            t = (lAij - lα[i] - lαj) / s
            if t > t_feas
                t_feas = t
                thresh = exp(2 * vmin * t_feas)
            end
        end
    end
    for i in ax
        a[i] *= exp(t_feas * T(v[i]))
    end
    return a
end

# Leave-one-out geometric mean, an alternative starting point to the AbsLog{2}
# geometric-mean init for the AbsLinear soft cover. The geometric mean weights every nonzero
# entry equally, so an entry with |A[i,j]| far below the rest (in the scale-invariant sense
# of its log-residual z[i,j] = log|A[i,j]| - α[i] - α[j] at the unweighted minimum) skews
# the start into a worse basin than the exact-zero limit. Here the entry with the most
# negative residual is dropped from the support and the geometric mean recomputed, giving a
# start in the basin that treats that entry as effectively zero; the AbsLinear objective is
# finite at r = 0, so refinement then varies continuously as the entry vanishes.
#
# Scale-covariance: the residuals z are scale-invariant, so selecting the entry by argmin z
# is covariant, as is the reduced-support geometric mean. Residual ties are broken by
# ascending raw |A[i,j]| — NOT scale-invariant, but exact ties are precisely where
# covariance is unachievable: whenever A is scale-equivalent to a row/column permutation of
# itself (true of EVERY symmetric 2×2 with nonzero off-diagonal, via t² = A[2,2]/A[1,1]),
# the competing basins have exactly equal objectives, so no deterministic algorithm can be
# simultaneously scale-covariant, permutation-equivariant, and continuous there. The raw
# magnitude is the only continuity-relevant information left, and using it only on ties
# confines the covariance exception to that degenerate class. (Weighting all entries by raw
# |A[i,j]|² instead would carry per-entry physical units — incommensurate sums — and break
# covariance on an open set of matrices.)
#
# Returns `true` and fills `a` with the leave-one-out start, or returns `false` (leaving `a`
# unspecified) when no entry can be dropped: empty support, or dropping the selected entry
# would empty some row's support.
function _leaveout_logmean_init!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`_leaveout_logmean_init!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a)))"))
    nza = unconstrained_min!(AbsLog{2}(), a, A)
    sum(nza) == 0 && return false
    # Most negative residual over the support, with a roundoff-tolerant tie set: exact ties
    # (e.g. z[1,1] == z[2,2] for every 2×2) must not be ordered by floating-point noise.
    zmin = T(Inf)
    for j in ax, i in first(ax):j
        Aij = abs(A[i, j])
        iszero(Aij) && continue
        zmin = min(zmin, log(T(Aij)) - log(a[i]) - log(a[j]))
    end
    ztol = 64 * eps(T) * max(one(T), abs(zmin))
    ibest = jbest = first(ax) - 1
    Abest = T(Inf)
    for j in ax, i in first(ax):j
        Aij = T(abs(A[i, j]))
        iszero(Aij) && continue
        z = log(Aij) - log(a[i]) - log(a[j])
        if z <= zmin + ztol && Aij < Abest
            ibest, jbest, Abest = i, j, Aij
        end
    end
    # Dropping entry (i,j) removes one support count from row i and (if off-diagonal) row j.
    nza[ibest] > 1 || return false
    ibest == jbest || nza[jbest] > 1 || return false
    # Minimize the unconstrained AbsLog{2} objective over the reduced support by
    # Gauss-Seidel on its normal equations, starting from the full-support solution
    # already in `a`. The closed-form geometric-mean formula used by
    # `unconstrained_min!` is exactly scale-covariant only for rank-1 support
    # patterns, which the reduced support never is; a Gauss-Seidel update, by
    # contrast, is exactly covariant from any covariant iterate, for any sweep
    # count, so basin selection downstream cannot depend on the frame.
    α = similar(a)
    for i in ax
        α[i] = iszero(nza[i]) ? zero(T) : log(a[i])
    end
    for _ in 1:8
        for i in ax
            iszero(nza[i]) && continue
            num = zero(T)   # Σ_j W[i,j] (log|A[i,j]| - α[j]), α[i]-coefficient split out
            den = zero(T)
            for j in ax
                (min(i, j) == ibest && max(i, j) == jbest) && continue
                Aij = abs(A[i, j])
                iszero(Aij) && continue
                lAij = log(Aij)
                if j == i
                    num += lAij
                    den += 2
                else
                    num += lAij - α[j]
                    den += 1
                end
            end
            iszero(den) && continue   # row's only support was the dropped entry (guarded above)
            α[i] = num / den
        end
    end
    for i in ax
        a[i] = iszero(nza[i]) ? zero(T) : exp(α[i])
    end
    return true
end

# ============================================================
# symcover
# ============================================================

"""
    a = symcover(ϕ, A; iter=3)
    a = symcover(A; iter=3)

Given a square matrix `A` assumed to be symmetric, return a vector `a` representing
a symmetric hard cover of `A`: `a[i] * a[j] >= abs(A[i, j])` for all `i`, `j`.

The penalty function `ϕ` controls what objective is approximately minimized:
- `AbsLog{p}()` (p=1 or 2): initializes from the unconstrained AbsLog{2} minimum and tightens.
- `AbsLinear{p}()`: initializes from the AbsLog{2} minimum, moves along the eigenvector of
  greatest curvature to reach feasibility, then tightens.

The default `ϕ = AbsLinear{2}()`. After initialization, `iter` iterations of the tightening
algorithm (Algorithm 1 of the manuscript) are applied.

For the `AbsLinear` initialization, the feasibility step recomputes `log|A[i, j]|` for
under-covered entries. Passing a scratch matrix as `cache` (with the same axes as `A`) lets
that logarithm be taken once, in the geometric-mean pass, and reused. This is worthwhile when
covering many matrices of the same size: allocate `cache = similar(A, float(eltype(A)))` once
and reuse it across calls. The result is unchanged; only the redundant logarithms are saved.

See also: [`symcover_min`](@ref), [`soft_symcover`](@ref), [`cover`](@ref).

# Examples

```jldoctest
julia> A = [4 1; 1 4];

julia> a = symcover(A)
2-element Vector{Float64}:
 2.0
 2.0

julia> a * a'   # covers |A|: a[i]*a[j] >= abs(A[i, j])
2×2 Matrix{Float64}:
 4.0  4.0
 4.0  4.0
```
"""
symcover(A::AbstractMatrix; iter::Int=3, cache=nothing) = symcover(AbsLinear{2}(), A; iter, cache)

function symcover(ϕ::AbsLog, A::AbstractMatrix; iter::Int=3)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(A))
    a = similar(A, T, ax)
    unconstrained_min!(AbsLog{2}(), a, A)
    # Clamp diagonal: must have a[i]² >= |A[i,i]|
    for i in ax
        Aii = T(abs(A[i, i]))
        if a[i]^2 < Aii
            a[i] = sqrt(Aii)
        end
    end
    # Boost off-diagonal: ensure a[i]*a[j] >= |A[i,j]| for all i<j
    for j in ax
        aj = a[j]
        for i in first(ax):j-1
            Aij = T(abs(A[i, j]))
            iszero(Aij) && continue
            ai = a[i]
            aprod = ai * aj
            if aprod < Aij
                s = sqrt(Aij / aprod)
                a[i] = s * ai
                aj = s * aj
            end
        end
        a[j] = aj
    end
    return tighten_cover!(a, A; iter)
end

function symcover(ϕ::AbsLinear, A::AbstractMatrix; iter::Int=3, cache=nothing)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(A))
    a = similar(A, T, ax)
    _symcover_abslinear_init!(a, A; cache)
    return tighten_cover!(a, A; iter)
end

# ============================================================
# soft_symcover
# ============================================================

"""
    a = soft_symcover(ϕ, A; iter=20, starts=8, σ=2.0, rng=MersenneTwister(0))
    a = soft_symcover(A; iter=20, starts=8, σ=2.0, rng=MersenneTwister(0))

Given a square matrix `A` assumed to be symmetric, return a vector `a` approximately
minimizing the soft-cover objective `∑_{i,j} ϕ(|A[i,j]| / (a[i]*a[j]))`.

Unlike [`symcover`](@ref), there is no hard coverage constraint: `a[i]*a[j]` may be
less than `|A[i,j]|`, with violations penalized by `ϕ`.

Supported penalty functions:
- `AbsLog{2}()`: returns the analytical unconstrained minimum (no iterations needed).
- `AbsLog{1}()`: initializes from the AbsLog{2} minimum, then refines by coordinate descent
  with a log-space weighted-median step. The AbsLog{1} objective has a flat basin of equally
  good minima; this returns the deterministic, scale-covariant representative reached by
  coordinate descent from the AbsLog{2} minimum.
- `AbsLinear{2}()` (default): non-convex; refined by coordinate descent from `starts`
  scale-covariant starting points, keeping the lowest-objective result (see below).
- `AbsLinear{1}()`: initializes from the `AbsLinear{2}()` result, coordinate descent uses a
  weighted-median step.

For the `AbsLinear` penalties the objective is non-convex, so `starts` starting points are
tried and the best kept. The first four are deterministic — the geometric-mean minimum, the
tightened hard cover, a curvature-eigenvector feasibility init, and a leave-one-out geometric
mean that drops the support entry with the most negative log-residual (this last start keeps
the result continuous as an entry `|A[i,j]|` approaches zero) — and the rest are
multiplicative log-normal perturbations `a .* exp.(σ .* ξ)` of the geometric-mean point with
spread `σ`, `ξ` drawn from `rng`. Every start co-varies with a diagonal rescaling of `A` and
the objective is scale-invariant, so the selection is scale-covariant. The default `rng` is a
fresh `MersenneTwister(0)` per call, making repeated calls (and the two frames of a covariance
check) agree; pass your own `rng` for reproducibility you control, since default RNG streams
are not stable across Julia versions.

See also: [`symcover`](@ref), [`cover_objective`](@ref), [`soft_symcover_min`](@ref).

# Examples

The multistart converges to the covariant minimizer to within its objective
tolerance; round to compare against exact values.

```jldoctest
julia> A = [4 -1; -1 0];

julia> round.(soft_symcover(A); digits=4)
2-element Vector{Float64}:
 2.0
 0.5

julia> round.(soft_symcover([0 1; 1 0]); digits=4)
2-element Vector{Float64}:
 1.0
 1.0
```
"""
soft_symcover(A::AbstractMatrix; iter::Int=20, starts::Int=8, σ::Real=2.0,
              rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED)) =
    soft_symcover(AbsLinear{2}(), A; iter, starts, σ, rng)

function soft_symcover(ϕ::AbsLog{2}, A::AbstractMatrix)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    T = float(eltype(A))
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    unconstrained_min!(ϕ, a, A)   # analytical minimum; no iterations needed
    return a
end

function soft_symcover(ϕ::AbsLog{1}, A::AbstractMatrix; iter::Int=20)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    T = float(eltype(A))
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    unconstrained_min!(AbsLog{2}(), a, A)   # convex AbsLog{2} minimum: a good start
    _abslog1_iter!(a, A, iter)
    return a
end

function soft_symcover(ϕ::AbsLinear{2}, A::AbstractMatrix; iter::Int=20, starts::Int=8, σ::Real=2.0,
                       rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    return _soft_symcover_abslinear2(A, iter, starts, σ, rng)
end

function soft_symcover(ϕ::AbsLinear{1}, A::AbstractMatrix; iter::Int=20, starts::Int=8, σ::Real=2.0,
                       rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    # Initialize from the AbsLinear{2} soft cover (a good starting point for AbsLinear{1});
    # the multistart is spent there, then the AbsLinear{1} weighted-median descent refines.
    a = soft_symcover(AbsLinear{2}(), A; iter=5, starts, σ, rng)
    _abslinear1_iter!(a, A, iter)
    return a
end

# Default seed for the multistart perturbation RNG. Callers wanting reproducibility across
# Julia versions (whose default RNG streams are not stable) should pass their own `rng`.
const _MULTISTART_SEED = 0

# Relative-objective margin a later start must beat the incumbent by to replace it. Set well
# above the descent's objective tolerance (~1e-14) and well below any genuine basin gap, so
# same-basin convergence noise never flips the selection (which would break covariance) while
# real improvements always do.
const _MULTISTART_SWITCHTOL = 1e-9

# Labeled candidate starts for the symmetric AbsLinear{2} multistart, in selection order.
# Deterministic starts: the AbsLog{2} geometric-mean minimum (also the perturbation base),
# the tightened hard cover, the curvature-eigenvector feasibility init, the leave-one-out
# geometric mean (when a support entry can be dropped), and — only when `A` has a zero entry
# — the greedy feasible cover `init_feasible!`. The feasible start is gated because on a
# fully dense `A` it never uniquely wins, and "which entries are zero" is invariant under a
# diagonal rescaling `D*A*D`, so the gate keeps the selection scale-covariant. Remaining
# slots, up to `starts` total, are multiplicative log-normal perturbations `a_g .* exp.(σ .* ξ)`
# of the geometric-mean point, `ξ` drawn from `rng` (drawn for every index so the stream is
# frame-independent). `starts` below the number of deterministic starts truncates the list.
function _soft_symcover_abslinear2_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
    ax = axes(A, 1)
    T = float(eltype(A))
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    ag = similar(Array{T}, ax)
    unconstrained_min!(AbsLog{2}(), ag, A)   # geometric mean; also the perturbation base
    labels = ["geomean"]
    inits = [copy(ag)]
    length(inits) < starts && (push!(labels, "hardcover"); push!(inits, symcover(AbsLinear{2}(), A)))
    if length(inits) < starts
        ce = similar(ag); _symcover_abslinear_init!(ce, A); push!(labels, "eigvec"); push!(inits, ce)
    end
    if length(inits) < starts
        lo = similar(ag); _leaveout_logmean_init!(lo, A) && (push!(labels, "leaveout"); push!(inits, lo))
    end
    if length(inits) < starts && any(iszero, A)
        fe = similar(ag); init_feasible!(fe, A); push!(labels, "feasible"); push!(inits, fe)
    end
    k = 0
    while length(inits) < starts
        p = similar(ag)
        for i in ax
            ξ = randn(rng)
            p[i] = ag[i] > 0 ? ag[i] * exp(T(σ) * T(ξ)) : zero(T)
        end
        k += 1; push!(labels, "rand$k"); push!(inits, p)
    end
    return labels, inits
end

# Index of the multistart winner among candidate objectives `objs`: the earliest candidate not
# beaten by a strict relative improvement. Switching only on a genuine improvement is what keeps
# the selection scale-covariant — candidates landing in the same basin converge to the same
# objective only to the descent tolerance (~1e-14), and switching on that noise would forfeit
# covariance, since the incumbent (ordered geometric-mean first) is the most covariant start.
# Real basin improvements are far larger. This is the single source of the selection rule, so a
# caller that captures `objs` (below) recovers the winner exactly, without re-deriving it.
function _multistart_select(objs)
    besti = firstindex(objs)
    Ebest = objs[besti]
    for k in eachindex(objs)
        objs[k] < Ebest * (1 - _MULTISTART_SWITCHTOL) && ((besti, Ebest) = (k, objs[k]))
    end
    return besti
end

# Scale-covariant multistart for the symmetric AbsLinear{2} soft cover. Runs the single-start
# coordinate descent `_abslinear2_iter!` from the candidate list built by
# `_soft_symcover_abslinear2_inits` and returns the candidate `_multistart_select` picks. Every
# start co-varies with a diagonal rescaling `D*A*D` and the objective is scale-invariant, so the
# selection is scale-covariant; passing the same `rng` state across the two frames (as the
# default fresh-seeded RNG does) makes it reproducible.
#
# For cheap provenance auditing (which initialization earned the selection), pass `labels` and/or
# `objs` as empty vectors: they are filled in place with every candidate's label and final
# objective, in candidate order, so the winner is `labels[_multistart_select(objs)]`.
function _soft_symcover_abslinear2(A::AbstractMatrix, iter::Int, starts::Int, σ::Real, rng;
                                   labels=nothing, objs=nothing)
    labs, inits = _soft_symcover_abslinear2_inits(A, starts, σ, rng)
    E = map(inits) do a
        _abslinear2_iter!(a, A, iter)
        cover_objective(AbsLinear{2}(), a, A)
    end
    labels === nothing || append!(labels, labs)
    objs === nothing || append!(objs, E)
    return inits[_multistart_select(E)]
end

# Coordinate-descent iteration for AbsLinear{2} soft cover.
# Each coordinate a[k] is updated to the exact minimizer of
#   (1 - d/x²)² + ∑_{j≠k} (1 - c_j/x)²
# where d = |A[k,k]| and c_j = |A[k,j]|/a[j].
# Closed form when d=0 (x = s2/s1); Newton on a cubic otherwise.
#
# `iter` bounds the sweeps; the descent exits early once every coordinate's
# stationarity residual r_k = ∑_j (1 - ρ)ρ (ρ = |A[k,j]|/(a[k]a[j])), the
# gradient of the objective in log a[k], has magnitude below `tol` at the start
# of a sweep. The residual is available for free from the sums already formed
# for the update (r_k = s1/a[k] - s2/a[k]² + d/a[k]² - d²/a[k]⁴), it is the exact
# quantity optimality demands be zero, and it is scale-invariant (each ρ is), so
# covariant restarts of a rescaled problem exit on the same sweep and the
# multistart selection stays covariant.
function _abslinear2_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=1e-8) where T
    ax = eachindex(a)
    for _ in 1:iter
        maxres = zero(T)
        for k in ax
            d  = T(abs(A[k, k]))   # diagonal entry
            s1 = zero(T)            # ∑_{j≠k: A[k,j]≠0} |A[k,j]| / a[j]
            s2 = zero(T)            # ∑_{j≠k: A[k,j]≠0} |A[k,j]|² / a[j]²
            for j in ax
                j == k && continue
                Akj = T(abs(A[k, j]))
                iszero(Akj) && continue
                inv_aj = one(T) / a[j]
                s1 += Akj * inv_aj
                s2 += Akj^2 * inv_aj^2
            end
            ak = a[k]
            if !iszero(ak)
                inv_ak = one(T) / ak
                res = (s1 - (s2 - d) * inv_ak) * inv_ak - d^2 * inv_ak^4
                maxres = max(maxres, abs(res))
            end
            if iszero(s1)
                x = iszero(d) ? zero(T) : sqrt(d)
            elseif iszero(d)
                x = s2 / s1
            else
                # The cubic g(x) = s1*x³ + (d - s2)*x² - d² has exactly one positive root
                # (one Descartes sign change), and it is bracketed by √d and s2/s1:
                # g(√d) = d*(s1*√d - s2) and g(s2/s1) = d*(s2²/s1² - d) have opposite signs.
                # Safeguarded Newton with geometric bisection: the bracket endpoints can be
                # separated by hundreds of orders of magnitude, so fallback steps must bisect
                # in log space to converge in O(60) iterations.
                lo, hi = minmax(sqrt(d), s2 / s1)
                x = sqrt(lo * hi)
                while hi - lo > 2 * eps(hi)
                    gx = s1*x^3 + (d - s2)*x^2 - d^2
                    iszero(gx) && break
                    if gx > 0
                        hi = x
                    else
                        lo = x
                    end
                    gxp = 3*s1*x^2 + 2*(d - s2)*x
                    xn = x - gx/gxp
                    x = lo < xn < hi ? xn : sqrt(lo * hi)
                end
            end
            a[k] = x
        end
        maxres <= T(tol) && break
    end
    return a
end

# Weighted median of the values in `c` using the values themselves as weights: the
# point `m` with ∑_{cᵢ<m} cᵢ = ∑_{cᵢ>m} cᵢ. This minimizes ∑ᵢ |1 - cᵢ/x| over x > 0 —
# the gradient is (1/x²)(∑_{cᵢ<x} cᵢ − ∑_{cᵢ>x} cᵢ), whose positive 1/x² factor leaves
# the root at that balance point. Sorts `c` in place and returns the lower weighted
# median, a deterministic, scale-covariant tie-break on the flat basin the AbsLinear{1}
# objective admits.
function _weighted_self_median!(c::AbstractVector{T}) where T
    sort!(c)
    half = sum(c) / 2
    cum  = zero(T)
    wm   = first(c)
    for ci in c
        cum += ci
        wm   = ci
        cum >= half && break
    end
    return wm
end

# Coordinate-descent iteration for AbsLinear{1} soft cover.
# Each coordinate a[k] is updated to minimize ∑_j |1 - |A[k,j]|/(a[k]*a[j])|.
# For the off-diagonal sum, the minimizer is the weighted median of c_j with weights c_j,
# where c_j = |A[k,j]|/a[j].  When A[k,k] ≠ 0 we also compare against sqrt(|A[k,k]|).
#
# `iter` bounds the sweeps; the descent exits early once the largest relative
# coordinate movement in a sweep drops to `tol`. The median update reaches an
# exact fixed point (identical ordering selects the same value), so movement
# falls to zero there. Relative movement is scale-invariant, so covariant
# restarts of a rescaled problem exit on the same sweep.
function _abslinear1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=1e-12) where T
    ax  = eachindex(a)
    buf = Vector{T}(undef, length(ax))   # reusable buffer for c_j values
    for _ in 1:iter
        maxrel = zero(T)
        for k in ax
            d  = T(abs(A[k, k]))
            nc = 0
            for j in ax
                j == k && continue
                Akj = T(abs(A[k, j]))
                iszero(Akj) && continue
                aj = a[j]
                iszero(aj) && continue
                nc += 1
                buf[nc] = Akj / aj
            end
            if nc == 0
                x = iszero(d) ? zero(T) : sqrt(d)
            else
                c = view(buf, 1:nc)
                wm = _weighted_self_median!(c)   # sorts `c` in place
                if iszero(d)
                    x = wm
                else
                    # Compare objective at weighted median vs sqrt(d)
                    sq_d = sqrt(d)
                    obj_at(x) = abs(1 - d/x^2) + sum(abs(1 - ci/x) for ci in c)
                    x = obj_at(wm) <= obj_at(sq_d) ? wm : sq_d
                end
            end
            ak  = a[k]
            den = max(abs(x), abs(ak))
            iszero(den) || (maxrel = max(maxrel, abs(x - ak) / den))
            a[k] = x
        end
        maxrel <= T(tol) && break
    end
    return a
end

# Coordinate-descent iteration for AbsLog{1} soft cover, working in log space (α = log a).
# Each coordinate α[k] is updated to minimize ∑_{j: A[k,j]≠0} |α[k] + α[j] - log|A[k,j]||.
# Holding the neighbours fixed, the minimizer over α[k] is the median of the points
# log|A[k,j]| - α[j], one per off-diagonal nonzero entry. The diagonal term is
# |2α[k] - log|A[k,k]|| = 2|α[k] - log|A[k,k]|/2|, i.e. the point log|A[k,k]|/2 with weight two,
# represented here by inserting it twice so a plain median carries the weighting. The AbsLog{1}
# minimum is a flat basin; the lower median is chosen for a deterministic, scale-covariant result.
#
# `iter` bounds the sweeps; the descent exits early once the largest relative
# coordinate movement in a sweep drops to `tol`. The median update reaches an
# exact fixed point, so movement falls to zero there. Relative movement is
# scale-invariant, so covariant restarts of a rescaled problem exit on the same
# sweep.
function _abslog1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=1e-12) where T
    ax  = eachindex(a)
    buf = Vector{T}(undef, 2 * length(ax) + 1)   # off-diagonals (×1) + diagonal (×2)
    for _ in 1:iter
        maxrel = zero(T)
        for k in ax
            iszero(a[k]) && continue    # zero rows/columns stay uncovered
            n = 0
            Akk = T(abs(A[k, k]))
            if !iszero(Akk)
                lhalf = log(Akk) / 2
                buf[n += 1] = lhalf
                buf[n += 1] = lhalf
            end
            for j in ax
                j == k && continue
                Akj = T(abs(A[k, j]))
                iszero(Akj) && continue
                aj = a[j]
                iszero(aj) && continue
                buf[n += 1] = log(Akj) - log(aj)
            end
            n == 0 && continue
            c = view(buf, 1:n)
            sort!(c)
            x   = exp(c[(n + 1) ÷ 2])   # lower median
            ak  = a[k]
            den = max(abs(x), abs(ak))
            iszero(den) || (maxrel = max(maxrel, abs(x - ak) / den))
            a[k] = x
        end
        maxrel <= T(tol) && break
    end
    return a
end

# ============================================================
# cover
# ============================================================

"""
    a, b = cover(ϕ, A; iter=3)
    a, b = cover(A; iter=3)

Given a matrix `A`, return vectors `a` and `b` such that `a[i] * b[j] >= abs(A[i, j])`
for all `i`, `j`. The initialization is the AbsLog{2} unconstrained minimum (geometric
mean of nonzero entries per row/column), independent of `ϕ`. After boosting to feasibility,
`iter` tightening iterations are applied.

The default `ϕ = AbsLinear{2}()`.

See also: [`cover_min`](@ref), [`symcover`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [1 2 3; 6 5 4];

julia> a, b = cover(A)
([1.2674308473260654, 3.4759059767492304], [1.7261686708831454, 1.61137045961268, 2.366993044495631])

julia> a * b'
2×3 Matrix{Float64}:
 2.1878  2.0423   3.0
 6.0     5.60097  8.22745
```
"""
cover(A::AbstractMatrix; iter::Int=3) = cover(AbsLinear{2}(), A; iter)

function cover(ϕ, A::AbstractMatrix; iter::Int=3)
    T = float(eltype(A))
    a = zeros(T, axes(A, 1))
    b = zeros(T, axes(A, 2))
    unconstrained_min!(AbsLog{2}(), a, b, A)
    # Boost to feasibility
    for j in axes(A, 2)
        for i in axes(A, 1)
            Aij, ai, bj = abs(A[i, j]), a[i], b[j]
            aprod = ai * bj
            aprod >= Aij && continue
            s = sqrt(Aij / aprod)
            a[i] = s * ai
            b[j] = s * bj
        end
    end
    return tighten_cover!(a, b, A; iter)
end

# ============================================================
# soft_cover
# ============================================================

"""
    a, b = soft_cover(ϕ, A; iter=100, starts=8, σ=2.0, rng=MersenneTwister(0))
    a, b = soft_cover(A; iter=100, starts=8, σ=2.0, rng=MersenneTwister(0))

Given a matrix `A`, return vectors `a` and `b` approximately minimizing the soft-cover
objective `∑_{i,j} ϕ(|A[i,j]| / (a[i]*b[j]))`. This is the asymmetric analog of
[`soft_symcover`](@ref).

Unlike [`cover`](@ref), there is no hard coverage constraint: `a[i]*b[j]` may be less than
`|A[i,j]|`, with violations penalized by `ϕ`.

Supported penalty functions:
- `AbsLinear{2}()` (default): in the inverse-scale variables `u = 1 ./ a`, `v = 1 ./ b`, the
  objective `∑_{i,j∈S} (1 - |A[i,j]| u[i] v[j])²` (sum over the nonzero support `S`) is
  biconvex, so alternating least squares with the closed-form half-sweeps

      u[i] = ∑_j |A[i,j]| v[j] / ∑_j (|A[i,j]| v[j])²   (dually for v[j])

  is monotone, stopping when the relative objective decrease drops below `1e-14` or after
  `iter` sweeps.
- `AbsLinear{1}()`: initializes from the `AbsLinear{2}()` result, then refines by alternating
  weighted-median updates — each row/column block is minimized exactly, so the descent is
  monotone. Its flat basins are broken by a deterministic lower-median tie-break, giving a
  scale-covariant representative.

Rows or columns of `A` that are entirely zero receive scale `0`.

The objective is non-convex, so `starts` starting points are tried and the lowest-objective
result kept: the geometric-mean init, the tightened hard cover [`cover`](@ref), and — for the
remaining starts — multiplicative log-normal perturbations `a .* exp.(σ .* ξ)`, with spread
`σ`, of the geometric-mean point, `ξ` drawn from `rng`. Every start co-varies with an
independent row/column rescaling of `A` and the objective is scale-invariant, so the selection
is scale-covariant. The default `rng` is a fresh `MersenneTwister(0)` per call, making repeated
calls (and the two frames of a covariance check) agree; pass your own `rng` for reproducibility
you control, since default RNG streams are not stable across Julia versions.

See also: [`cover`](@ref), [`soft_symcover`](@ref), [`cover_objective`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{4})\\d+" => s"\\1"
julia> A = [1 2 3; 6 5 4];

julia> a, b = soft_cover(A);

julia> a * b'
2×3 Matrix{Float64}:
 1.93288  1.97239  2.50673
 4.97144  5.07307  6.44741
```
"""
soft_cover(A::AbstractMatrix; iter::Int=100, starts::Int=8, σ::Real=2.0,
           rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED)) =
    soft_cover(AbsLinear{2}(), A; iter, starts, σ, rng)

function soft_cover(ϕ::AbsLinear{2}, A::AbstractMatrix; iter::Int=100, starts::Int=8, σ::Real=2.0,
                    rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    return _soft_cover_abslinear2(A, iter, starts, σ, rng)
end

function soft_cover(ϕ::AbsLinear{1}, A::AbstractMatrix; iter::Int=100, starts::Int=8, σ::Real=2.0,
                    rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    # Spend the multistart on the AbsLinear{2} cover (a good basin selector), then refine with
    # the AbsLinear{1} weighted-median descent.
    a, b = soft_cover(AbsLinear{2}(), A; iter=5, starts, σ, rng)
    _abslinear1_iter_asym!(a, b, A, iter)
    return a, b
end

# Labeled `(a, b)` candidate starts for the asymmetric AbsLinear{2} multistart, in selection
# order. Deterministic starts: the geometric-mean init `cover(A; iter=0)` (also the perturbation
# base) and the tightened hard cover `cover(A)`, obtained by tightening a copy of the
# geometric-mean init so the shared passes run once. Remaining slots, up to `starts` total, are
# multiplicative log-normal perturbations `a_g .* exp.(σ .* ξ)`, `b_g .* exp.(σ .* η)` of the
# geometric-mean point, `ξ`/`η` drawn from `rng` (drawn for every index so the stream is
# frame-independent).
function _soft_cover_abslinear2_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
    T = float(eltype(A))
    ag, bg = cover(A; iter=0)   # geometric-mean init (boosted, untightened); perturbation base
    labels = ["geomean"]
    inits = [(copy(ag), copy(bg))]
    # The tightened hard cover `cover(A)` differs from `(ag, bg)` only by its tightening
    # iterations, so tighten a copy rather than recomputing the shared geometric-mean and
    # feasibility passes. `iter=3` matches `cover`'s default.
    length(inits) < starts && (push!(labels, "hardcover"); push!(inits, tighten_cover!(copy(ag), copy(bg), A; iter=3)))
    k = 0
    while length(inits) < starts
        a = similar(ag); b = similar(bg)
        for i in axes(A, 1)
            ξ = randn(rng)
            a[i] = ag[i] > 0 ? ag[i] * exp(T(σ) * T(ξ)) : zero(T)
        end
        for j in axes(A, 2)
            η = randn(rng)
            b[j] = bg[j] > 0 ? bg[j] * exp(T(σ) * T(η)) : zero(T)
        end
        k += 1; push!(labels, "rand$k"); push!(inits, (a, b))
    end
    return labels, inits
end

# Scale-covariant multistart for the asymmetric AbsLinear{2} soft cover. Runs the single-start
# alternating least squares `_mscm_als!` from the candidate list built by
# `_soft_cover_abslinear2_inits` and returns the pair `_multistart_select` picks. Every start
# co-varies with an independent row/column rescaling `D_r*A*D_c` and the objective is
# scale-invariant, so the selection is scale-covariant; passing the same `rng` state across the
# two frames (as the default fresh-seeded RNG does) makes it reproducible.
#
# As for `_soft_symcover_abslinear2`, passing `labels`/`objs` as empty vectors fills them in
# place with every candidate's label and final objective, so the winner is
# `labels[_multistart_select(objs)]`.
function _soft_cover_abslinear2(A::AbstractMatrix, iter::Int, starts::Int, σ::Real, rng;
                                labels=nothing, objs=nothing)
    labs, inits = _soft_cover_abslinear2_inits(A, starts, σ, rng)
    E = map(inits) do (a, b)
        _mscm_als!(a, b, A, iter)
        cover_objective(AbsLinear{2}(), a, b, A)
    end
    labels === nothing || append!(labels, labs)
    objs === nothing || append!(objs, E)
    return inits[_multistart_select(E)]
end

# Alternating least squares for the AbsLinear{2} soft cover in the inverse-scale variables
# u = 1 ./ a, v = 1 ./ b. With M = |A| restricted to its nonzero support, the objective
# E = ∑ (1 - M[i,j] u[i] v[j])² is biconvex; each half-sweep sets u[i] (resp. v[j]) to its
# exact minimizer. Rows/columns with empty support keep scale 0 and are held fixed.
# Refines `a`, `b` in place starting from their incoming values.
function _mscm_als!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, iter::Int;
                    tol=1e-14)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axr) vs $(eachindex(a))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(axc) vs $(eachindex(b))"))
    T = float(promote_type(eltype(a), eltype(b), eltype(A)))
    # Invert to inverse-scale variables; empty-support rows/columns (scale 0) stay at 0.
    u = map(x -> x > 0 ? inv(T(x)) : zero(T), a)
    v = map(x -> x > 0 ? inv(T(x)) : zero(T), b)
    E = _mscm_objective(A, u, v)
    for _ in 1:iter
        for i in axr
            num = den = zero(T)
            for j in axc
                Aij = abs(T(A[i, j]))
                iszero(Aij) && continue
                Av = Aij * v[j]
                num += Av
                den += Av * Av
            end
            den > 0 && (u[i] = num / den)
        end
        for j in axc
            num = den = zero(T)
            for i in axr
                Aij = abs(T(A[i, j]))
                iszero(Aij) && continue
                Au = Aij * u[i]
                num += Au
                den += Au * Au
            end
            den > 0 && (v[j] = num / den)
        end
        Enew = _mscm_objective(A, u, v)
        E - Enew <= tol * max(E, one(T)) && (E = Enew; break)
        E = Enew
    end
    for i in axr
        a[i] = iszero(u[i]) ? zero(eltype(a)) : inv(u[i])
    end
    for j in axc
        b[j] = iszero(v[j]) ? zero(eltype(b)) : inv(v[j])
    end
    return a, b
end

# Soft-cover objective in inverse-scale variables: ∑_{i,j: A[i,j]≠0} (1 - |A[i,j]| u[i] v[j])².
function _mscm_objective(A::AbstractMatrix, u::AbstractVector, v::AbstractVector)
    T = float(promote_type(eltype(A), eltype(u), eltype(v)))
    E = zero(T)
    for j in axes(A, 2)
        vj = v[j]
        for i in axes(A, 1)
            Aij = abs(T(A[i, j]))
            iszero(Aij) && continue
            r = one(T) - Aij * u[i] * vj
            E += r * r
        end
    end
    return E
end

# Alternating weighted-median descent for the asymmetric AbsLinear{1} soft cover.
# Updating a[i] with b fixed minimizes ∑_j |1 - |A[i,j]|/(a[i] b[j])| over a[i] > 0, whose
# minimizer is the weighted median of c_j = |A[i,j]|/b[j] weighted by the same c_j (see
# `_weighted_self_median!`); the b-update is dual. There is no self-coupled diagonal term (the
# (i,i) entry enters the row update through b[i] like any other column), so each full sweep is
# an exact block minimization and the objective decreases monotonically. Refines `a`, `b` in
# place from their incoming values; exits early once the largest relative coordinate movement
# in a sweep drops to `tol` (scale-invariant, so covariant restarts of a rescaled problem exit
# on the same sweep). Rows/columns with empty support keep scale 0.
function _abslinear1_iter_asym!(a::AbstractVector{T}, b::AbstractVector{T}, A::AbstractMatrix,
                                iter::Int; tol::Real=1e-12) where T
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axr) vs $(eachindex(a))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(axc) vs $(eachindex(b))"))
    bufc = Vector{T}(undef, length(axc))   # c_j buffer for an a-row update
    bufr = Vector{T}(undef, length(axr))   # c_i buffer for a b-column update
    for _ in 1:iter
        maxrel = zero(T)
        for i in axr
            nc = 0
            for j in axc
                Aij = T(abs(A[i, j]))
                iszero(Aij) && continue
                bj = b[j]
                iszero(bj) && continue
                nc += 1
                bufc[nc] = Aij / bj
            end
            x = nc == 0 ? zero(T) : _weighted_self_median!(view(bufc, 1:nc))
            ai  = a[i]
            den = max(abs(x), abs(ai))
            iszero(den) || (maxrel = max(maxrel, abs(x - ai) / den))
            a[i] = x
        end
        for j in axc
            nc = 0
            for i in axr
                Aij = T(abs(A[i, j]))
                iszero(Aij) && continue
                ai = a[i]
                iszero(ai) && continue
                nc += 1
                bufr[nc] = Aij / ai
            end
            x = nc == 0 ? zero(T) : _weighted_self_median!(view(bufr, 1:nc))
            bj  = b[j]
            den = max(abs(x), abs(bj))
            iszero(den) || (maxrel = max(maxrel, abs(x - bj) / den))
            b[j] = x
        end
        maxrel <= T(tol) && break
    end
    return a, b
end

# ============================================================
# tighten_cover!
# ============================================================

function tighten_cover!(a::AbstractVector{T}, A::AbstractMatrix; iter::Int=3) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        # A is assumed symmetric and only the upper triangle is read; each pair's
        # ratio updates both endpoints, so this sees the same multiset of ratios
        # as a full-grid sweep at half the cost.
        for j in eachindex(a)
            aratioj, aj = aratio[j], a[j]
            for i in first(ax):j
                Aij = T(abs(A[i, j]))
                r = ifelse(iszero(Aij), typemax(T), a[i] * aj / Aij)
                aratio[i] = min(aratio[i], r)
                aratioj   = min(aratioj, r)
            end
            aratio[j] = aratioj
        end
        a ./= sqrt.(aratio)
    end
    return a
end

function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::AbstractMatrix; iter::Int=3) where T
    eachindex(a) == axes(A, 1) || throw(DimensionMismatch("indices of a must match row-indexing of A"))
    eachindex(b) == axes(A, 2) || throw(DimensionMismatch("indices of b must match column-indexing of A"))
    aratio = fill(typemax(T), eachindex(a))
    bratio = fill(typemax(T), eachindex(b))
    for _ in 1:iter
        fill!(aratio, typemax(T))
        fill!(bratio, typemax(T))
        for j in eachindex(b)
            bratioj, bj = bratio[j], b[j]
            for i in eachindex(a)
                Aij = T(abs(A[i, j]))
                r   = ifelse(iszero(Aij), typemax(T), a[i] * bj / Aij)
                aratio[i] = min(aratio[i], r)
                bratioj   = min(bratioj, r)
            end
            bratio[j] = bratioj
        end
        a ./= sqrt.(aratio)
        b ./= sqrt.(bratio)
    end
    return a, b
end

# ============================================================
# Native AbsLog{2} MCM solver
# ============================================================

# Inner linear solve for the AbsLog{2} MCM Newton steps. `:auto` (the default)
# forms and factorizes the reweighted normal equations densely, which is fastest
# for dense supports: an LAPACK Cholesky beats the matrix-free path because each
# LSQR iteration costs O(nnz) = O(n²) there. `:lsqr` forces the matrix-free path,
# whose per-iteration cost is O(nnz); it is the intended solve for large sparse
# supports (where nnz ≪ n²) and is used by the structured/sparse methods.

# Matrix-free LSQR (Paige & Saunders) for the weighted least-squares problem
# `min ‖M x - b‖` underlying the reweighted normal equations `MᵀM x = Mᵀb`.
# `Amul!(y, x)` overwrites `y` with `M*x`; `Atmul!(z, y)` overwrites `z` with
# `Mᵀ*y`. Warm-started from `x0`. LSQR is used in preference to CG on the normal
# equations because it works with the condition number of `M` (≈ √κ at penalty
# strength κ) rather than that of `MᵀM` (≈ κ); at κ = 1e8 the squared conditioning
# breaks CG while LSQR stays accurate.
#
# The penalty least-squares problem is inconsistent (its optimal residual is
# nonzero), so the stopping test is on the normal-equations residual
# ‖Mᵀ(b - Mx)‖ ≤ atol · ‖M‖ · ‖b - Mx‖, both estimated from the bidiagonalization
# scalars (‖Mᵀr‖ = ϕbar·α·|c|, ‖r‖ = ϕbar, ‖M‖ from the Frobenius norm of the
# bidiagonal). Returns `(x, iters)`.
function _lsqr(Amul!, Atmul!, b::AbstractVector{T}, x0::AbstractVector{T};
               atol=1e-12, maxiter::Int=2 * (length(b) + length(x0)) + 100) where {T}
    x = copy(x0)
    u = similar(b)
    Amul!(u, x)
    @. u = b - u
    β = norm(u)
    β > 0 && (u ./= β)
    v = similar(x0)
    Atmul!(v, u)
    α = norm(v)
    α > 0 && (v ./= α)
    w = copy(v)
    tmpm = similar(u)
    tmpn = similar(v)
    ϕbar = β
    ρbar = α
    anorm2 = α^2          # Frobenius norm² of the lower bidiagonal ≈ ‖M‖²
    (iszero(β) || iszero(α)) && return x, 0   # x0 already optimal
    iters = 0
    for k in 1:maxiter
        iters = k
        # Golub-Kahan bidiagonalization step.
        Amul!(tmpm, v)
        @. u = tmpm - α * u
        β = norm(u)
        β > 0 && (u ./= β)
        Atmul!(tmpn, u)
        @. v = tmpn - β * v
        α = norm(v)
        α > 0 && (v ./= α)
        anorm2 += β^2 + α^2
        # Orthogonal transformation applied to the bidiagonal system.
        ρ = hypot(ρbar, β)
        iszero(ρ) && break
        c = ρbar / ρ
        s = β / ρ
        θ = s * α
        ρbar = -c * α
        ϕ = c * ϕbar
        ϕbar = s * ϕbar
        @. x += (ϕ / ρ) * w
        @. w = v - (θ / ρ) * w
        # Stop when the normal-equations residual is negligible relative to ‖M‖‖r‖,
        # or when the least-squares residual itself has vanished (consistent system).
        arnorm = ϕbar * α * abs(c)
        rnorm = abs(ϕbar)
        (arnorm <= atol * sqrt(anorm2) * rnorm || iszero(rnorm) || iszero(β)) && break
    end
    return x, iters
end

# Symmetric AbsLog{2} hard cover via a one-sided quadratic penalty on the
# log-residuals z_ij = α_i + α_j - log|A_ij| (α = log a):
#
#   f_κ(α) = ∑_{ij ∈ support} w(z_ij) z_ij²,   w = 1 for z ≥ 0, κ for z < 0.
#
# As κ → ∞ the minimizer approaches the constrained (hard-cover) optimum. Each κ
# stage runs a damped semismooth Newton iteration: freeze the weights at the
# current α, solve the reweighted normal equations `B α = f` (an SDD system with
# the sparsity of the nonzero-pattern graph), and take a backtracking line
# search toward that point (which ensures convergence). A final uniform shift
# makes the cover exactly feasible.
function symcover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, _ = _symcover_min_abslog2(A; kwargs...)
    return a
end

# Worker for `symcover_min(::AbsLog{2})`. Returns `(a, stats)` where `stats` is a
# NamedTuple `(; nsolves, lsqriters, linsolve)` recording the number of inner linear
# solves, the total LSQR iterations (0 on the dense path), and which path ran — used
# by the benchmarks. `linsolve` is `:auto`/`:dense` (dense factorization) or `:lsqr`
# (matrix-free, for sparse supports).
function _symcover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                               maxouter::Int=40, linsolve::Symbol=:auto)
    linsolve in (:auto, :dense, :lsqr) ||
        throw(ArgumentError("linsolve must be :auto, :dense, or :lsqr; got :$linsolve"))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    T = float(eltype(A))
    n = length(ax)
    use_lsqr = linsolve === :lsqr
    # log|A| on the support S; the Newton solve runs on 1-based positions 1:n and
    # is scattered back onto `a` through `ax` so `A`'s own axes are honored.
    C = zeros(T, n, n)
    S = falses(n, n)
    for (jp, j) in enumerate(ax), (ip, i) in enumerate(ax)
        Aij = abs(T(A[i, j]))
        iszero(Aij) && continue
        C[ip, jp] = log(Aij)
        S[ip, jp] = true
    end
    hassupp = [any(@view S[ip, :]) for ip in 1:n]
    # Support entries, one per residual z_ij = α_i + α_j - log|A_ij|.
    edges = Tuple{Int,Int}[]
    for jp in 1:n, ip in 1:n
        S[ip, jp] && push!(edges, (ip, jp))
    end
    ne = length(edges)
    fκ = function (α, κ)
        v = zero(T)
        for (ip, jp) in edges
            z = α[ip] + α[jp] - C[ip, jp]
            v += (z < 0 ? T(κ) : oneunit(T)) * z^2
        end
        return v
    end
    # Each Newton step freezes the weights at the current α and solves the reweighted
    # least-squares problem `min ‖√W (Rα - c)‖`, `(Rα)_e = α_i + α_j`, whose normal
    # equations are the signless Laplacian system `B α = f`. The dense path forms and
    # factorizes `B` (a support-free variable gets an identity row; a minimal
    # scale-relative ridge lifts the bipartite gauge null space, e.g. the `[0 1; 1 0]`
    # support graph whose signless Laplacian is singular). The LSQR path applies `√W R`
    # and its transpose matrix-free and warm-starts from the incoming iterate; it
    # solves the least-squares form directly, so its accuracy tracks the conditioning
    # of `√W R` (≈ √κ) rather than that of `B` (≈ κ).
    ws = zeros(T, ne)   # √weight per support entry, frozen during one solve
    cv = zeros(T, ne)   # √weight · log|A_ij| (LSQR right-hand side)
    W = zeros(T, n, n)  # weights (dense path)
    f = zeros(T, n)
    nsolves = Ref(0)
    nlsqr = Ref(0)
    solve_weighted = function (α, κ)
        nsolves[] += 1
        if use_lsqr
            for (e, (ip, jp)) in enumerate(edges)
                w = κ === nothing ? oneunit(T) : ((α[ip] + α[jp] - C[ip, jp]) < 0 ? T(κ) : oneunit(T))
                sw = sqrt(w)
                ws[e] = sw
                cv[e] = sw * C[ip, jp]
            end
            Amul! = function (y, x)
                for (e, (ip, jp)) in enumerate(edges)
                    y[e] = ws[e] * (x[ip] + x[jp])
                end
                return y
            end
            Atmul! = function (z, y)
                fill!(z, zero(T))
                for (e, (ip, jp)) in enumerate(edges)
                    t = ws[e] * y[e]
                    z[ip] += t
                    z[jp] += t
                end
                return z
            end
            sol, it = _lsqr(Amul!, Atmul!, cv, α)
            nlsqr[] += it
            return sol
        else
            fill!(W, zero(T))
            fill!(f, zero(T))
            for (ip, jp) in edges
                w = κ === nothing ? oneunit(T) : ((α[ip] + α[jp] - C[ip, jp]) < 0 ? T(κ) : oneunit(T))
                W[ip, jp] = w
                f[ip] += w * C[ip, jp]
            end
            B = zeros(T, n, n)
            for (ip, jp) in edges
                w = W[ip, jp]
                B[ip, ip] += w
                B[ip, jp] += w
            end
            # Minimal scale-relative ridge, sized by the largest diagonal, lifts the
            # bipartite gauge null space; support-free variables get an identity row.
            dmax = zero(T)
            for ip in 1:n
                dmax = max(dmax, B[ip, ip])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for ip in 1:n
                B[ip, ip] += hassupp[ip] ? ridge : oneunit(T)
            end
            return Symmetric(B) \ f
        end
    end
    α = solve_weighted(zeros(T, n), nothing)
    for κ in κs
        fcur = fκ(α, κ)
        for _ in 1:maxouter
            αnew = solve_weighted(α, κ)
            t = one(T)
            fnew = fκ(αnew, κ)
            while fnew > fcur && t > 1e-10
                t /= 2
                fnew = fκ(α .+ t .* (αnew .- α), κ)
            end
            α = α .+ t .* (αnew .- α)
            fcur - fnew <= 1e-12 * max(fcur, one(T)) && break
            fcur = fnew
        end
    end
    # Uniform boost to exact feasibility: α_i + α_j ≥ log|A_ij| for all support.
    γ = zero(T)
    for jp in 1:n, ip in 1:n
        S[ip, jp] || continue
        γ = max(γ, (C[ip, jp] - α[ip] - α[jp]) / 2)
    end
    a = similar(A, T, ax)
    for (ip, i) in enumerate(ax)
        a[i] = hassupp[ip] ? exp(α[ip] + γ) : zero(T)
    end
    return a, (; nsolves=nsolves[], lsqriters=nlsqr[], linsolve=(use_lsqr ? :lsqr : :dense))
end

# Asymmetric AbsLog{2} hard cover via the same one-sided quadratic penalty as
# `symcover_min`, on stacked log-scales x = (α; β) (α = log a over rows, β = log b
# over columns) with residuals z_ij = α_i + β_j - log|A_ij|. The row and column
# scales share a gauge freedom (α_i, β_j) → (α_i + s, β_j - s) that leaves every
# residual unchanged; during the solve it is fixed by adding v0*v0ᵀ,
# v0 = [ones(m); -ones(n)], to the normal equations, and afterwards the result is
# shifted along that gauge to the balance convention ∑ nzaᵢ αᵢ = ∑ nzbⱼ βⱼ
# (nzaᵢ, nzbⱼ = nonzero counts of row i, column j) so it is deterministic.
function cover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, b, _ = _cover_min_abslog2(A; kwargs...)
    return a, b
end

# Worker for `cover_min(::AbsLog{2})`. Returns `(a, b, stats)` with `stats` a
# NamedTuple `(; nsolves, lsqriters, linsolve)` (see `_symcover_min_abslog2`).
function _cover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                            maxouter::Int=40, linsolve::Symbol=:auto)
    linsolve in (:auto, :dense, :lsqr) ||
        throw(ArgumentError("linsolve must be :auto, :dense, or :lsqr; got :$linsolve"))
    axr = axes(A, 1)
    axc = axes(A, 2)
    T = float(eltype(A))
    m = length(axr)
    n = length(axc)
    N = m + n
    use_lsqr = linsolve === :lsqr
    # log|A| on the support S; internal positions 1:m index rows, m+1:m+n index
    # columns, and results are scattered back through axr/axc so A's axes are honored.
    C = zeros(T, m, n)
    S = falses(m, n)
    for (jp, j) in enumerate(axc), (ip, i) in enumerate(axr)
        Aij = abs(T(A[i, j]))
        iszero(Aij) && continue
        C[ip, jp] = log(Aij)
        S[ip, jp] = true
    end
    hasrow = [any(@view S[ip, :]) for ip in 1:m]
    hascol = [any(@view S[:, jp]) for jp in 1:n]
    # Gauge vector: ±1 on supported variables, 0 on support-free ones (which carry
    # no constraint and are decoupled with an identity row in `solve_weighted`).
    v0 = zeros(T, N)
    for ip in 1:m
        hasrow[ip] && (v0[ip] = one(T))
    end
    for jp in 1:n
        hascol[jp] && (v0[m+jp] = -one(T))
    end
    # Support entries as edges linking a row position ip to a column position m+jp.
    edges = Tuple{Int,Int}[]
    for jp in 1:n, ip in 1:m
        S[ip, jp] && push!(edges, (ip, m + jp))
    end
    ne = length(edges)
    fκ = function (x, κ)
        v = zero(T)
        for (p, q) in edges
            z = x[p] + x[q] - C[p, q-m]
            v += (z < 0 ? T(κ) : oneunit(T)) * z^2
        end
        return v
    end
    # Each Newton step solves the reweighted least-squares problem for the stacked
    # scales x = (α; β), residuals z_ij = α_i + β_j - log|A_ij|. Row and column scales
    # share the (e; −e) gauge; both paths pin it. The dense path adds the rank-1 term
    # v0*v0ᵀ to the normal equations `B x = f` and factorizes (support-free variables
    # get an identity row). The LSQR path appends one gauge row `v0ᵀ x = 0` to the
    # least-squares system so `√W R` has full column rank, applies it matrix-free, and
    # warm-starts from the incoming iterate. After the solve a closed-form shift moves
    # the result to the balance convention, so the pinned gauge is not observable.
    W = zeros(T, m, n)      # per-support weights (dense path)
    f = zeros(T, N)
    ws = zeros(T, ne)       # √weight per support entry (LSQR path)
    cv = zeros(T, ne + 1)   # √weight · log|A_ij|, with a trailing 0 gauge target
    nsolves = Ref(0)
    nlsqr = Ref(0)
    solve_weighted = function (x, κ)
        nsolves[] += 1
        if use_lsqr
            for (e, (p, q)) in enumerate(edges)
                c = C[p, q-m]
                w = κ === nothing ? oneunit(T) : ((x[p] + x[q] - c) < 0 ? T(κ) : oneunit(T))
                sw = sqrt(w)
                ws[e] = sw
                cv[e] = sw * c
            end
            g = ne + 1   # index of the appended gauge row
            Amul! = function (y, xx)
                for (e, (p, q)) in enumerate(edges)
                    y[e] = ws[e] * (xx[p] + xx[q])
                end
                y[g] = dot(v0, xx)
                return y
            end
            Atmul! = function (z, y)
                fill!(z, zero(T))
                for (e, (p, q)) in enumerate(edges)
                    t = ws[e] * y[e]
                    z[p] += t
                    z[q] += t
                end
                @. z += v0 * y[g]
                return z
            end
            sol, it = _lsqr(Amul!, Atmul!, cv, x)
            nlsqr[] += it
            return sol
        else
            fill!(W, zero(T))
            fill!(f, zero(T))
            for (p, q) in edges
                jp = q - m
                c = C[p, jp]
                w = κ === nothing ? oneunit(T) : ((x[p] + x[q] - c) < 0 ? T(κ) : oneunit(T))
                W[p, jp] = w
                f[p] += w * c
                f[q] += w * c
            end
            B = v0 * v0'
            for (p, q) in edges
                w = W[p, q-m]
                B[p, p] += w
                B[q, q] += w
                B[p, q] += w
                B[q, p] += w
            end
            # A support whose bipartite graph splits into k connected components carries k
            # independent (e; −e) gauges; v0*v0ᵀ pins only the global one, leaving k−1
            # singular directions. A minimal scale-relative ridge on the supported
            # diagonals lifts them (the same device the symmetric solver uses for the
            # bipartite null space). The RHS is orthogonal to every gauge null vector, so
            # the ridge leaves the recovered scales essentially unperturbed, and the
            # per-component gauge it fixes is unobservable — no product a_i·b_j spans two
            # components. Support-free variables get an identity row.
            dmax = zero(T)
            for p in 1:N
                dmax = max(dmax, B[p, p])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for ip in 1:m
                B[ip, ip] = hasrow[ip] ? B[ip, ip] + ridge : one(T)
            end
            for jp in 1:n
                q = m + jp
                B[q, q] = hascol[jp] ? B[q, q] + ridge : one(T)
            end
            return Symmetric(B) \ f
        end
    end
    x = solve_weighted(zeros(T, N), nothing)
    for κ in κs
        fcur = fκ(x, κ)
        for _ in 1:maxouter
            xnew = solve_weighted(x, κ)
            t = one(T)
            fnew = fκ(xnew, κ)
            while fnew > fcur && t > 1e-10
                t /= 2
                fnew = fκ(x .+ t .* (xnew .- x), κ)
            end
            x = x .+ t .* (xnew .- x)
            fcur - fnew <= 1e-12 * max(fcur, one(T)) && break
            fcur = fnew
        end
    end
    # Uniform boost to exact feasibility: α_i + β_j ≥ log|A_ij| on the support.
    γ = zero(T)
    for jp in 1:n, ip in 1:m
        S[ip, jp] || continue
        γ = max(γ, (C[ip, jp] - x[ip] - x[m+jp]) / 2)
    end
    for p in 1:N
        x[p] += γ
    end
    # Shift along the (e; -e) gauge to the balance convention ∑ nzaᵢ αᵢ = ∑ nzbⱼ βⱼ.
    nnz = count(S)
    Lα = zero(T)
    Lβ = zero(T)
    for ip in 1:m
        Lα += count(@view S[ip, :]) * x[ip]
    end
    for jp in 1:n
        Lβ += count(@view S[:, jp]) * x[m+jp]
    end
    s = nnz > 0 ? (Lβ - Lα) / (2 * nnz) : zero(T)
    a = similar(A, T, axr)
    b = similar(A, T, axc)
    for (ip, i) in enumerate(axr)
        a[i] = hasrow[ip] ? exp(x[ip] + s) : zero(T)
    end
    for (jp, j) in enumerate(axc)
        b[j] = hascol[jp] ? exp(x[m+jp] - s) : zero(T)
    end
    return a, b, (; nsolves=nsolves[], lsqriters=nlsqr[], linsolve=(use_lsqr ? :lsqr : :dense))
end


# ============================================================
# Extension function stubs
# (implementations live in SIAJuMP and SIAIpopt extensions)
# ============================================================

"""
    a = symcover_min(ϕ, A; kwargs...)

Return the ϕ-minimal symmetric hard cover of `A`: the vector `a` minimizing
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*a[j]))` subject to `a[i]*a[j] >= |A[i,j]|` for every nonzero
entry of `A`.

Supported ϕ values:
- `AbsLog{2}()`: solved natively (no external solver). Accepts keyword arguments
  `κs` (the penalty-continuation schedule, default `(1e2, 1e4, 1e6, 1e8)`),
  `maxouter` (Newton steps per stage, default `40`), and `linsolve` (the inner
  linear solve: `:auto`/`:dense` use a dense factorization of the reweighted
  normal equations; `:lsqr` uses matrix-free LSQR (per-iteration cost O(nnz),
  intended for large sparse supports)).
- `AbsLog{1}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt.

!!! note
    Even the native solver is more expensive than the [`symcover`](@ref) heuristic.

See also: [`cover_min`](@ref), [`symcover`](@ref).
"""
function symcover_min end

# Internal exact reference implemented by the SIAJuMP extension; used only to
# cross-check the native `symcover_min(::AbsLog{2})` in the test suite.
function symcover_min_jump end

"""
    a, b = cover_min(ϕ, A)

Return the ϕ-minimal asymmetric hard cover of `A`: the vectors `a`, `b` minimizing
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*b[j]))` subject to `a[i]*b[j] >= |A[i,j]|` for every nonzero
entry of `A`. The row/column scales are pinned to the balance convention
`∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` (`nzaᵢ`, `nzbⱼ` = nonzero counts of row `i`,
column `j`) so the result is deterministic.

Supported ϕ values:
- `AbsLog{2}()`: solved natively (no external solver). Accepts keyword arguments
  `κs` (the penalty-continuation schedule, default `(1e2, 1e4, 1e6, 1e8)`),
  `maxouter` (Newton steps per stage, default `40`), and `linsolve` (the inner
  linear solve: `:auto`/`:dense` use a dense factorization of the reweighted
  normal equations; `:lsqr` uses matrix-free LSQR (per-iteration cost O(nnz),
  intended for large sparse supports)).
- `AbsLog{1}()`: requires JuMP and HiGHS.

!!! note
    Even the native solver is more expensive than the [`cover`](@ref) heuristic.

See also: [`symcover_min`](@ref), [`cover`](@ref).
"""
function cover_min end

# Internal exact reference implemented by the SIAJuMP extension; used only to
# cross-check the native `cover_min(::AbsLog{2})` in the test suite.
function cover_min_jump end

"""
    a = soft_symcover_min(ϕ, A)

Return the ϕ-minimal symmetric soft cover of `A`: minimizes `∑_{i,j} ϕ(|A[i,j]|/(a[i]*a[j]))`
with no coverage constraints.

Supported ϕ values and required extensions:
- `AbsLog{2}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt.
"""
function soft_symcover_min end

# ============================================================
# Adjoint and Transpose wrappers
# ============================================================

cover_objective(ϕ, a, b, A::Adjoint)   = cover_objective(ϕ, b, a, parent(A))
cover_objective(ϕ, a, b, A::Transpose) = cover_objective(ϕ, b, a, parent(A))

function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::Adjoint; kwargs...) where T
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end
function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::Transpose; kwargs...) where T
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end

function cover(ϕ, A::Adjoint; kwargs...)
    a, b = cover(ϕ, parent(A); kwargs...)
    return b, a
end
function cover(ϕ, A::Transpose; kwargs...)
    a, b = cover(ϕ, parent(A); kwargs...)
    return b, a
end
