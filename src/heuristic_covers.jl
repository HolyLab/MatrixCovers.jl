# Fast O(mn) hard-cover heuristics `symcover` and `cover`, together with the
# tightening and initialization routines they (and the soft covers) build on.

# ============================================================
# Public interface
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

# Adjoint/Transpose wrappers for cover.
function cover(ϕ, A::Adjoint; kwargs...)
    a, b = cover(ϕ, parent(A); kwargs...)
    return b, a
end
function cover(ϕ, A::Transpose; kwargs...)
    a, b = cover(ϕ, parent(A); kwargs...)
    return b, a
end

# ============================================================
# Internal helpers
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

# Adjoint/Transpose wrappers for tighten_cover!.
function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::Adjoint; kwargs...) where T
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end
function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::Transpose; kwargs...) where T
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end

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
