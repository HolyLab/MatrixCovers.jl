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
function unconstrained_min!(::AbsLog{2}, a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`unconstrained_min!(ϕ, a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a))"))
    loga = fill!(similar(a), zero(T))
    nza  = zeros(Int, ax)
    for j in ax
        for i in first(ax):j
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            lAij = log(Aij)
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
    N = sum(nza)
    iszero(N) && return zeros(Float64, length(nza))
    maxn = maximum(nza)
    λ = 2.0 * maxn          # initial guess: above all n_i so all (λ - n_i) > 0
    for _ in 1:20           # generous bound; quadratic convergence exits early
        s1 = sum(nza[i]^2 / (λ - nza[i]) for i in eachindex(nza) if !iszero(nza[i]))
        abs(s1 - N) < 1e-12 * N && break
        s2 = sum(nza[i]^2 / (λ - nza[i])^2 for i in eachindex(nza) if !iszero(nza[i]))
        iszero(s2) && break
        λ += (s1 - N) / s2
    end
    v = [iszero(nza[i]) ? 0.0 : nza[i] / (λ - nza[i]) for i in eachindex(nza)]
    nv = sqrt(sum(abs2, v))
    return iszero(nv) ? v : v ./ nv
end

# 1D search minimizing the AbsLinear{2} objective along the ray α₀ + t*v in log space.
# Uses a coarse grid scan followed by golden-section refinement.
function _abslinear2_linesearch(α₀::AbstractVector, v::AbstractVector, A::AbstractMatrix)
    ax = eachindex(α₀)
    function f(t)
        s = 0.0
        for j in ax
            for i in ax
                Aij = abs(A[i, j])
                # AbsLinear: zero entries contribute 1 regardless of t
                if iszero(Aij)
                    s += 1.0
                    continue
                end
                q = Aij * exp(-α₀[i] - t*v[i] - α₀[j] - t*v[j])
                s += (1 - q)^2
            end
        end
        return s
    end

    # Coarse scan
    best_t, best_f = 0.0, f(0.0)
    for t in (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
        ft = f(t)
        if ft < best_f
            best_f = ft
            best_t = t
        end
    end

    # Golden-section refinement around best_t
    ρ = 2.0 - (1.0 + sqrt(5.0)) / 2.0   # 2 - φ = 1/φ²
    a, b = best_t - 0.6, best_t + 0.6
    c, d = a + ρ*(b-a), b - ρ*(b-a)
    fc, fd = f(c), f(d)
    for _ in 1:30
        if fc < fd
            b = d; d = c; fd = fc
            c = a + ρ*(b-a); fc = f(c)
        else
            a = c; c = d; fc = fd
            d = b - ρ*(b-a); fd = f(d)
        end
        abs(b - a) < 1e-8 && break
    end
    return (a + b) / 2
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

See also: [`symcover_min`](@ref), [`soft_symcover`](@ref), [`cover`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [4 -1; -1 0];

julia> a = symcover(A)
2-element Vector{Float64}:
 2.0
 0.5

julia> a * a'
2×2 Matrix{Float64}:
 4.0  1.0
 1.0  0.25
```
"""
symcover(A::AbstractMatrix; iter::Int=3) = symcover(AbsLinear{2}(), A; iter)

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

function symcover(ϕ::AbsLinear, A::AbstractMatrix; iter::Int=3)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(A))
    a = similar(A, T, ax)
    _symcover_abslinear_init!(a, A)
    return tighten_cover!(a, A; iter)
end

# ============================================================
# soft_symcover
# ============================================================

"""
    a = soft_symcover(ϕ, A; iter=20)
    a = soft_symcover(A; iter=20)

Given a square matrix `A` assumed to be symmetric, return a vector `a` approximately
minimizing the soft-cover objective `∑_{i,j} ϕ(|A[i,j]| / (a[i]*a[j]))`.

Unlike [`symcover`](@ref), there is no hard coverage constraint: `a[i]*a[j]` may be
less than `|A[i,j]|`, with violations penalized by `ϕ`.

Supported penalty functions:
- `AbsLog{2}()`: returns the analytical unconstrained minimum (no iterations needed).
- `AbsLinear{2}()` (default): initializes from the AbsLog{2} minimum + eigenvector
  linesearch, then refines by coordinate descent.
- `AbsLinear{1}()`: same initialization, coordinate descent uses a weighted-median step.

See also: [`symcover`](@ref), [`cover_objective`](@ref), [`soft_symcover_min`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [4 -1; -1 0];

julia> a = soft_symcover(A)
2-element Vector{Float64}:
 2.0
 0.5

julia> a = soft_symcover([0 1; 1 0])
2-element Vector{Float64}:
 1.0
 1.0
```
"""
soft_symcover(A::AbstractMatrix; iter::Int=20) = soft_symcover(AbsLinear{2}(), A; iter)

function soft_symcover(ϕ::AbsLog{2}, A::AbstractMatrix)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    T = float(eltype(A))
    a = similar(A, T, ax)
    unconstrained_min!(ϕ, a, A)   # analytical minimum; no iterations needed
    return a
end

function soft_symcover(ϕ::AbsLinear{2}, A::AbstractMatrix; iter::Int=20)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    T = float(eltype(A))
    a = similar(A, T, ax)
    _symcover_abslinear_init!(a, A)
    _abslinear2_iter!(a, A, iter)
    return a
end

function soft_symcover(ϕ::AbsLinear{1}, A::AbstractMatrix; iter::Int=20)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    # Initialize from AbsLinear{2} solution (good starting point for AbsLinear{1})
    a = soft_symcover(AbsLinear{2}(), A; iter=5)
    _abslinear1_iter!(a, A, iter)
    return a
end

# Coordinate-descent iteration for AbsLinear{2} soft cover.
# Each coordinate a[k] is updated to the exact minimizer of
#   (1 - d/x²)² + ∑_{j≠k} (1 - c_j/x)²
# where d = |A[k,k]| and c_j = |A[k,j]|/a[j].
# Closed form when d=0 (x = s2/s1); Newton on a cubic otherwise.
function _abslinear2_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int) where T
    ax = eachindex(a)
    for _ in 1:iter
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
            if iszero(s1)
                a[k] = iszero(d) ? zero(T) : sqrt(d)
            elseif iszero(d)
                a[k] = s2 / s1
            else
                # Newton on cubic s1*x³ + (d - s2)*x² - d² = 0
                x = s2 / s1   # d=0 root; good starting point
                for _ in 1:8
                    fx  = s1*x^3 + (d - s2)*x^2 - d^2
                    fxp = 3*s1*x^2 + 2*(d - s2)*x
                    iszero(fxp) && break
                    x = max(x - fx/fxp, eps(T))
                end
                a[k] = x
            end
        end
    end
    return a
end

# Coordinate-descent iteration for AbsLinear{1} soft cover.
# Each coordinate a[k] is updated to minimize ∑_j |1 - |A[k,j]|/(a[k]*a[j])|.
# For the off-diagonal sum, the minimizer is the weighted median of c_j with weights c_j,
# where c_j = |A[k,j]|/a[j].  When A[k,k] ≠ 0 we also compare against sqrt(|A[k,k]|).
function _abslinear1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int) where T
    ax  = eachindex(a)
    buf = Vector{T}(undef, length(ax))   # reusable buffer for c_j values
    for _ in 1:iter
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
                a[k] = iszero(d) ? zero(T) : sqrt(d)
                continue
            end
            # Weighted median of buf[1:nc] with weights buf[1:nc]
            c = view(buf, 1:nc)
            sort!(c)
            total = sum(c)
            half  = total / 2
            wm    = c[1]
            cum   = zero(T)
            for ci in c
                cum += ci
                wm   = ci
                cum >= half && break
            end
            if iszero(d)
                a[k] = wm
            else
                # Compare objective at weighted median vs sqrt(d)
                sq_d = sqrt(d)
                obj_at(x) = abs(1 - d/x^2) + sum(abs(1 - ci/x) for ci in c)
                a[k] = obj_at(wm) <= obj_at(sq_d) ? wm : sq_d
            end
        end
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
# tighten_cover!
# ============================================================

function tighten_cover!(a::AbstractVector{T}, A::AbstractMatrix; iter::Int=3) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        for j in eachindex(a)
            aratioj, aj = aratio[j], a[j]
            for i in eachindex(a)
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
# Extension function stubs
# (implementations live in SIAJuMP and SIAIpopt extensions)
# ============================================================

"""
    a = symcover_min(ϕ, A)

Return the ϕ-minimal symmetric hard cover of `A`: minimizes `∑_{i,j} ϕ(|A[i,j]|/(a[i]*a[j]))`
subject to `a[i]*a[j] >= |A[i,j]|` for all `i,j`.

Supported ϕ values and required extensions:
- `AbsLog{1}()`, `AbsLog{2}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt.

!!! note
    This function is exact but slow. See [`symcover`](@ref) for a fast heuristic.
"""
function symcover_min end

"""
    a, b = cover_min(ϕ, A)

Return the ϕ-minimal asymmetric hard cover of `A`.
Currently supported for `AbsLog{1}()` and `AbsLog{2}()` (requires JuMP and HiGHS).

!!! note
    This function is exact but slow. See [`cover`](@ref) for a fast heuristic.
"""
function cover_min end

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
