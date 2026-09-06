# Objective-minimal hard covers. `AbsLog{2}` is native; extensions provide the
# other penalties.

# ============================================================
# Public interface
# ============================================================

"""
    a = symcover_min(ϕ, A; kwargs...)
    a = symcover_min(A; kwargs...)

Return the ϕ-minimal symmetric hard cover of `A`: the vector `a` minimizing
`∑ ϕ(|A[i,j]|/(a[i]*a[j]))` subject to `a[i]*a[j] >= |A[i,j]|`. The
default penalty is `AbsLog{2}()`.

Supported ϕ values:
- `AbsLog{2}()`: native.
- `AbsLog{1}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()` and `AbsLinear{2}()`: require JuMP and Ipopt and return the
  best local minimum found from `strategies`.

`AbsLog` is convex in the log scales. If `AbsLog{1}` has multiple minima, the
method selects the one with the smallest `AbsLog{2}` objective.

# Extended help

The native solver accepts `κ` (initial augmented-Lagrangian penalty, default
`1e2`), `maxouter` (multiplier updates, default `32`; `0` returns the
unconstrained fit), `maxiter` (Newton steps per update), `fillbudget` (see
below), and `linsolve`:

- `:dense` factorizes dense normal equations at O(n³) per Newton step.
- `:woodbury` handles nearly dense `Float64` support as a sparse correction. It
  requires at most `n ÷ 4` missing entries per row and `4n` in total.
- `:lsqr` is matrix-free with O(nnz) work per iteration and is the sparse-matrix
  default.
- `:auto` chooses `:woodbury` when supported, `:lsqr` when the stored support
  fills at most a quarter of the grid, and `:dense` otherwise.

The solver increases `κ` when the KKT residual contracts slowly. Statistics
include the penalty weights (`κs`) and KKT residuals (`kkt`).

For `Float64`, `:lsqr` uses a Cholesky preconditioner when its predicted storage
does not exceed `fillbudget` bytes (default `2^30`). Otherwise it uses a diagonal
preconditioner. The returned statistics identify the choice as `precond`.

If the solver warns that the result may not minimize the objective, increase
`maxouter` or, rarely, `κ`.

The native solver computes in `Float64` for narrower input types, then converts
the result to the required element type.

See also: [`cover_min`](@ref), [`symcover`](@ref), [`symcover_min!`](@ref).
"""
function symcover_min end
symcover_min(A::AbstractMatrix; kwargs...) = symcover_min(AbsLog{2}(), A; kwargs...)

"""
    a, b = cover_min(ϕ, A)
    a, b = cover_min(A)

Return the ϕ-minimal asymmetric hard cover of `A`: vectors `a`, `b` minimizing
`∑ ϕ(|A[i,j]|/(a[i]*b[j]))` subject to `a[i]*b[j] >= |A[i,j]|`. The
default penalty is `AbsLog{2}()`. Within each support component, the factors use
the balance convention
`∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` (`nzaᵢ`, `nzbⱼ` = nonzero counts of row `i`,
column `j`).

Supported ϕ values:
- `AbsLog{2}()`: native.
- `AbsLog{1}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()` and `AbsLinear{2}()`: require JuMP and Ipopt and return the
  best local minimum found from `strategies`.

`AbsLog` is convex in the log scales. If `AbsLog{1}` has multiple minima, the
method selects the one with the smallest `AbsLog{2}` objective.

# Extended help

The native solver accepts the same `κ`, `maxouter`, `maxiter`, `fillbudget`, and
`linsolve` keywords as [`symcover_min`](@ref). For `:woodbury`, an `m × n`
matrix may omit at most
`min(m,n) ÷ 4` entries per row or column and `4 * max(m,n)` entries in total.
`:dense` costs O((m+n)³) per Newton step; sparse matrices default to `:lsqr`.

See also: [`symcover_min`](@ref), [`cover`](@ref), [`cover_min!`](@ref).
"""
function cover_min end
cover_min(A::AbstractMatrix; kwargs...) = cover_min(AbsLog{2}(), A; kwargs...)

"""
    a = symcover_min!(ϕ, a, A; kwargs...)
    a = symcover_min!(a, A; kwargs...)

Refine the symmetric hard cover `a` in place. The no-ϕ form uses `AbsLog{2}()`;
supported penalties and keywords match [`symcover_min`](@ref).

`a` must cover `A` and be positive on supported rows. Unsupported scales are
ignored on input and set to zero. Use [`initialize_symcover`](@ref) or
[`symcover`](@ref) to construct a start.

The `AbsLog` result is independent of the start. For `AbsLog{1}`, ties are broken
by the `AbsLog{2}` objective. Local minima under `AbsLinear` can depend on the
start.

See also: [`initialize_symcover`](@ref), [`symcover_min`](@ref), [`cover_min!`](@ref).
"""
function symcover_min! end
symcover_min!(a::AbstractVector, A::AbstractMatrix; kwargs...) =
    symcover_min!(AbsLog{2}(), a, A; kwargs...)

"""
    a, b = cover_min!(ϕ, a, b, A; kwargs...)
    a, b = cover_min!(a, b, A; kwargs...)

Refine the asymmetric hard cover `(a, b)` in place. The start must cover `A` and
be positive on supported rows and columns; unsupported scales are zeroed. The
no-ϕ form uses `AbsLog{2}()`; supported penalties and keywords match
[`cover_min`](@ref).

Equivalent starts `(c*a, b/c)` give the same balanced result.

See also: [`initialize_cover`](@ref), [`cover_min`](@ref), [`symcover_min!`](@ref).
"""
function cover_min! end
cover_min!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    cover_min!(AbsLog{2}(), a, b, A; kwargs...)

# Symmetric `AbsLog{2}` hard cover using a one-sided quadratic penalty on
# `z_ij = α_i + α_j - log|A_ij|`:
#
#   f_κ(α) = ∑_{ij ∈ support} w(z_ij) z_ij²,   w = 1 for z ≥ 0, κ for z < 0.
#
# Each `κ` stage freezes the weights, solves the normal equations, and uses a
# backtracking line search. A final uniform shift restores feasibility.
function symcover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, _ = _symcover_min_abslog2(A; kwargs...)
    return a
end

# Asymmetric counterpart on stacked log scales `(α; β)`. The solve pins the
# global gauge; `_cover_min_abslog2` regularizes and balances the remaining
# component gauges.
function cover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, b, _ = _cover_min_abslog2(A; kwargs...)
    return a, b
end

# The convex objective has the same result from every valid start.
function symcover_min!(::AbsLog{2}, a::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_symcover_start!(a, A)
    anew, _ = _symcover_min_abslog2(A; start=a, kwargs...)
    a .= anew
    return a
end

function cover_min!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_cover_start!(a, b, A)
    anew, bnew, _ = _cover_min_abslog2(A; start=(a, b), kwargs...)
    a .= anew
    b .= bnew
    return a, b
end

# Multistart drivers for the `AbsLinear` kernels in MatrixCoversIpoptExt.
function symcover_min(ϕ::AbsLinear, A::AbstractMatrix; strategies=SYMCOVER_MIN_STRATEGIES)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    isempty(strategies) &&
        throw(ArgumentError("symcover_min: `strategies` must name at least one starting cover"))
    T = float(real(eltype(A)))
    starts = [similar(Array{T}, ax) for _ in strategies]
    # Skip strategies that cannot produce a start for `A`.
    built = [_initialize_symcover!(a, A, strategy, :inflate) for (a, strategy) in zip(starts, strategies)]
    covers = [symcover_min!(ϕ, a, A) for (a, ok) in zip(starts, built) if ok]
    isempty(covers) &&
        throw(ArgumentError("symcover_min: no strategy in $(string(strategies)) yields a starting cover of `A`"))
    return covers[_multistart_select([cover_objective(ϕ, a, A) for a in covers])]
end

function cover_min(ϕ::AbsLinear, A::AbstractMatrix; strategies=COVER_MIN_STRATEGIES)
    isempty(strategies) &&
        throw(ArgumentError("cover_min: `strategies` must name at least one starting cover"))
    covers = [initialize_cover(A; strategy) for strategy in strategies]
    for (a, b) in covers
        cover_min!(ϕ, a, b, A)
    end
    return covers[_multistart_select([cover_objective(ϕ, a, b, A) for (a, b) in covers])]
end

# ============================================================
# Internal helpers
# ============================================================

# Roundoff allowance when validating log-domain heuristic starts.
const START_FEASIBILITY_ULPS = 64

_start_slack(lv::T, li::T, lj::T) where {T} =
    START_FEASIBILITY_ULPS * eps(T) * max(oneunit(T), abs(lv), abs(li), abs(lj))

# Validate and normalize a symmetric hard-cover start.
function _prepare_symcover_start!(a::AbstractVector, A::AbstractMatrix, fname=:symcover_min!)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("$fname requires a square matrix"))
    require_abs_symmetric(A, fname)
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(ax))"))
    T = float(eltype(a))
    supp = fill!(similar(a, Bool), false)
    foreach_support_sym(A) do i, j, v
        supp[i] = true
        supp[j] = true
    end
    for i in ax
        supp[i] || (a[i] = zero(eltype(a)))
    end
    foreach_support_sym(A) do i, j, v
        (isfinite(a[i]) && a[i] > zero(a[i])) ||
            throw(ArgumentError("symcover_min! requires a start with finite positive scale on every supported row, got a[$(string(i))] = $(string(a[i]))"))
        (isfinite(a[j]) && a[j] > zero(a[j])) ||
            throw(ArgumentError("symcover_min! requires a start with finite positive scale on every supported row, got a[$(string(j))] = $(string(a[j]))"))
        lv, li, lj = log(T(v)), log(T(a[i])), log(T(a[j]))
        lv - li - lj <= _start_slack(lv, li, lj) ||
            throw(ArgumentError("symcover_min! requires a start that covers `A`, but a[$(string(i))]*a[$(string(j))] = $(string(a[i] * a[j])) < $(string(v)) = abs(A[$(string(i)),$(string(j))]); see initialize_symcover"))
    end
    return inflate_feasible!(a, A)
end

# Validate, normalize, and balance an asymmetric hard-cover start.
function _prepare_cover_start!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    T = float(promote_type(eltype(a), eltype(b)))
    suppa = fill!(similar(a, Bool), false)
    suppb = fill!(similar(b, Bool), false)
    foreach_support(A) do i, j, v
        suppa[i] = true
        suppb[j] = true
    end
    for i in eachindex(a)
        suppa[i] || (a[i] = zero(eltype(a)))
    end
    for j in eachindex(b)
        suppb[j] || (b[j] = zero(eltype(b)))
    end
    foreach_support(A) do i, j, v
        (isfinite(a[i]) && a[i] > zero(a[i])) ||
            throw(ArgumentError("cover_min! requires a start with finite positive scale on every supported row, got a[$(string(i))] = $(string(a[i]))"))
        (isfinite(b[j]) && b[j] > zero(b[j])) ||
            throw(ArgumentError("cover_min! requires a start with finite positive scale on every supported column, got b[$(string(j))] = $(string(b[j]))"))
        lv, li, lj = log(T(v)), log(T(a[i])), log(T(b[j]))
        lv - li - lj <= _start_slack(lv, li, lj) ||
            throw(ArgumentError("cover_min! requires a start that covers `A`, but a[$(string(i))]*b[$(string(j))] = $(string(a[i] * b[j])) < $(string(v)) = abs(A[$(string(i)),$(string(j))]); see initialize_cover"))
    end
    _balance_cover!(a, b, A)
    return inflate_feasible!(a, b, A)
end


# Inner solves for `AbsLog{2}` Newton steps:
#
# - `:dense` factorizes the normal equations.
# - `:woodbury` represents them as sparse `C + U*U'`, using conjugate gradients
#   while the condition estimate is small and sparse Cholesky otherwise. It is
#   restricted to nearly dense `Float64` problems.
# - `:lsqr` applies the weighted residual operator `M` matrix-free, with sparse
#   Cholesky right-preconditioning for `Float64`.
#   It is not interchangeable with CG on the normal equations: LSQR's accuracy
#   tracks the condition number of `M` (≈ √κ), CG's that of `MᵀM` (≈ κ), and at
#   κ = 1e8 the latter exhausts double precision.
# - `:auto` selects `:woodbury` when supported, `:lsqr` when the stored support
#   fills at most `AUTO_LSQR_MAX_DENSITY` of the grid, and `:dense` otherwise.

# Maximum support density for the `:auto` LSQR path. At higher densities the
# exact dense solve has lower overhead and can stop on a sign-stable step.
const AUTO_LSQR_MAX_DENSITY = 1 // 4

# Condition estimate above which Woodbury uses sparse Cholesky instead of CG.
const WOODBURY_CG_KAPPA = 1000

# Solve `(C + U*U')x = f` from the factorization `F` of `C` using the
# Woodbury identity. `rhs` and `sol` hold the combined `[f U]` solve.
function _woodbury_solve!(x, F::SparseCholesky, U, f, rhs, sol)
    k = size(U, 2)
    copyto!(view(rhs, :, 1), f)
    copyto!(view(rhs, :, 2:k+1), U)
    solve!(sol, F, CHOLMOD_A, rhs)
    y = view(sol, :, 1)
    Y = view(sol, :, 2:k+1)
    K = U' * Y
    for i in axes(K, 1)
        K[i, i] += oneunit(eltype(K))
    end
    g = K \ (U' * y)
    copyto!(x, y)
    return mul!(x, Y, g, -1, 1)
end

# Jacobi-preconditioned CG for `B*x = f`. Convergence is confirmed with a fresh
# residual; failure lets the caller fall back to factorization.
function _pcg!(Bmul!, x, dg, f, r, z, d, Ad, maxiter::Int, tol)
    iters = 0
    Bmul!(r, x)
    @. r = f - r
    nrm = norm(r)
    fresh = true          # `r` holds `f − B x`, not the recurrence's estimate of it
    @. z = r / dg
    copyto!(d, z)
    rz = dot(r, z)
    while iters < maxiter
        if nrm <= tol
            fresh && return iters, true
            iters += 1
            Bmul!(Ad, x)
            @. r = f - Ad
            nrm = norm(r)
            fresh = true
            @. z = r / dg
            copyto!(d, z)
            rz = dot(r, z)
            continue
        end
        iters += 1
        Bmul!(Ad, d)
        dAd = dot(d, Ad)
        dAd > 0 || break
        a = rz / dAd
        @. x += a * d
        @. r -= a * Ad
        fresh = false
        nrm = norm(r)
        @. z = r / dg
        rznew = dot(r, z)
        @. d = z + (rznew / rz) * d
        rz = rznew
    end
    return iters, fresh && nrm <= tol
end

# Matrix-free LSQR for `min norm(M*x-b)`, warm-started from `x0`. The stopping
# test uses the normal-equations residual estimated from the bidiagonalization.
# Returns `(x, iters)`.
function _lsqr(Amul!, Atmul!, b::AbstractVector{T}, x0::AbstractVector{T};
               atol=5000 * eps(T), maxiter::Int=2 * (length(b) + length(x0)) + 100) where {T}
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

# Support layouts for residuals `x[p] + x[q] - log|A[i,j]|`. Symmetric
# off-diagonal entries are stored once with multiplicity two.

# One element per support entry for LSQR and dense solves.
struct EdgeList{T}
    edges::Vector{Tuple{Int,Int}}   # support entries as pairs of unknowns
    cvals::Vector{T}                # log|A_ij| per stored entry
end

# Dense grid for Woodbury sweeps; `-Inf` marks entries outside the support.
struct Grid{T}
    C::Matrix{T}
end

# Linear system over stacked log scales.
struct SupportSystem{T,S}
    N::Int
    supp::S                         # support layout: `EdgeList` or `Grid`
    symmetric::Bool                 # an off-diagonal entry stands for both orientations
    hassupp::BitVector              # unknowns carrying at least one support entry
    dfull::Vector{T}                # complete-support diagonal (Woodbury path only)
    zedges::Vector{Tuple{Int,Int}}  # zero set, as pairs of unknowns (Woodbury path only)
    U::Matrix{T}                    # low-rank block of `B + v0·v0ᵀ = C + U·Uᵀ` (Woodbury path only)
    v0::Vector{T}                   # gauge: dense adds `v0·v0ᵀ`, LSQR appends the row `v0ᵀx = 0`
    function SupportSystem{T,S}(N, supp, symmetric, hassupp, dfull, zedges, U, v0) where {T,S}
        return new{T,S}(N, supp, symmetric, hassupp, dfull, zedges, U, v0)
    end
end
SupportSystem{T}(N, supp::S, args...) where {T,S} = SupportSystem{T,S}(N, supp, args...)

# Support sweeps, specialized by layout.
#
# Per-entry augmented-Lagrangian term with multiplier λ ≥ 0 and penalty weight κ:
#
#     ψ(z) = z²               for z ≥ b,
#     ψ(z) = κ(z − s)² + s·b  for z < b,
#
# where `s = λ/(2κ)` and `b = λ/(2(κ-1))`. The omitted constant is independent
# of `z`. Precomputed scales keep the branch test identical across sweeps.

# Objective `f(x) = Σ_e mult_e·ψ_e(z_e)`.
function _fal(x, κ, λ, sscale, bscale, supp::EdgeList{T}, symmetric::Bool) where {T}
    edges, cvals = supp.edges, supp.cvals
    κT = T(κ)
    v = zero(T)
    for (e, (p, q)) in enumerate(edges)
        z = x[p] + x[q] - cvals[e]
        l = λ[e]
        viol = z < l * bscale
        s = ifelse(viol, l * sscale, zero(T))
        w = ifelse(viol, κT, oneunit(T))
        v += ((symmetric && p != q) ? 2 : 1) * (w * (z - s)^2 + s * (l * bscale))
    end
    return v
end

function _fal(x, κ, λ, sscale, bscale, supp::Grid{T}, symmetric::Bool) where {T}
    C = supp.C
    m, n = size(C)
    κT = T(κ)
    v = zero(T)
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            lj = view(λ, 1:j-1, j)
            xi = view(x, 1:j-1)
            vj = zero(T)
            @simd for i in _eachindex(cj, xi, lj)
                c = cj[i]
                z = xi[i] + xj - c
                l = lj[i]
                viol = z < l * bscale
                s = ifelse(viol, l * sscale, zero(T))
                w = ifelse(viol, κT, oneunit(T))
                vj += ifelse(isfinite(c), w * (z - s)^2 + s * (l * bscale), zero(T))
            end
            c = C[j, j]
            z = 2xj - c
            l = λ[j, j]
            viol = z < l * bscale
            s = ifelse(viol, l * sscale, zero(T))
            w = ifelse(viol, κT, oneunit(T))
            v += 2vj + ifelse(isfinite(c), w * (z - s)^2 + s * (l * bscale), zero(T))
        end
    else
        xr = view(x, 1:m)
        for j in 1:n
            xj = x[m+j]
            cj = view(C, :, j)
            lj = view(λ, :, j)
            vj = zero(T)
            @simd for i in _eachindex(cj, xr, lj)
                c = cj[i]
                z = xr[i] + xj - c
                l = lj[i]
                viol = z < l * bscale
                s = ifelse(viol, l * sscale, zero(T))
                w = ifelse(viol, κT, oneunit(T))
                vj += ifelse(isfinite(c), w * (z - s)^2 + s * (l * bscale), zero(T))
            end
            v += vj
        end
    end
    return v
end

# Compute the objective and the active set in one sweep.
function _falpat(x, κ, λ, sscale, bscale, pat, supp::EdgeList{T}, symmetric::Bool) where {T}
    edges, cvals = supp.edges, supp.cvals
    κT = T(κ)
    v = zero(T)
    same = true
    for (e, (p, q)) in enumerate(edges)
        z = x[p] + x[q] - cvals[e]
        l = λ[e]
        viol = z < l * bscale
        s = ifelse(viol, l * sscale, zero(T))
        w = ifelse(viol, κT, oneunit(T))
        v += ((symmetric && p != q) ? 2 : 1) * (w * (z - s)^2 + s * (l * bscale))
        same &= viol == pat[e]
    end
    return v, same
end

function _falpat(x, κ, λ, sscale, bscale, pat, supp::Grid{T}, symmetric::Bool) where {T}
    C = supp.C
    m, n = size(C)
    κT = T(κ)
    v = zero(T)
    ndiff = 0
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            pj = view(pat, 1:j-1, j)
            lj = view(λ, 1:j-1, j)
            xi = view(x, 1:j-1)
            vj = zero(T)
            # Keep the Boolean pattern out of the vectorized floating-point loop.
            @simd for i in _eachindex(cj, xi, lj)
                c = cj[i]
                z = xi[i] + xj - c
                l = lj[i]
                viol = z < l * bscale
                s = ifelse(viol, l * sscale, zero(T))
                w = ifelse(viol, κT, oneunit(T))
                vj += ifelse(isfinite(c), w * (z - s)^2 + s * (l * bscale), zero(T))
            end
            # Stop comparing after the first pattern change.
            if ndiff == 0
                dj = 0
                @simd for i in _eachindex(cj, pj, xi, lj)
                    c = cj[i]
                    dj += ifelse((isfinite(c) & (xi[i] + xj - c < lj[i] * bscale)) == pj[i], 0, 1)
                end
                ndiff += dj
            end
            c = C[j, j]
            fin = isfinite(c)
            z = 2xj - c
            l = λ[j, j]
            viol = z < l * bscale
            s = ifelse(viol, l * sscale, zero(T))
            w = ifelse(viol, κT, oneunit(T))
            v += 2vj + ifelse(fin, w * (z - s)^2 + s * (l * bscale), zero(T))
            ndiff += ifelse((fin & viol) == pat[j, j], 0, 1)
        end
    else
        xr = view(x, 1:m)
        for j in 1:n
            xj = x[m+j]
            cj = view(C, :, j)
            pj = view(pat, :, j)
            lj = view(λ, :, j)
            vj = zero(T)
            @simd for i in _eachindex(cj, xr, lj)
                c = cj[i]
                z = xr[i] + xj - c
                l = lj[i]
                viol = z < l * bscale
                s = ifelse(viol, l * sscale, zero(T))
                w = ifelse(viol, κT, oneunit(T))
                vj += ifelse(isfinite(c), w * (z - s)^2 + s * (l * bscale), zero(T))
            end
            if ndiff == 0
                dj = 0
                @simd for i in _eachindex(cj, pj, xr, lj)
                    c = cj[i]
                    dj += ifelse((isfinite(c) & (xr[i] + xj - c < lj[i] * bscale)) == pj[i], 0, 1)
                end
                ndiff += dj
            end
            v += vj
        end
    end
    return v, ndiff == 0
end

# Multiplier storage matching the violation-pattern layout.
_multiplier_storage(::Type{T}, supp::EdgeList) where {T} = zeros(T, length(supp.edges))
_multiplier_storage(::Type{T}, supp::Grid) where {T} = zeros(T, size(supp.C))

# Update multipliers and return `max(-z, min(z, λ))` over the support.
function _update_multipliers!(λ, x, κ, supp::EdgeList{T}, symmetric::Bool) where {T}
    edges, cvals = supp.edges, supp.cvals
    dλ = 2 * (T(κ) - oneunit(T))
    v = zero(T)
    for (e, (p, q)) in enumerate(edges)
        z = x[p] + x[q] - cvals[e]
        l = max(zero(T), λ[e] - dλ * z)
        λ[e] = l
        v = max(v, -z, min(z, l))
    end
    return v
end

function _update_multipliers!(λ, x, κ, supp::Grid{T}, symmetric::Bool) where {T}
    C = supp.C
    m, n = size(C)
    dλ = 2 * (T(κ) - oneunit(T))
    v = zero(T)
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            lj = view(λ, 1:j-1, j)
            xi = view(x, 1:j-1)
            vj = typemin(T)
            @simd for i in _eachindex(cj, xi, lj)
                c = cj[i]
                fin = isfinite(c)
                z = xi[i] + xj - c
                l = ifelse(fin, max(zero(T), lj[i] - dλ * z), zero(T))
                lj[i] = l
                vj = max(vj, ifelse(fin, max(-z, min(z, l)), typemin(T)))
            end
            c = C[j, j]
            fin = isfinite(c)
            z = 2xj - c
            l = ifelse(fin, max(zero(T), λ[j, j] - dλ * z), zero(T))
            λ[j, j] = l
            v = max(v, vj, ifelse(fin, max(-z, min(z, l)), typemin(T)), zero(T))
        end
    else
        xr = view(x, 1:m)
        for j in 1:n
            xj = x[m+j]
            cj = view(C, :, j)
            lj = view(λ, :, j)
            vj = typemin(T)
            @simd for i in _eachindex(cj, xr, lj)
                c = cj[i]
                fin = isfinite(c)
                z = xr[i] + xj - c
                l = ifelse(fin, max(zero(T), lj[i] - dλ * z), zero(T))
                lj[i] = l
                vj = max(vj, ifelse(fin, max(-z, min(z, l)), typemin(T)))
            end
            v = max(v, vj, zero(T))
        end
    end
    return v
end

# Violated-set storage. `Matrix{Bool}` permits vectorized column views.
_violation_pattern(supp::EdgeList) = falses(length(supp.edges))
_violation_pattern(supp::Grid) = fill(false, size(supp.C))

# Smallest uniform log-scale shift that restores feasibility.
function _boost_shift(x, supp::EdgeList{T}, symmetric::Bool) where {T}
    edges, cvals = supp.edges, supp.cvals
    γ = zero(T)
    for (e, (p, q)) in enumerate(edges)
        γ = max(γ, (cvals[e] - x[p] - x[q]) / 2)
    end
    return γ
end

function _boost_shift(x, supp::Grid{T}, symmetric::Bool) where {T}
    C = supp.C
    m, n = size(C)
    γ = zero(T)
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            xi = view(x, 1:j-1)
            gj = zero(T)
            @simd for i in eachindex(cj, xi)
                gj = max(gj, (cj[i] - xi[i] - xj) / 2)
            end
            γ = max(γ, gj, (C[j, j] - 2xj) / 2)
        end
    else
        xr = view(x, 1:m)
        for j in 1:n
            xj = x[m+j]
            cj = view(C, :, j)
            gj = zero(T)
            @simd for i in eachindex(cj, xr)
                gj = max(gj, (cj[i] - xr[i] - xj) / 2)
            end
            γ = max(γ, gj)
        end
    end
    return γ
end

# Assemble the Woodbury right-hand side, active set, degrees, and diagonal.
# `κ === nothing` denotes the unweighted solve. Off-support entries of `C` are
# `-Inf`, so products with them go through `ifelse(isfinite(c), ...)`: `0 * -Inf`
# is NaN.
# Active entries have weight `κ` and target `c + λ/(2κ)`, adding `λ/2` to the
# right-hand side wherever `w*c` lands.
# `vrow` stores active off-diagonal rows in compressed-column order; `vcnt`
# stores their column counts. `dg` and `degV` carry diagonal entries.
function _assemble_woodbury!(f, dg, degV, vrow, vcnt, vpat, x, κ, λ, bscale,
                             supp::Grid{T}, symmetric::Bool, dκ) where {T}
    C = supp.C
    m, n = size(C)
    weighted = κ !== nothing
    κT = weighted ? T(κ) : oneunit(T)
    half = oneunit(T) / 2
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            lj = view(λ, 1:j-1, j)
            xi = view(x, 1:j-1)
            fi = view(f, 1:j-1)
            vj = view(vpat, 1:j-1, j)
            fq = zero(T)
            @simd for i in _eachindex(cj, xi, fi, vj, lj)
                c = cj[i]
                fin = isfinite(c)
                viol = weighted & fin & (xi[i] + xj - c < lj[i] * bscale)
                w = ifelse(viol, κT, oneunit(T))
                wc = ifelse(fin, w * c + ifelse(viol, half * lj[i], zero(T)), zero(T))
                fi[i] += wc
                fq += wc
                vj[i] = viol
            end
            c = C[j, j]
            fin = isfinite(c)
            viol = weighted & fin & (2xj - c < λ[j, j] * bscale)
            w = ifelse(viol, κT, oneunit(T))
            f[j] += fq + ifelse(fin, w * c + ifelse(viol, half * λ[j, j], zero(T)), zero(T))
            vpat[j, j] = viol
            nv = 0
            for i in eachindex(vj)
                vj[i] || continue
                push!(vrow, i)
                nv += 1
                degV[i] += 1
                degV[j] += 1
                dg[i] += dκ
                dg[j] += dκ
            end
            vcnt[j] = nv
            # A symmetric diagonal entry sits at both ends of its own residual, so it
            # lands on `dg[j]` twice while counting once in `degV`.
            if vpat[j, j]
                degV[j] += 1
                dg[j] += 2dκ
            end
        end
    else
        xr = view(x, 1:m)
        fr = view(f, 1:m)
        for j in 1:n
            q = m + j
            xj = x[q]
            cj = view(C, :, j)
            lj = view(λ, :, j)
            vj = view(vpat, :, j)
            fq = zero(T)
            @simd for i in _eachindex(cj, xr, fr, vj, lj)
                c = cj[i]
                fin = isfinite(c)
                viol = weighted & fin & (xr[i] + xj - c < lj[i] * bscale)
                w = ifelse(viol, κT, oneunit(T))
                wc = ifelse(fin, w * c + ifelse(viol, half * lj[i], zero(T)), zero(T))
                fr[i] += wc
                fq += wc
                vj[i] = viol
            end
            f[q] += fq
            nv = 0
            for i in eachindex(vj)
                vj[i] || continue
                push!(vrow, i)
                nv += 1
                degV[i] += 1
                degV[q] += 1
                dg[i] += dκ
                dg[q] += dκ
            end
            vcnt[q] = nv
        end
    end
    return f
end

# Assemble the upper triangle of the sparse Woodbury correction in CSC order,
# reusing its storage across solves.
function _assemble_C!(colptr::Vector{Int}, rowval::Vector{Int}, nzval::Vector{T},
                      zptr, zrow, vptr, vrow, dg, dκ, N::Int) where {T}
    colptr[1] = 1
    for q in 1:N
        colptr[q+1] = colptr[q] + (zptr[q+1] - zptr[q]) + (vptr[q+1] - vptr[q]) + 1
    end
    nnz = colptr[N+1] - 1
    length(rowval) == nnz || resize!(rowval, nnz)
    length(nzval) == nnz || resize!(nzval, nnz)
    for q in 1:N
        s = colptr[q]
        zs, ze = zptr[q], zptr[q+1] - 1
        vs, ve = vptr[q], vptr[q+1] - 1
        while zs <= ze && vs <= ve
            if zrow[zs] < vrow[vs]
                rowval[s] = zrow[zs]; nzval[s] = -oneunit(T); zs += 1
            else
                rowval[s] = vrow[vs]; nzval[s] = dκ; vs += 1
            end
            s += 1
        end
        while zs <= ze
            rowval[s] = zrow[zs]; nzval[s] = -oneunit(T); zs += 1; s += 1
        end
        while vs <= ve
            rowval[s] = vrow[vs]; nzval[s] = dκ; vs += 1; s += 1
        end
        # `dg` carries the same diagonal plus the identity the ridge loop added.
        rowval[s] = q
        nzval[s] = dg[q] - oneunit(T)
    end
    return SparseMatrixCSC(N, N, colptr, rowval, nzval)
end

# `y = Symmetric(Cu) * x` for an upper-triangular compressed-column `Cu`.
function _symmul!(y::AbstractVector{T}, Cu::SparseMatrixCSC{T}, x::AbstractVector{T}) where {T}
    fill!(y, zero(T))
    rv = rowvals(Cu)
    nz = nonzeros(Cu)
    for q in axes(Cu, 2)
        xq = x[q]
        s = zero(T)
        for k in nzrange(Cu, q)
            p = rv[k]
            v = nz[k]
            if p == q
                s += v * xq
            else
                y[p] += v * xq
                s += v * x[p]
            end
        end
        # Later columns add the remaining terms to `y[q]`.
        y[q] += s
    end
    return y
end

# Augmented-Lagrangian defaults.
const AL_PENALTY = 1e2
const AL_MAXOUTER = 32

# Warn when the outer iteration ends with a KKT residual well above the
# inner solver's accuracy floor.
function _warn_unconverged(fname::Symbol, stats, maxouter::Int)
    stats.converged && return nothing
    v = isempty(stats.kkt) ? oftype(stats.vwarn, NaN) : stats.kkt[end]
    v > stats.vwarn || return nothing
    @warn "$fname: the multiplier iteration ended after $(stats.nouter) of maxouter=$maxouter updates with KKT residual $v (tolerance $(stats.vtol)); the result covers `A` but may not minimize the objective. Increase `maxouter` or `κ`."
    return nothing
end

# Position of `S[i,j]` in `nonzeros(S)`; the entry must be stored.
function _nzindex(S::SparseMatrixCSC, i::Int, j::Int)
    r = nzrange(S, j)
    rv = rowvals(S)
    k = searchsortedfirst(view(rv, r), i)
    k <= length(r) && rv[r[k]] == i ||
        throw(ArgumentError("the preconditioner pattern is missing entry ($i, $j)"))
    return r[k]
end

# Unweighted normal-matrix pattern for the LSQR preconditioner. The ridge makes
# bipartite support components positive definite. `N == 0` disables it.
function _precond_pattern(::Type{T}, supp::EdgeList, v0, N::Int, mult) where {T}
    N == 0 && return spzeros(T, 0, 0)
    # Diagonal values first; each off-diagonal edge is stored in both triangles.
    Mv = zeros(T, N)
    colptr = zeros(Int, N + 1)
    colptr[1] = 1
    for (p, q) in supp.edges
        if p == q
            Mv[p] += 4 * oneunit(T)
        else
            w = mult(p, q) * oneunit(T)
            Mv[p] += w
            Mv[q] += w
            colptr[p+1] += 1
            colptr[q+1] += 1
        end
    end
    dmax = zero(T)
    for p in 1:N
        Mv[p] += v0[p]^2
        dmax = max(dmax, Mv[p])
        colptr[p+1] += 1
    end
    ρ = _precond_ridge(dmax)
    cumsum!(colptr, colptr)
    nz = colptr[N+1] - 1
    rowval = zeros(Int, nz)
    nzval = zeros(T, nz)
    cursor = colptr[1:N]
    for (p, q) in supp.edges
        p == q && continue
        w = mult(p, q) * oneunit(T)
        rowval[cursor[q]] = p; nzval[cursor[q]] = w; cursor[q] += 1
        rowval[cursor[p]] = q; nzval[cursor[p]] = w; cursor[p] += 1
    end
    for p in 1:N
        rowval[cursor[p]] = p
        nzval[cursor[p]] = Mv[p] + ρ
    end
    # Rows within a column arrive in edge order; the factorization needs them sorted.
    for q in 1:N
        r = colptr[q]:colptr[q+1]-1
        perm = sortperm(view(rowval, r))
        rowval[r] = rowval[r][perm]
        nzval[r] = nzval[r][perm]
    end
    return SparseMatrixCSC(N, N, colptr, rowval, nzval)
end

_precond_pattern(::Type{T}, ::Grid, v0, N::Int, mult) where {T} = spzeros(T, 0, 0)

# Scale-relative ridge for a positive-definite preconditioner.
_precond_ridge(dmax::T) where {T} = (dmax > 0 ? dmax : oneunit(T)) * sqrt(eps(T))

# Default storage limit, in bytes, for the LSQR Cholesky preconditioner.
# Tripping this switches to diagonal preconditioning, reducing memory
# consumption but increasing the number of iterations for convergence.
const LSQR_FILL_BUDGET = 1 << 30

# Return the symbolic factorization of `M` and its predicted number of values.
function _precond_analysis(M::SparseMatrixCSC{Float64,Int})
    F = analyze!(SparseCholesky(), M)
    return F, factor_entries(F)
end

# Storage for the dense normal-equation factorization of the `:dense` path.
_dense_factor_type(::Type{Float64}) = LinearAlgebra.BunchKaufman{Float64,Matrix{Float64},Vector{Int}}
_dense_factor_type(::Type{T}) where {T} = LinearAlgebra.LU{T,Matrix{T},Vector{Int}}

# `AbsLog{2}` augmented-Lagrangian iteration. `boost=true` applies a final
# feasibility shift; the support layout selects the inner solver.
function _abslog2_auglag(sys::SupportSystem{T}, x0;
                               κ::Real, maxouter::Int, maxiter::Int, linsolve::Symbol, boost::Bool,
                               fillbudget::Real=LSQR_FILL_BUDGET) where {T}
    κ > 1 || throw(ArgumentError("κ must exceed 1; got $κ"))
    maxouter >= 0 || throw(ArgumentError("maxouter must be nonnegative; got $maxouter"))
    N = sys.N
    supp = sys.supp
    v0 = sys.v0
    U = sys.U
    use_woodbury = supp isa Grid
    ne = supp isa EdgeList ? length(supp.edges) : 0
    use_lsqr = linsolve === :lsqr
    # CHOLMOD preconditioning is limited to Float64.
    use_precond = use_lsqr && T === Float64
    # Residual multiplicity of each stored entry.
    symmetric = sys.symmetric
    mult = (p, q) -> (symmetric && p != q) ? 2 : 1
    # Multiplier estimates for the support constraints.
    λ = _multiplier_storage(T, supp)
    # Base diagonal of `C`, corrected for the zero set.
    czero = copy(sys.dfull)
    for (p, q) in sys.zedges
        czero[p] -= oneunit(T)
        czero[q] -= oneunit(T)
    end
    # Each Newton step solves `min norm(sqrt(W)*(R*x-c))`. Dense and Woodbury
    # paths solve regularized normal equations; LSQR applies `sqrt(W)*R`
    # matrix-free with a gauge row.
    f = zeros(T, N)
    ws = zeros(T, ne)       # √weight per stored entry, frozen during one solve
    cv = zeros(T, ne + 1)   # √weight · log|A_ij|, with a trailing 0 gauge target
    # Violated entries under the current frozen weights.
    vpat = _violation_pattern(supp)
    # The dense and factor-preconditioned solves refactor only when the weights
    # change, and the weights depend only on (κ, active set): cache that key.
    prevκ = Ref(zero(T))
    prevpat = _violation_pattern(supp)
    Bfact = Ref{Union{Nothing,_dense_factor_type(T)}}(nothing)
    vrow = Int[]                             # violated rows, grouped by column (Woodbury path only)
    vcnt = zeros(Int, use_woodbury ? N : 0)  # violated off-diagonal entries per column
    vptr = zeros(Int, use_woodbury ? N + 1 : 0)
    degV = zeros(Int, use_woodbury ? N : 0)  # violated entries per unknown
    dg = zeros(T, use_woodbury ? N : 0)      # diagonal of `B`, for the ridge and the CG preconditioner
    # Zero set grouped by column, excluding the diagonal, which `dg` carries.
    zptr = zeros(Int, use_woodbury ? N + 1 : 0)
    zrow = Int[]
    if use_woodbury
        for (p, q) in sys.zedges
            p == q || (zptr[q+1] += 1)
        end
        zptr[1] = 1
        cumsum!(zptr, zptr)
        resize!(zrow, zptr[end] - 1)
        zcursor = zptr[1:end-1]
        # `sys.zedges` ordering keeps each compressed column sorted.
        for (p, q) in sys.zedges
            p == q && continue
            zrow[zcursor[q]] = p
            zcursor[q] += 1
        end
    end
    # Storage for the sparse correction, reused across solves.
    Ccolptr = zeros(Int, use_woodbury ? N + 1 : 0)
    Crowval = Int[]
    Cnzval = T[]
    px = zeros(T, use_precond ? N : 0)   # scale vector recovered from the LSQR variable
    pxv = zeros(T, use_precond ? N : 0)  # `(L'P) \ y`, the unscaled LSQR variable
    pg = zeros(T, use_precond ? N : 0)   # `Rᵀ√W y` before the preconditioner is applied
    mdiag = zeros(T, use_precond ? N : 0)   # weighted degrees, the preconditioner's diagonal
    # The normal-matrix pattern is constant, so one symbolic analysis serves all
    # stages. Use its diagonal if the predicted Cholesky factor exceeds the budget.
    Msp = _precond_pattern(T, supp, v0, use_precond ? N : 0, mult)
    MF, fill_entries = use_precond ? _precond_analysis(Msp) : (nothing, 0)
    use_factor = use_precond && sizeof(T) * fill_entries <= fillbudget
    # Positions of the entries each factored solve overwrites: the diagonal, and
    # both copies of each off-diagonal support entry.
    dpos = use_factor ? [_nzindex(Msp, p, p) for p in 1:N] : Int[]
    epos = zeros(Int, use_factor ? 2 * ne : 0)
    if use_factor
        for (e, (p, q)) in enumerate(supp.edges)
            p == q && continue
            epos[2 * e - 1] = _nzindex(Msp, p, q)
            epos[2 * e] = _nzindex(Msp, q, p)
        end
    end
    psqrt = zeros(T, use_factor ? 0 : (use_precond ? N : 0))  # `K` of the diagonal preconditioner
    rhs = zeros(T, use_woodbury ? N : 0, size(U, 2) + 1)
    wsol = similar(rhs)
    WF = use_woodbury ? SparseCholesky() : nothing
    dmin = use_woodbury ? minimum(sys.dfull) : oneunit(T)
    cgx = zeros(T, use_woodbury ? N : 0)
    cgr = zeros(T, use_woodbury ? N : 0)
    cgz = zeros(T, use_woodbury ? N : 0)
    cgd = zeros(T, use_woodbury ? N : 0)
    cgAd = zeros(T, use_woodbury ? N : 0)
    nsolves = Ref(0)
    nlsqr = Ref(0)
    ncg = Ref(0)
    nchol = Ref(0)
    solve_weighted = function (x, κ)
        nsolves[] += 1
        if supp isa Grid
            # `B + v0*v0' = C + U*U'`, with `C` corrected for zero and
            # violated entries.
            dκ = κ === nothing ? zero(T) : T(κ) - oneunit(T)
            bscalel = dκ > 0 ? inv(2 * dκ) : zero(T)
            fill!(f, zero(T))
            copyto!(dg, czero)
            fill!(degV, 0)
            empty!(vrow)
            _assemble_woodbury!(f, dg, degV, vrow, vcnt, vpat, x, κ, λ, bscalel, supp, symmetric, dκ)
            vptr[1] = 1
            for q in 1:N
                vptr[q+1] = vptr[q] + vcnt[q]
            end
            # Match the ridge used by the dense path.
            dmax = zero(T)
            maxdegV = 0
            for p in 1:N
                dmax = max(dmax, dg[p] + oneunit(T))
                maxdegV = max(maxdegV, degV[p])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for p in 1:N
                dg[p] += oneunit(T) + ridge
            end
            C = _assemble_C!(Ccolptr, Crowval, Cnzval, zptr, zrow, vptr, vrow, dg, dκ, N)
            # Use CG while the Gershgorin condition estimate remains small.
            κest = oneunit(T) + dκ * 2 * maxdegV / dmin
            if κest <= WOODBURY_CG_KAPPA
                copyto!(cgx, x)
                Bmul! = function (yy, xx)
                    _symmul!(yy, C, xx)
                    # Indicator columns make the low-rank term a block sum.
                    for k in axes(U, 2)
                        s = zero(T)
                        for p in 1:N
                            u = U[p, k]
                            iszero(u) || (s += u * xx[p])
                        end
                        for p in 1:N
                            u = U[p, k]
                            iszero(u) || (yy[p] += u * s)
                        end
                    end
                    return yy
                end
                it, ok = _pcg!(Bmul!, cgx, dg, f, cgr, cgz, cgd, cgAd,
                               50 + 20 * ceil(Int, sqrt(κest)), 100 * eps(T) * norm(f))
                ncg[] += it
                ok && return copy(cgx)
            end
            nchol[] += 1
            analyze!(WF, C)
            factorize!(WF, C)
            return _woodbury_solve!(zeros(T, N), WF, U, f, rhs, wsol)
        end
        # The closures below capture `edges`; a captured variable assigned in
        # more than one branch would be boxed and lose its type.
        edges = supp.edges
        cvals = supp.cvals
        if use_lsqr
            weighted = κ !== nothing
            κl = weighted ? T(κ) : oneunit(T)
            sscalel = weighted ? inv(2 * κl) : zero(T)
            bscalel = weighted ? inv(2 * (κl - oneunit(T))) : zero(T)
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                l = λ[e]
                viol = weighted && (x[p] + x[q] - c) < l * bscalel
                vpat[e] = viol
                sw = sqrt(mult(p, q) * (viol ? κl : oneunit(T)))
                ws[e] = sw
                # Active entries fit toward the shifted target `c + λ/(2κ)`.
                cv[e] = sw * (viol ? c + l * sscalel : c)
            end
            g = ne + 1   # index of the appended gauge row
            if use_factor
                if prevκ[] != κl || vpat != prevpat
                    # Refill the preconditioner with the weights this solve
                    # freezes and refactor it in place: the pattern, and so the
                    # symbolic analysis, is the same on every solve. Weighted
                    # degrees `mdiag` are the diagonal of `RᵀWR`.
                    fill!(mdiag, zero(T))
                    nzv = nonzeros(Msp)
                    for (e, (p, q)) in enumerate(edges)
                        w = ws[e]^2
                        if p == q
                            mdiag[p] += 4 * w
                        else
                            mdiag[p] += w
                            mdiag[q] += w
                            nzv[epos[2 * e - 1]] = w
                            nzv[epos[2 * e]] = w
                        end
                    end
                    dmax = zero(T)
                    for p in 1:N
                        mdiag[p] += v0[p]^2
                        dmax = max(dmax, mdiag[p])
                    end
                    ρ = _precond_ridge(dmax)
                    for p in 1:N
                        nzv[dpos[p]] = mdiag[p] + ρ
                    end
                    factorize!(MF, Msp)
                    prevκ[] = κl
                    copyto!(prevpat, vpat)
                end
                # `K = P'L` is the preconditioner; LSQR iterates on `y = K' x`.
                Pmul! = function (y, yv)
                    solve_up!(pxv, MF, yv)
                    for (e, (p, q)) in enumerate(edges)
                        y[e] = ws[e] * (pxv[p] + pxv[q])
                    end
                    y[g] = dot(v0, pxv)
                    return y
                end
                Ptmul! = function (z, y)
                    fill!(pg, zero(T))
                    for (e, (p, q)) in enumerate(edges)
                        t = ws[e] * y[e]
                        pg[p] += t
                        pg[q] += t
                    end
                    @. pg += v0 * y[g]
                    solve_ptl!(z, MF, pg)
                    return z
                end
                mul!(px, Msp, x)
                solve_ptl!(px, MF, px)
                soly, it = _lsqr(Pmul!, Ptmul!, cv, px)
                nlsqr[] += it
                return solve_up!(soly, MF, soly)
            elseif use_precond
                # Weighted degrees, the diagonal of `RᵀWR`.
                fill!(mdiag, zero(T))
                for (e, (p, q)) in enumerate(edges)
                    w = ws[e]^2
                    if p == q
                        mdiag[p] += 4 * w
                    else
                        mdiag[p] += w
                        mdiag[q] += w
                    end
                end
                # An unknown outside the support gets an identity row.
                for p in 1:N
                    d = mdiag[p] + v0[p]^2
                    psqrt[p] = sqrt(d > 0 ? d : oneunit(T))
                end
                # Diagonal `K` needs only elementwise scaling.
                Dmul! = function (y, yv)
                    @. px = yv / psqrt
                    for (e, (p, q)) in enumerate(edges)
                        y[e] = ws[e] * (px[p] + px[q])
                    end
                    y[g] = dot(v0, px)
                    return y
                end
                Dtmul! = function (z, y)
                    fill!(pg, zero(T))
                    for (e, (p, q)) in enumerate(edges)
                        t = ws[e] * y[e]
                        pg[p] += t
                        pg[q] += t
                    end
                    @. pg += v0 * y[g]
                    @. z = pg / psqrt
                    return z
                end
                soly, it = _lsqr(Dmul!, Dtmul!, cv, psqrt .* x)
                nlsqr[] += it
                return soly ./ psqrt
            end
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
            fill!(f, zero(T))
            weighted = κ !== nothing
            κl = weighted ? T(κ) : oneunit(T)
            bscalel = weighted ? inv(2 * (κl - oneunit(T))) : zero(T)
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                l = λ[e]
                viol = weighted && (x[p] + x[q] - c) < l * bscalel
                vpat[e] = viol
                w = viol ? κl : oneunit(T)
                # Active entries fit toward `c + λ/(2κ)`, adding `w*s = λ/2`.
                t = viol ? w * c + l / 2 : w * c
                f[p] += t
                q == p || (f[q] += t)
            end
            if Bfact[] === nothing || prevκ[] != κl || vpat != prevpat
                B = v0 * v0'
                for (e, (p, q)) in enumerate(edges)
                    w = vpat[e] ? κl : oneunit(T)
                    B[p, p] += w
                    B[p, q] += w
                    if q != p
                        B[q, q] += w
                        B[q, p] += w
                    end
                end
                # A small scale-relative ridge lifts unpinned gauge directions.
                # Support-free variables receive an identity row.
                dmax = zero(T)
                for p in 1:N
                    dmax = max(dmax, B[p, p])
                end
                ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
                for p in 1:N
                    B[p, p] = sys.hassupp[p] ? B[p, p] + ridge : oneunit(T)
                end
                # Narrower types promote to Float64 before reaching this path,
                # so anything else is wider than LAPACK handles.
                Bfact[] = T === Float64 ? bunchkaufman!(Symmetric(B)) : lu!(B)
                prevκ[] = κl
                copyto!(prevpat, vpat)
            end
            return (Bfact[] \ f)::Vector{T}
        end
    end
    x = x0 === nothing ? solve_weighted(zeros(T, N), nothing) : x0
    # Absolute thresholds preserve covariance under rescaling because the
    # residuals are logarithmic.
    vtol = 1000 * eps(T)
    vwarn = max(sqrt(eps(T)), T(1e-6))
    κcap = T(use_lsqr ? 1e5 : 1e8)
    κcur = T(κ)
    exits = Symbol[]
    drops = T[]
    viols = T[]
    κtrace = T[]
    converged = maxouter == 0
    vprev = T(Inf)
    nstall = 0
    for _ in 1:maxouter
        sscale = inv(2 * κcur)
        bscale = inv(2 * (κcur - oneunit(T)))
        fcur = _fal(x, κcur, λ, sscale, bscale, supp, symmetric)
        exit = :maxiter
        drop = zero(T)
        for _ in 1:maxiter
            xnew = solve_weighted(x, κcur)
            t = one(T)
            xt = xnew
            fnew, stable = _falpat(xt, κcur, λ, sscale, bscale, vpat, supp, symmetric)
            while fnew > fcur && t > 500_000 * eps(T)
                t /= 2
                xt = x .+ t .* (xnew .- x)
                fnew = _fal(xt, κcur, λ, sscale, bscale, supp, symmetric)
                stable = false
            end
            x = xt
            drop = (fcur - fnew) / max(fcur, one(T))
            # For exact inner solves, an unchanged active set ends the descent.
            if !use_lsqr && stable
                exit = :stable
                break
            end
            if fcur - fnew <= 5000 * eps(T) * max(fcur, one(T))
                exit = :decrease
                break
            end
            fcur = fnew
        end
        push!(exits, exit)
        push!(drops, drop)
        push!(κtrace, κcur)
        v = _update_multipliers!(λ, x, κcur, supp, symmetric)
        push!(viols, v)
        if v <= vtol
            converged = true
            break
        end
        if 10 * v > 9 * vprev
            # Stop after three passes at the inner solver's accuracy floor.
            nstall += 1
            nstall >= 3 && break
        else
            nstall = 0
        end
        if v > vprev / 10 && κcur < κcap
            κcur = min(10 * κcur, κcap)
        end
        vprev = v
    end
    # Hard covers receive a final uniform feasibility shift.
    if boost
        γ = _boost_shift(x, supp, symmetric)
        for p in 1:N
            x[p] += γ
        end
    end
    return x, (; nsolves=nsolves[], lsqriters=nlsqr[], cgiters=ncg[],
               cholsolves=nchol[], nouter=length(exits), exits=Tuple(exits),
               drops=Tuple(drops), kkt=Tuple(viols), κs=Tuple(κtrace),
               converged, vtol, vwarn,
               linsolve=(use_lsqr ? :lsqr : use_woodbury ? :woodbury : :dense),
               precond=(!use_precond ? :none : use_factor ? :factor : :diagonal))
end

# Worker for `symcover_min(::AbsLog{2})`, returning `(a, stats)`. A supplied
# `start` replaces the cold initial solve. Narrow types compute in `Float64`.
function _symcover_min_abslog2(A::AbstractMatrix; κ::Real=AL_PENALTY, maxouter::Int=AL_MAXOUTER,
                               maxiter::Int=40, linsolve::Symbol=:auto, start=nothing,
                               boost::Bool=true, fillbudget::Real=LSQR_FILL_BUDGET,
                               fname=:symcover_min)
    linsolve in (:auto, :dense, :lsqr, :woodbury) ||
        throw(ArgumentError("linsolve must be :auto, :dense, :lsqr, or :woodbury; got :$linsolve"))
    # Shared symmetry check for native symmetric minimal covers.
    require_abs_symmetric(A, fname)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    # The problem depends only on `abs.(A)`, so the working type is real.
    T = float(real(eltype(A)))
    # Solver tolerances require at least Float64 resolution.
    if eps(T) > eps(Float64)
        a64, stats = _symcover_min_abslog2(convert(AbstractMatrix{promote_type(eltype(A), Float64)}, A);
                                           κ, maxouter, maxiter, linsolve, start, boost, fillbudget, fname)
        # Narrowing rounds to nearest and so can round a product below its entry.
        a = T.(a64)
        boost && _certify_cover!(a, A, fname)
        return a, stats
    end
    n = length(ax)
    use_lsqr = linsolve === :lsqr
    # One counting traversal decides the solver; the layout it needs is built after.
    o = first(ax) - 1
    nza = zeros(Int, n)    # support entries per row, counted in both orientations
    foreach_support_sym(A) do i, j, v
        nza[i-o] += 1
        i == j || (nza[j-o] += 1)
    end
    hassupp = falses(n)
    nsupp = 0              # support entries of `A`, counted in both orientations
    maxzero = 0            # largest number of zeros in any row of `A`
    for ip in 1:n
        hassupp[ip] = nza[ip] > 0
        nsupp += nza[ip]
        maxzero = max(maxzero, n - nza[ip])
    end
    # `n*I - L_Z` is positive definite only while no row carries more than
    # `n ÷ 4` zeros; the total budget bounds cost.
    nzero = n * n - nsupp
    zbudget = 4 * n
    use_woodbury = false
    if !use_lsqr && linsolve !== :dense
        ok = T === Float64 && maxzero <= n ÷ 4 && nzero <= zbudget
        if linsolve === :woodbury && !ok
            T === Float64 ||
                throw(ArgumentError("linsolve=:woodbury requires Float64 arithmetic, but `A` works in $T; use :dense or :lsqr"))
            maxzero <= n ÷ 4 ||
                throw(ArgumentError("linsolve=:woodbury requires every row of `A` to have at most n ÷ 4 = $(n ÷ 4) zeros; got $maxzero"))
            throw(ArgumentError("linsolve=:woodbury requires `A` to have at most 4n = $zbudget zeros in total; got $nzero"))
        end
        use_woodbury = ok
    end
    if linsolve === :auto && !use_woodbury && nsupp <= AUTO_LSQR_MAX_DENSITY * (n * n)
        use_lsqr = true
        linsolve = :lsqr
    end
    # Start the LSQR iteration from the heuristic cover. `maxouter == 0` returns
    # the unweighted fit instead.
    if start === nothing && use_lsqr && maxouter > 0
        start = symcover(A)
    end
    # Woodbury uses a grid; dense and LSQR use an edge list.
    supp = if use_woodbury
        C = fill(T(-Inf), n, n)
        foreach_support_sym(A) do i, j, v
            C[i-o, j-o] = log(T(v))
        end
        Grid{T}(C)
    else
        G = _sym_support(A, T)
        edges = Tuple{Int,Int}[]
        cvals = T[]
        for (ip, i) in enumerate(ax)
            for s in _slots(G, i)
                jp = G.idx[s] - first(ax) + 1
                if jp >= ip
                    push!(edges, (ip, jp))
                    push!(cvals, log(G.val[s]))
                end
            end
        end
        EdgeList{T}(edges, cvals)
    end
    # Zero set defining the off-diagonal pattern of sparse `C`, grouped by row.
    zedges = Tuple{Int,Int}[]
    if use_woodbury
        Cgrid = supp.C
        for ip in 1:n
            for jp in ip:n
                isfinite(Cgrid[ip, jp]) || push!(zedges, (ip, jp))
            end
        end
    end
    # The ridge handles singular symmetric support graphs.
    sys = SupportSystem{T}(n, supp, true, hassupp,
                           use_woodbury ? fill(T(n), n) : T[], zedges,
                           ones(T, use_woodbury ? n : 0, use_woodbury ? 1 : 0),
                           zeros(T, n))
    x0 = start === nothing ? nothing :
         T[hassupp[ip] ? log(T(start[i])) : zero(T) for (ip, i) in enumerate(ax)]
    α, stats = _abslog2_auglag(sys, x0; κ, maxouter, maxiter, linsolve, boost, fillbudget)
    _warn_unconverged(fname, stats, maxouter)
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    for (ip, i) in enumerate(ax)
        a[i] = hassupp[ip] ? exp(α[ip]) : zero(T)
    end
    boost && _certify_cover!(a, A, fname)
    return a, stats
end

# Worker for `cover_min(::AbsLog{2})`, returning `(a, b, stats)`.
function _cover_min_abslog2(A::AbstractMatrix; κ::Real=AL_PENALTY, maxouter::Int=AL_MAXOUTER,
                            maxiter::Int=40, linsolve::Symbol=:auto, start=nothing,
                            boost::Bool=true, fillbudget::Real=LSQR_FILL_BUDGET)
    linsolve in (:auto, :dense, :lsqr, :woodbury) ||
        throw(ArgumentError("linsolve must be :auto, :dense, :lsqr, or :woodbury; got :$linsolve"))
    axr = axes(A, 1)
    axc = axes(A, 2)
    # The problem depends only on `abs.(A)`, so the working type is real.
    T = float(real(eltype(A)))
    # Solver tolerances require at least Float64 resolution.
    if eps(T) > eps(Float64)
        a64, b64, stats = _cover_min_abslog2(convert(AbstractMatrix{promote_type(eltype(A), Float64)}, A);
                                             κ, maxouter, maxiter, linsolve, start, boost, fillbudget)
        # Narrowing rounds to nearest and so can round a product below its entry.
        a, b = T.(a64), T.(b64)
        boost && _certify_cover!(a, b, A, :cover_min)
        return a, b, stats
    end
    m = length(axr)
    n = length(axc)
    N = m + n
    use_lsqr = linsolve === :lsqr
    # Stack row positions before column positions; scatter results back to `A`'s axes.
    or = first(axr) - 1
    oc = first(axc) - 1
    nzrow = zeros(Int, m)   # support entries per row, for the balance convention
    nzcol = zeros(Int, n)   # ditto per column
    foreach_support(A) do i, j, v
        nzrow[i-or] += 1
        nzcol[j-oc] += 1
    end
    ne = sum(nzrow)
    hasrow = nzrow .> 0
    hascol = nzcol .> 0
    # `min(m,n)*I - L_Z` is positive definite only while no row or column
    # carries more than `min(m,n) ÷ 4` zeros; the total budget bounds cost.
    maxzero = 0
    for ip in 1:m
        maxzero = max(maxzero, n - nzrow[ip])
    end
    for jp in 1:n
        maxzero = max(maxzero, m - nzcol[jp])
    end
    zbound = min(m, n) ÷ 4
    # Enforce both per-axis and total zero-count limits.
    nzero = m * n - ne
    zbudget = 4 * max(m, n)
    use_woodbury = false
    if !use_lsqr && linsolve !== :dense
        ok = T === Float64 && maxzero <= zbound && nzero <= zbudget
        if linsolve === :woodbury && !ok
            T === Float64 ||
                throw(ArgumentError("linsolve=:woodbury requires Float64 arithmetic, but `A` works in $T; use :dense or :lsqr"))
            maxzero <= zbound ||
                throw(ArgumentError("linsolve=:woodbury requires every row and column of `A` to have at most min(m, n) ÷ 4 = $zbound zeros; got $maxzero"))
            throw(ArgumentError("linsolve=:woodbury requires `A` to have at most 4·max(m, n) = $zbudget zeros in total; got $nzero"))
        end
        use_woodbury = ok
    end
    if linsolve === :auto && !use_woodbury && ne <= AUTO_LSQR_MAX_DENSITY * (m * n)
        use_lsqr = true
        linsolve = :lsqr
    end
    # Start the LSQR iteration from the heuristic cover.
    if start === nothing && use_lsqr && maxouter > 0
        start = cover(A)
    end
    # Woodbury uses a grid; dense and LSQR use an edge list.
    supp = if use_woodbury
        C = fill(T(-Inf), m, n)
        foreach_support(A) do i, j, v
            C[i-or, j-oc] = log(T(v))
        end
        Grid{T}(C)
    else
        G = _row_support(A, T)
        edges = Tuple{Int,Int}[]
        cvals = T[]
        for (ip, i) in enumerate(axr)
            for s in _slots(G, i)
                push!(edges, (ip, m + G.idx[s] - first(axc) + 1))
                push!(cvals, log(G.val[s]))
            end
        end
        EdgeList{T}(edges, cvals)
    end
    # Zero set defining the off-diagonal pattern of sparse `C`.
    zedges = Tuple{Int,Int}[]
    dfull = zeros(T, use_woodbury ? N : 0)
    Umat = zeros(T, use_woodbury ? N : 0, use_woodbury ? 2 : 0)
    if use_woodbury
        for ip in 1:m
            dfull[ip] = T(n)
            Umat[ip, 1] = oneunit(T)
        end
        for jp in 1:n
            dfull[m+jp] = T(m)
            Umat[m+jp, 2] = oneunit(T)
        end
        Cgrid = supp.C
        for ip in 1:m
            for jp in 1:n
                isfinite(Cgrid[ip, jp]) || push!(zedges, (ip, m + jp))
            end
        end
    end
    # Pin the global row/column gauge on supported variables.
    v0 = zeros(T, N)
    for ip in 1:m
        hasrow[ip] && (v0[ip] = one(T))
    end
    for jp in 1:n
        hascol[jp] && (v0[m+jp] = -one(T))
    end
    sys = SupportSystem{T}(N, supp, false, vcat(hasrow, hascol), dfull, zedges, Umat, v0)
    x0 = if start === nothing
        nothing
    else
        sa, sb = start
        s0 = zeros(T, N)
        for (ip, i) in enumerate(axr)
            hasrow[ip] && (s0[ip] = log(T(sa[i])))
        end
        for (jp, j) in enumerate(axc)
            hascol[jp] && (s0[m+jp] = log(T(sb[j])))
        end
        s0
    end
    x, stats = _abslog2_auglag(sys, x0; κ, maxouter, maxiter, linsolve, boost, fillbudget)
    _warn_unconverged(:cover_min, stats, maxouter)
    # Apply the balance convention independently to each support component.
    rowcomp, colcomp, ncomp, _, _ = _support_components(A)
    Lα = zeros(T, ncomp)
    Lβ = zeros(T, ncomp)
    nec = zeros(Int, ncomp)
    for ip in 1:m
        c = rowcomp[ip]
        c == 0 && continue
        Lα[c] += nzrow[ip] * x[ip]
        nec[c] += nzrow[ip]
    end
    for jp in 1:n
        c = colcomp[jp]
        c == 0 && continue
        Lβ[c] += nzcol[jp] * x[m+jp]
    end
    s = Lβ
    for c in 1:ncomp
        s[c] = (Lβ[c] - Lα[c]) / (2 * nec[c])
    end
    # Dense scale vectors matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (ip, i) in enumerate(axr)
        a[i] = hasrow[ip] ? exp(x[ip] + s[rowcomp[ip]]) : zero(T)
    end
    for (jp, j) in enumerate(axc)
        b[j] = hascol[jp] ? exp(x[m+jp] - s[colcomp[jp]]) : zero(T)
    end
    boost && _certify_cover!(a, b, A, :cover_min)
    return a, b, stats
end

# Soft `AbsLog{2}` covers are the unweighted initial solve with no feasibility
# shift. Convexity makes the constrained iteration and multistart unnecessary.
_soft_symcover_min_abslog2(A::AbstractMatrix; kwargs...) =
    _symcover_min_abslog2(A; maxouter=0, boost=false, fname=:soft_symcover_min, kwargs...)
_soft_cover_min_abslog2(A::AbstractMatrix; kwargs...) =
    _cover_min_abslog2(A; maxouter=0, boost=false, kwargs...)

# JuMP reference used to test the native symmetric solver.
function symcover_min_jump end

# JuMP reference used to test the native asymmetric solver.
function cover_min_jump end

# Reject all non-solved statuses, including `ALMOST_*`. Taking the status keeps
# this helper independent of JuMP.
function check_solved(status, solver, fname)
    Symbol(status) in (:OPTIMAL, :LOCALLY_SOLVED) ||
        error("$fname: $solver terminated with status $status")
    return nothing
end
