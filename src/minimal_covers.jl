# Objective-minimal hard covers. The default AbsLog{2} penalty is solved natively
# here; the other penalties are provided by the SIAJuMP and SIAIpopt extensions,
# whose entry points are declared as stubs below.

# ============================================================
# Public interface
# ============================================================

"""
    a = symcover_min(ϕ, A; kwargs...)
    a = symcover_min(A; kwargs...)

Return the ϕ-minimal symmetric hard cover of `A`: the vector `a` minimizing
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*a[j]))` subject to `a[i]*a[j] >= |A[i,j]|` for every nonzero
entry of `A`. The no-ϕ form defaults to `AbsLog{2}()`, matching [`symcover`](@ref).

Supported ϕ values:
- `AbsLog{2}()`: solved natively (no external solver). Accepts keyword arguments
  `κs` (the penalty-continuation schedule, default `(1e2, 1e4, 1e6, 1e8)`),
  `maxiter` (Newton steps per stage, default `40`), and `linsolve` (the inner
  linear solve: `:auto`/`:dense` use a dense factorization of the reweighted
  normal equations; `:lsqr` uses matrix-free LSQR (per-iteration cost O(nnz),
  intended for large sparse supports)). `linsolve` defaults to `:auto` for
  dense `A`; the `SparseMatrixCSC`/`Symmetric`/`Hermitian` sparse methods
  (from the SparseArrays extension) default to `:lsqr` instead, since a dense
  factorization of the reweighted normal equations is the wrong solve when
  `nnz ≪ n²`.
- `AbsLog{1}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt.

!!! note
    Even the native solver is more expensive than the [`symcover`](@ref) heuristic.

See also: [`cover_min`](@ref), [`symcover`](@ref).
"""
function symcover_min end
symcover_min(A::AbstractMatrix; kwargs...) = symcover_min(AbsLog{2}(), A; kwargs...)

"""
    a, b = cover_min(ϕ, A)
    a, b = cover_min(A)

Return the ϕ-minimal asymmetric hard cover of `A`: the vectors `a`, `b` minimizing
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*b[j]))` subject to `a[i]*b[j] >= |A[i,j]|` for every nonzero
entry of `A`. The row/column scales are pinned to the balance convention
`∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` (`nzaᵢ`, `nzbⱼ` = nonzero counts of row `i`,
column `j`) so the result is deterministic. The no-ϕ form defaults to `AbsLog{2}()`,
matching [`cover`](@ref).

Supported ϕ values:
- `AbsLog{2}()`: solved natively (no external solver). Accepts keyword arguments
  `κs` (the penalty-continuation schedule, default `(1e2, 1e4, 1e6, 1e8)`),
  `maxiter` (Newton steps per stage, default `40`), and `linsolve` (the inner
  linear solve: `:auto`/`:dense` use a dense factorization of the reweighted
  normal equations; `:lsqr` uses matrix-free LSQR (per-iteration cost O(nnz),
  intended for large sparse supports)). `linsolve` defaults to `:auto` for
  dense `A`; the `SparseMatrixCSC` sparse method (from the SparseArrays
  extension) defaults to `:lsqr` instead, since a dense factorization of the
  reweighted normal equations is the wrong solve when `nnz ≪ n²`.
- `AbsLog{1}()`: requires JuMP and HiGHS.

!!! note
    Even the native solver is more expensive than the [`cover`](@ref) heuristic.

See also: [`symcover_min`](@ref), [`cover`](@ref).
"""
function cover_min end
cover_min(A::AbstractMatrix; kwargs...) = cover_min(AbsLog{2}(), A; kwargs...)

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

# ============================================================
# Internal helpers
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

# Worker for `symcover_min(::AbsLog{2})`. Returns `(a, stats)` where `stats` is a
# NamedTuple `(; nsolves, lsqriters, linsolve)` recording the number of inner linear
# solves, the total LSQR iterations (0 on the dense path), and which path ran — used
# by the benchmarks. `linsolve` is `:auto`/`:dense` (dense factorization) or `:lsqr`
# (matrix-free, for sparse supports).
function _symcover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                               maxiter::Int=40, linsolve::Symbol=:auto)
    linsolve in (:auto, :dense, :lsqr) ||
        throw(ArgumentError("linsolve must be :auto, :dense, or :lsqr; got :$linsolve"))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    # The problem only ever depends on abs.(A), a real quantity, so the working type
    # stays real even for complex A (e.g. a complex Hermitian) — Complex has no total
    # order, and the reweighted Newton solve below compares residuals with `<`/`min`.
    T = float(real(eltype(A)))
    n = length(ax)
    use_lsqr = linsolve === :lsqr
    # log|A| on the support S; the Newton solve runs on 1-based positions 1:n and
    # is scattered back onto `a` through `ax` so `A`'s own axes are honored.
    C = zeros(T, n, n)
    S = falses(n, n)
    for (jp, j) in enumerate(ax), (ip, i) in enumerate(ax)
        Aij = abs(A[i, j])
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
        for _ in 1:maxiter
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

# Worker for `cover_min(::AbsLog{2})`. Returns `(a, b, stats)` with `stats` a
# NamedTuple `(; nsolves, lsqriters, linsolve)` (see `_symcover_min_abslog2`).
function _cover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                            maxiter::Int=40, linsolve::Symbol=:auto)
    linsolve in (:auto, :dense, :lsqr) ||
        throw(ArgumentError("linsolve must be :auto, :dense, or :lsqr; got :$linsolve"))
    axr = axes(A, 1)
    axc = axes(A, 2)
    # The problem only ever depends on abs.(A), a real quantity, so the working type
    # stays real even for complex A (e.g. a complex Hermitian) — Complex has no total
    # order, and the reweighted Newton solve below compares residuals with `<`/`min`.
    T = float(real(eltype(A)))
    m = length(axr)
    n = length(axc)
    N = m + n
    use_lsqr = linsolve === :lsqr
    # log|A| on the support S; internal positions 1:m index rows, m+1:m+n index
    # columns, and results are scattered back through axr/axc so A's axes are honored.
    C = zeros(T, m, n)
    S = falses(m, n)
    for (jp, j) in enumerate(axc), (ip, i) in enumerate(axr)
        Aij = abs(A[i, j])
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
        for _ in 1:maxiter
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

# Internal exact reference implemented by the SIAJuMP extension; used only to
# cross-check the native `symcover_min(::AbsLog{2})` in the test suite.
function symcover_min_jump end

# Internal exact reference implemented by the SIAJuMP extension; used only to
# cross-check the native `cover_min(::AbsLog{2})` in the test suite.
function cover_min_jump end
