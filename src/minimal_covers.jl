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
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt. These objectives are
  non-convex, so the solver returns the minimum of the basin it starts in. Rather than
  commit to one start, these methods refine each of `strategies` — the
  [`initialize_symcover`](@ref) menu, by default `$(SYMCOVER_MIN_STRATEGIES)` — and return
  the best cover found, at a cost of one solve per start. A strategy that `A` admits no
  start for is skipped. The result is the best *local* minimum on that menu: the multistart
  is a hedge against a poor basin, not a certificate of global optimality.

The `AbsLog` penalties are convex in the log-scales, so for them the minimum value is
unique and no such hedge is needed. `AbsLog{2}` has a unique minimizer too. `AbsLog{1}`
does not: its optimum is a whole face of the feasible polytope, whose members are
genuinely different covers that happen to score alike. The one returned is the member
of that face minimizing the `AbsLog{2}` objective.

!!! note
    Even the native solver is more expensive than the [`symcover`](@ref) heuristic.

See also: [`cover_min`](@ref), [`symcover`](@ref), [`symcover_min!`](@ref).
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
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt. These objectives are
  non-convex, so the solver returns the minimum of the basin it starts in. Rather than
  commit to one start, these methods refine each of `strategies` — the
  [`initialize_cover`](@ref) menu, by default `$(COVER_MIN_STRATEGIES)` — and return the
  best cover found, at a cost of one solve per start. The result is the best *local*
  minimum on that menu: the multistart is a hedge against a poor basin, not a certificate
  of global optimality.

The `AbsLog` penalties are convex in the log-scales, so for them the minimum value is
unique and no such hedge is needed. `AbsLog{2}` has a unique minimizer too. `AbsLog{1}`
does not: its optimum is a whole face of the feasible polytope, whose members are
genuinely different covers that happen to score alike. The one returned is the member
of that face minimizing the `AbsLog{2}` objective.

!!! note
    Even the native solver is more expensive than the [`cover`](@ref) heuristic.

See also: [`symcover_min`](@ref), [`cover`](@ref), [`cover_min!`](@ref).
"""
function cover_min end
cover_min(A::AbstractMatrix; kwargs...) = cover_min(AbsLog{2}(), A; kwargs...)

"""
    a = symcover_min!(ϕ, a, A; kwargs...)
    a = symcover_min!(a, A; kwargs...)

Refine the starting cover `a` into the ϕ-minimal symmetric hard cover of `A`, in
place. This is the second half of the initialize/refine pair: `a` must already be
a starting point, as produced by [`initialize_symcover`](@ref) (or by
[`symcover`](@ref)). The no-ϕ form defaults to `AbsLog{2}()`, matching
[`symcover_min`](@ref), whose keyword arguments and supported ϕ values these
methods share.

`a` must be strictly positive on every row of `A` that carries support, and must
cover `A` — `a[i]*a[j] >= abs(A[i,j])` — to within the roundoff of the log-domain
arithmetic; otherwise an `ArgumentError` is raised. Scales on rows carrying no
support are inert: whatever they hold on input, they are zero on output.

How much the start matters depends on ϕ. Under the `AbsLog` penalties the result is
start-independent: they are convex in the log-scales, `AbsLog{2}` has a unique
minimizer, and `AbsLog{1}` — whose optimum is a whole face of equally-scoring covers
— is pinned to the member of that face minimizing the `AbsLog{2}` objective. The
`AbsLinear` penalties are non-convex, and the identified local minima depend on
the start(s).

See also: [`initialize_symcover`](@ref), [`symcover_min`](@ref), [`cover_min!`](@ref).
"""
function symcover_min! end
symcover_min!(a::AbstractVector, A::AbstractMatrix; kwargs...) =
    symcover_min!(AbsLog{2}(), a, A; kwargs...)

"""
    a, b = cover_min!(ϕ, a, b, A; kwargs...)
    a, b = cover_min!(a, b, A; kwargs...)

Refine the starting cover `(a, b)` into the ϕ-minimal asymmetric hard cover of
`A`, in place. This is the asymmetric counterpart of [`symcover_min!`](@ref), and
carries the same contract on the start: strict positivity on every supported row
and column, coverage of `A` to within roundoff, and inert scales on the
unsupported rows and columns. The no-ϕ form defaults to `AbsLog{2}()`, matching
[`cover_min`](@ref), whose keyword arguments and supported ϕ values these methods
share.

The product `a[i]*b[j]` is unchanged by `a -> c*a`, `b -> b/c`, so the start is
read only up to that gauge: `(a, b)` and `(2a, b/2)` give the same result, and
the result itself is pinned to the balance convention of [`cover_min`](@ref).

See also: [`initialize_cover`](@ref), [`cover_min`](@ref), [`symcover_min!`](@ref).
"""
function cover_min! end
cover_min!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    cover_min!(AbsLog{2}(), a, b, A; kwargs...)

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

# The AbsLog{2} objective is convex in the log-scales, so the continuation converges
# to the same cover from any start; the start is honored (it replaces the cold
# unweighted solve as the first iterate) but is not observable in the result.
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

# The AbsLinear objectives are non-convex, so a refinement reports the minimum of whichever
# basin its start lies in. The drivers below therefore refine every start on a menu and keep
# the best, which is what makes the result of the non-mutating entry point a property of `A`
# rather than of an initialization the caller never chose. The kernels they call — the
# `*_min!` refiners for the AbsLinear penalties — live in the SIAIpopt extension, but the
# menu and the selection are native, so the two families need only one description.
#
# The winner is picked by `_multistart_select`, the same scale-covariant rule the soft-cover
# multistarts use: a later start replaces the incumbent only on a genuine relative
# improvement, never on the roundoff by which two starts reaching the same basin differ.
function symcover_min(ϕ::AbsLinear, A::AbstractMatrix; strategies=SYMCOVER_MIN_STRATEGIES)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    T = float(real(eltype(A)))
    starts = [similar(Array{T}, ax) for _ in strategies]
    # A strategy for which `A` admits no start forfeits its slot; only a menu that yields
    # no start at all leaves nothing to refine.
    built = [_initialize_symcover!(a, A, strategy, :inflate) for (a, strategy) in zip(starts, strategies)]
    covers = [symcover_min!(ϕ, a, A) for (a, ok) in zip(starts, built) if ok]
    isempty(covers) &&
        throw(ArgumentError("symcover_min: no strategy in $strategies yields a starting cover of `A`"))
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

# Log-domain slack allowed of a start supplied to the `*_min!` refiners, in units of
# `eps(T)` scaled by the magnitudes entering the residual. The heuristics reach the
# coverage boundary through log-domain updates and so land on it only to within their
# own roundoff — a fresh `symcover` violates `a[i]*a[j] >= abs(A[i,j])` by a fraction
# of one such unit — and an exact test would reject them. This bound accepts that
# while still rejecting a start that misses coverage by any margin a solver would see.
const START_FEASIBILITY_ULPS = 64

_start_slack(lv::T, li::T, lj::T) where {T} =
    START_FEASIBILITY_ULPS * eps(T) * max(oneunit(T), abs(lv), abs(li), abs(lj))

# Shared prologue of the `symcover_min!` kernels: check that the caller's start is a
# cover of `A`, discard the inert scales on unsupported rows, and move the start onto
# the coverage boundary exactly, so every kernel begins from a feasible point.
function _prepare_symcover_start!(a::AbstractVector, A::AbstractMatrix)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min! requires a square matrix"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$ax"))
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
            throw(ArgumentError("symcover_min! requires a start with finite positive scale on every supported row, got a[$i] = $(a[i])"))
        (isfinite(a[j]) && a[j] > zero(a[j])) ||
            throw(ArgumentError("symcover_min! requires a start with finite positive scale on every supported row, got a[$j] = $(a[j])"))
        lv, li, lj = log(T(v)), log(T(a[i])), log(T(a[j]))
        lv - li - lj <= _start_slack(lv, li, lj) ||
            throw(ArgumentError("symcover_min! requires a start that covers `A`, but a[$i]*a[$j] = $(a[i] * a[j]) < $v = abs(A[$i,$j]); see initialize_symcover"))
    end
    return inflate_feasible!(a, A)
end

# Shared prologue of the `cover_min!` kernels; the asymmetric counterpart of
# `_prepare_symcover_start!`. The start is additionally pinned to the balance
# convention, so the refiners read it only up to the row/column gauge.
function _prepare_cover_start!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$(axes(A, 1))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(eachindex(b)), axes(A, 2)=$(axes(A, 2))"))
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
            throw(ArgumentError("cover_min! requires a start with finite positive scale on every supported row, got a[$i] = $(a[i])"))
        (isfinite(b[j]) && b[j] > zero(b[j])) ||
            throw(ArgumentError("cover_min! requires a start with finite positive scale on every supported column, got b[$j] = $(b[j])"))
        lv, li, lj = log(T(v)), log(T(a[i])), log(T(b[j]))
        lv - li - lj <= _start_slack(lv, li, lj) ||
            throw(ArgumentError("cover_min! requires a start that covers `A`, but a[$i]*b[$j] = $(a[i] * b[j]) < $v = abs(A[$i,$j]); see initialize_cover"))
    end
    _balance_cover!(a, b, A)
    return inflate_feasible!(a, b, A)
end


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
# (matrix-free, for sparse supports). `start`, when given, is a positive cover of `A`
# indexed like `axes(A, 1)` and supplies the first iterate in place of the cold
# unweighted solve; the objective is convex, so it changes the path but not the result.
function _symcover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                               maxiter::Int=40, linsolve::Symbol=:auto, start=nothing,
                               boost::Bool=true)
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
    α = start === nothing ? solve_weighted(zeros(T, n), nothing) :
        T[hassupp[ip] ? log(T(start[i])) : zero(T) for (ip, i) in enumerate(ax)]
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
    # `boost=false` leaves the iterate untouched, for the soft objective, which
    # imposes no coverage constraint and whose optimum the boost would move off.
    γ = zero(T)
    if boost
        for jp in 1:n, ip in 1:n
            S[ip, jp] || continue
            γ = max(γ, (C[ip, jp] - α[ip] - α[jp]) / 2)
        end
    end
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    for (ip, i) in enumerate(ax)
        a[i] = hassupp[ip] ? exp(α[ip] + γ) : zero(T)
    end
    return a, (; nsolves=nsolves[], lsqriters=nlsqr[], linsolve=(use_lsqr ? :lsqr : :dense))
end

# Worker for `cover_min(::AbsLog{2})`. Returns `(a, b, stats)` with `stats` a
# NamedTuple `(; nsolves, lsqriters, linsolve)` (see `_symcover_min_abslog2`).
# `start`, when given, is a positive cover `(a, b)` indexed like the rows and columns
# of `A`, supplying the first iterate in place of the cold unweighted solve.
function _cover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                            maxiter::Int=40, linsolve::Symbol=:auto, start=nothing,
                            boost::Bool=true)
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
    x = if start === nothing
        solve_weighted(zeros(T, N), nothing)
    else
        sa, sb = start
        x0 = zeros(T, N)
        for (ip, i) in enumerate(axr)
            hasrow[ip] && (x0[ip] = log(T(sa[i])))
        end
        for (jp, j) in enumerate(axc)
            hascol[jp] && (x0[m+jp] = log(T(sb[j])))
        end
        x0
    end
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
    # `boost=false` leaves the iterate untouched, for the soft objective, which
    # imposes no coverage constraint and whose optimum the boost would move off.
    # The balance shift below still applies: the gauge is a convention, not a
    # constraint, and every cover this package returns satisfies it.
    if boost
        γ = zero(T)
        for jp in 1:n, ip in 1:m
            S[ip, jp] || continue
            γ = max(γ, (C[ip, jp] - x[ip] - x[m+jp]) / 2)
        end
        for p in 1:N
            x[p] += γ
        end
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
    # Dense scale vectors matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (ip, i) in enumerate(axr)
        a[i] = hasrow[ip] ? exp(x[ip] + s) : zero(T)
    end
    for (jp, j) in enumerate(axc)
        b[j] = hascol[jp] ? exp(x[m+jp] - s) : zero(T)
    end
    return a, b, (; nsolves=nsolves[], lsqriters=nlsqr[], linsolve=(use_lsqr ? :lsqr : :dense))
end

# Workers for the soft (unconstrained) AbsLog{2} covers. The soft objective
# `∑_{i,j∈S} (log a_i + log a_j - log|A_ij|)²` is the hard workers' reweighted
# least-squares problem with every weight held at 1, which is the cold solve they
# already take as their first iterate: `κs=()` runs no penalty continuation, and
# `boost=false` keeps the unconstrained minimizer where it is. It is convex, so one
# linear solve settles it — no iteration and no multistart, unlike the non-convex
# `AbsLinear` soft covers.
#
# Both paths inherit the hard workers' handling of a singular signless Laplacian (the
# `[0 1; 1 0]` support graph among them) and of support-free rows and columns.
_soft_symcover_min_abslog2(A::AbstractMatrix; kwargs...) =
    _symcover_min_abslog2(A; κs=(), boost=false, kwargs...)
_soft_cover_min_abslog2(A::AbstractMatrix; kwargs...) =
    _cover_min_abslog2(A; κs=(), boost=false, kwargs...)

# Internal exact reference implemented by the SIAJuMP extension; used only to
# cross-check the native `symcover_min(::AbsLog{2})` in the test suite.
function symcover_min_jump end

# Internal exact reference implemented by the SIAJuMP extension; used only to
# cross-check the native `cover_min(::AbsLog{2})` in the test suite.
function cover_min_jump end
