# Newton refinement for the `PowerMean` soft covers.
#
# Over stacked log scales `x` (asymmetric: rows in `1:m`, columns in `m+1:m+n`;
# symmetric: one scale per row), the objective has one term per stored entry `e`
# joining unknowns `u[e] <= v[e]`:
#
#     E(x) = ∑_e mult[e] * ψ(x[u] + x[v] - L[e]),   ψ(z) = (exp(-p z) - 1 + p z)/p,
#
# with `L = log|A|`. An asymmetric entry has multiplicity one. A symmetric
# off-diagonal pair is stored once with multiplicity two and a diagonal entry once
# with multiplicity one, which is the full-grid sum of `cover_objective`. With
# weights `W[e] = exp(-p z[e])` (the coverage ratios to the power `p`),
#
#     ∇E = ∑_e mult[e] * (1 - W[e]) * (e_u + e_v),
#     ∇²E = p * ∑_e mult[e] * W[e] * (e_u + e_v) * (e_u + e_v)',
#
# i.e. `(n - W e, m - Wᵀ e)` and `p [diag(W e) W; Wᵀ diag(Wᵀ e)]` in the
# asymmetric case and `2(n - W e)` and `2p [diag(W e) + W]` in the symmetric one.
#
# The Hessian is singular exactly on the components of the support graph that are
# bipartite with no diagonal entry (every asymmetric component), with null vector
# `±1` on the two sides; the gradient is orthogonal to it. Each such component is
# grounded: one unknown's step is fixed at zero and its equation dropped, which
# leaves the rest of the system positive definite and consistent. The gauge of the
# result is fixed afterwards by the drivers' balance conventions.

struct PowerMeanSystem{T}
    u::Vector{Int}
    v::Vector{Int}
    L::Vector{T}
    mult::Vector{T}
    count::Vector{Int}      # stored entries in each unknown's row or column
    grid::Int               # number of cells of the full grid
end

# System of the asymmetric problem, from row-grouped `log|A|` and the column axis.
function _powermean_system(R::GroupedSupport{T}, colax::AbstractUnitRange) where {T}
    m, n = length(R.ax), length(colax)
    oc = first(colax) - 1
    ne = length(R.val)
    u = Vector{Int}(undef, ne)
    v = Vector{Int}(undef, ne)
    L = Vector{T}(undef, ne)
    count = zeros(Int, m + n)
    e = 0
    for (ip, i) in enumerate(R.ax)      # `ip` is the position of row `i`
        for s in _slots(R, i)
            e += 1
            jp = m + R.idx[s] - oc
            u[e], v[e], L[e] = ip, jp, R.val[s]
            count[ip] += 1
            count[jp] += 1
        end
    end
    return PowerMeanSystem{T}(u, v, L, ones(T, ne), count, m * n)
end

# System of the symmetric problem, from the symmetric row-grouped `log|A|`.
function _powermean_symsystem(S::GroupedSupport{T}) where {T}
    off = first(S.ax) - 1
    n = length(S.ax)
    u, v, L, mult = Int[], Int[], T[], T[]
    count = zeros(Int, n)
    for g in S.ax
        gp = g - off
        count[gp] = _ngroup(S, g)
        for s in _slots(S, g)
            hp = S.idx[s] - off
            hp >= gp || continue
            push!(u, gp); push!(v, hp); push!(L, S.val[s]); push!(mult, hp == gp ? 1 : 2)
        end
    end
    return PowerMeanSystem{T}(u, v, L, mult, count, n * n)
end

# Unknowns whose step is fixed at zero: those without support, and one unknown (of
# largest count) in each component that is bipartite with no diagonal entry.
function _powermean_pinned(sys::PowerMeanSystem)
    N = length(sys.count)
    ptr = zeros(Int, N + 1)
    for e in eachindex(sys.u, sys.v)
        ptr[sys.u[e]+1] += 1
        ptr[sys.v[e]+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    nbr = Vector{Int}(undef, ptr[end] - 1)
    cursor = ptr[1:N]
    for e in eachindex(sys.u, sys.v)
        a, b = sys.u[e], sys.v[e]
        nbr[cursor[a]] = b; cursor[a] += 1
        nbr[cursor[b]] = a; cursor[b] += 1
    end
    pinned = sys.count .== 0
    color = zeros(Int8, N)
    stack = Int[]
    for r in 1:N
        (color[r] != 0 || sys.count[r] == 0) && continue
        color[r] = 1
        push!(stack, r)
        bipartite = true
        best = r
        while !isempty(stack)
            q = pop!(stack)
            sys.count[q] > sys.count[best] && (best = q)
            for k in ptr[q]:ptr[q+1]-1
                h = nbr[k]
                if color[h] == 0
                    color[h] = -color[q]
                    push!(stack, h)
                elseif color[h] == color[q]
                    bipartite = false       # an odd cycle or a diagonal entry
                end
            end
        end
        bipartite && (pinned[best] = true)
    end
    return pinned
end

# Evaluate at `xt = x + t*d`: the weights `W`, gradient `g` and row sums `s` of the
# weights there, and the change of the objective from `x`, whose weights are `W0`.
# Returns the change and the sum of the magnitudes of its terms, the scale of its
# roundoff. Each term is `mult * (W0 * expm1(-p δ) + p δ)/p` with `δ = t*(d[u] + d[v])`,
# exact for the difference and free of the cancellation of subtracting two objectives.
function _powermean_eval!(W, g, s, xt, d, t::T, W0, sys::PowerMeanSystem{T}, p::T) where {T}
    fill!(g, zero(T))
    fill!(s, zero(T))
    ΔE = scale = zero(T)
    for e in eachindex(sys.u, sys.v, sys.L, sys.mult, W, W0)
        a, b, w = sys.u[e], sys.v[e], sys.mult[e]
        δ = t * (d[a] + d[b])
        q = W0[e] * expm1(-p * δ)
        ΔE += w * (q + p * δ)
        scale += w * (abs(q) + abs(p * δ))
        We = exp(-p * (xt[a] + xt[b] - sys.L[e]))
        W[e] = We
        g[a] += w * (1 - We)
        g[b] += w * (1 - We)
        s[a] += We
        a == b || (s[b] += We)
    end
    return ΔE / p, scale / p
end

# Largest `|s/count - 1|`: the imbalance of the sweeps.
function _powermean_newton_imbalance(s, count)
    imb = zero(eltype(s))
    for q in eachindex(s, count)
        count[q] > 0 && (imb = max(imb, abs(s[q] / count[q] - 1)))
    end
    return imb
end

# Inner linear solvers for the Newton step:
#
# - `:dense` factorizes the dense Hessian (Bunch-Kaufman for `Float64`, LU otherwise);
# - `:cholesky` factorizes the sparse Hessian with CHOLMOD (`Float64` only), with
#   one symbolic analysis for all steps;
# - `:cg` runs Jacobi-preconditioned conjugate gradients with one pass over the
#   support per product, to the relative residual `min(1/2, sqrt(imbalance))`.
#
# `:auto` uses the dense factorization on support that fills at least
# `AUTO_LSQR_MAX_DENSITY` of the grid, otherwise the sparse one when its predicted
# storage and flop count fit the budgets of the LSQR preconditioner
# (`LSQR_FILL_BUDGET`, `LSQR_FLOP_BUDGET`); a factorization whose predicted flop
# count exceeds `LSQR_FLOP_BUDGET` times the number of stored entries gives way to CG.
struct PowerMeanLinsolve{T,F}
    kind::Symbol
    H::Matrix{T}                    # `:dense`
    M::SparseMatrixCSC{Float64,Int} # `:cholesky`: upper triangle of the Hessian
    F::F
    epos::Vector{Int}               # position in `nonzeros(M)` of each off-diagonal entry
    dpos::Vector{Int}               # position of each diagonal entry
    dg::Vector{T}                   # diagonal of the Hessian
    f::Vector{T}                    # right-hand side
    work::NTuple{4,Vector{T}}       # CG vectors
end

function _powermean_linsolve(sys::PowerMeanSystem{T}, linsolve::Symbol) where {T}
    N = length(sys.count)
    nstored = sum(sys.mult; init=zero(T))
    dense_ok = N^3 / 3 <= LSQR_FLOP_BUDGET * nstored
    kind = linsolve
    M = spzeros(Float64, 0, 0)
    F = nothing
    if kind === :cholesky || (kind === :auto && T === Float64 && nstored < AUTO_LSQR_MAX_DENSITY * sys.grid)
        T === Float64 ||
            throw(ArgumentError("linsolve=:cholesky requires Float64 arithmetic, but the problem works in $T; use :dense or :cg"))
        M = _powermean_pattern(sys)
        F, entries, flops = _precond_analysis(M)
        if kind === :auto
            kind = sizeof(T) * entries <= LSQR_FILL_BUDGET && flops <= LSQR_FLOP_BUDGET * nstored ? :cholesky :
                   dense_ok ? :dense : :cg
        end
    elseif kind === :auto
        kind = dense_ok ? :dense : :cg
    end
    H = kind === :dense ? zeros(T, N, N) : zeros(T, 0, 0)
    epos = zeros(Int, kind === :cholesky ? length(sys.u) : 0)
    dpos = kind === :cholesky ? [_nzindex(M, q, q) for q in 1:N] : Int[]
    if kind === :cholesky
        for e in eachindex(sys.u, sys.v)
            a, b = sys.u[e], sys.v[e]
            a == b || (epos[e] = _nzindex(M, a, b))
        end
    end
    nw = kind === :cg ? N : 0
    work = (zeros(T, nw), zeros(T, nw), zeros(T, nw), zeros(T, nw))
    return PowerMeanLinsolve{T,typeof(F)}(kind, H, M, F, epos, dpos, zeros(T, N), zeros(T, N), work)
end

# Upper-triangular pattern of the Hessian, rows sorted within each column.
function _powermean_pattern(sys::PowerMeanSystem)
    N = length(sys.count)
    I = collect(1:N)
    J = collect(1:N)
    for e in eachindex(sys.u, sys.v)
        sys.u[e] == sys.v[e] && continue
        push!(I, sys.u[e])
        push!(J, sys.v[e])
    end
    return sparse(I, J, ones(Float64, length(I)), N, N)
end

# Solve `H d = -g` with the pinned unknowns' steps fixed at zero. Returns the
# number of passes over the support and of factorizations.
function _powermean_direction!(d, ls::PowerMeanLinsolve{T}, sys::PowerMeanSystem{T}, W, g,
                               pinned, imb::T, p::T) where {T}
    N = length(d)
    f, dg = ls.f, ls.dg
    for q in eachindex(f, g, pinned)
        f[q] = pinned[q] ? zero(T) : -g[q]
    end
    fill!(dg, zero(T))
    for e in eachindex(sys.u, sys.v, W)
        a, b = sys.u[e], sys.v[e]
        h = p * sys.mult[e] * W[e]
        dg[a] += h
        dg[b] += h
        a == b && (dg[a] += 2h)
    end
    # A scale-relative ridge keeps the factorizations definite should weights underflow.
    dmax = maximum(dg; init=zero(T))
    ridge = (dmax > 0 ? dmax : one(T)) * eps(T)
    for q in eachindex(dg, pinned)
        dg[q] = pinned[q] ? one(T) : dg[q]
    end
    if ls.kind === :dense
        H = ls.H
        fill!(H, zero(T))
        for e in eachindex(sys.u, sys.v, W)
            a, b = sys.u[e], sys.v[e]
            (a == b || pinned[a] || pinned[b]) && continue
            h = p * sys.mult[e] * W[e]
            H[a, b] += h
            H[b, a] += h
        end
        for q in 1:N
            H[q, q] = pinned[q] ? one(T) : dg[q] + ridge
        end
        Fd = T === Float64 ? bunchkaufman!(Symmetric(H)) : lu!(H)
        ldiv!(d, Fd, f)
        return 1, 1
    elseif ls.kind === :cholesky
        M = ls.M
        nzv = nonzeros(M)
        for e in eachindex(sys.u, sys.v, W, ls.epos)
            a, b = sys.u[e], sys.v[e]
            a == b && continue
            nzv[ls.epos[e]] = (pinned[a] || pinned[b]) ? zero(T) : p * sys.mult[e] * W[e]
        end
        for q in 1:N
            nzv[ls.dpos[q]] = pinned[q] ? one(T) : dg[q] + ridge
        end
        factorize!(ls.F, M)
        solve!(d, ls.F, CHOLMOD_A, f)
        return 1, 1
    end
    # Conjugate gradients from `d = 0`; pinned unknowns keep a zero step since
    # their residual starts and stays zero.
    nmul = Ref(0)
    Hmul! = function (y, xx)
        nmul[] += 1
        fill!(y, zero(T))
        for e in eachindex(sys.u, sys.v, W)
            a, b = sys.u[e], sys.v[e]
            h = p * sys.mult[e] * W[e]
            z = h * (xx[a] + xx[b])
            y[a] += z
            y[b] += z
        end
        for q in eachindex(y, pinned)
            pinned[q] && (y[q] = xx[q])
        end
        return y
    end
    fill!(d, zero(T))
    r, z, dd, Ad = ls.work
    η = min(one(T) / 2, sqrt(imb))
    _pcg!(Hmul!, d, dg, f, r, z, dd, Ad, 2N + 100, η * norm(f))
    return nmul[], 0
end

# Damped Newton iteration from `x` on the objective of `sys`, with Armijo
# backtracking. The objective is convex and coercive up to the gauge, so the
# iteration converges from any start. Near the solution the change of the
# objective falls below its roundoff, and a step is then accepted if it reduces
# the imbalance. Stops when the imbalance is at most `tol`, after `maxiter`
# steps, or when no step along the Newton direction is acceptable. Returns
# `(converged, nsteps, npasses, nfactor, imbalance, linsolve)`, counting in
# `npasses` every traversal of the support: evaluations, assemblies, and products.
function _powermean_newton!(x0::Vector{T}, sys::PowerMeanSystem{T}, p::T, maxiter::Integer, tol::T,
                            linsolve::Symbol) where {T}
    x = x0
    N = length(x)
    ne = length(sys.u)
    pinned = _powermean_pinned(sys)
    ls = _powermean_linsolve(sys, linsolve)
    W, Wt = zeros(T, ne), zeros(T, ne)
    g, gt, s, st = zeros(T, N), zeros(T, N), zeros(T, N), zeros(T, N)
    d, xt = zeros(T, N), similar(x)
    _powermean_eval!(W, g, s, x, d, zero(T), W, sys, p)
    npasses, nfactor = 1, 0
    imb = _powermean_newton_imbalance(s, sys.count)
    k = 0
    while imb > tol && k < maxiter
        k += 1
        np, nf = _powermean_direction!(d, ls, sys, W, g, pinned, imb, p)
        npasses += np
        nfactor += nf
        gd = dot(g, d)
        gd < 0 || break
        # Far from the solution, a nearly flat direction can give a step that
        # changes some weights by `exp(±p δ)` beyond the floating-point range;
        # start the search where every `p|δ|` is at most `POWERMEAN_MAXSTEP`.
        t = min(one(T), T(POWERMEAN_MAXSTEP) / (2p * maximum(abs, d)))
        accepted = false
        for _ in 1:POWERMEAN_NEWTON_NBACKTRACK
            @. xt = x + t * d
            ΔE, scale = _powermean_eval!(Wt, gt, st, xt, d, t, W, sys, p)
            npasses += 1
            imbt = _powermean_newton_imbalance(st, sys.count)
            if ΔE <= POWERMEAN_ARMIJO * t * gd
                accepted = true
            elseif isfinite(scale) && abs(ΔE) <= POWERMEAN_NOISE * eps(T) * scale
                # The objective cannot resolve this step: judge it by the imbalance.
                accepted = imbt < imb
                accepted || break
            end
            if accepted
                imb = imbt
                break
            end
            t /= 2
        end
        accepted || break
        x, xt = xt, x
        W, Wt = Wt, W
        g, gt = gt, g
        s, st = st, s
    end
    x === x0 || copyto!(x0, x)
    return imb <= tol, k, npasses, nfactor, imb, ls.kind
end

const POWERMEAN_NEWTON_NBACKTRACK = 40
const POWERMEAN_ARMIJO = 1e-4
const POWERMEAN_NOISE = 64
const POWERMEAN_MAXSTEP = 64
