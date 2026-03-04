module ScaleInvariantAnalysis

using LinearAlgebra
using SparseArrays
using PrimalDualLinearAlgebra
using LinearOperators
using Krylov
using PDMats

export condscale, divmag, dotabs, cover, symcover

include("linalg.jl")
include("utils.jl")

"""
    dotabs(x, y)

Compute the sum of absolute values of elementwise products of `x` and `y`:

    ∑_i |x[i] * y[i]|
"""
function dotabs(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    s = zero(eltype(x)) * zero(eltype(y))
    for i in eachindex(x, y)
        s += abs(x[i] * y[i])
    end
    return s
end


"""
    a = symcover(A; exact=false, regularize=false)

Given a matrix `A` assumed to be symmetric, return a vector `a` serving as a
 symmetric matrix cover, so that `|A[i,j]| <= a[i] * a[j]` for all `i, j`. `a[i]`
is nonnegative, and is zero only if `A[i, j] = 0` for all `j`.

With `exact=true`, `a` minimizes the objective function

    ∑_{i,j : A[i,j] ≠ 0} (log(|A[i,j]| / (a[i] * a[j])))²

and is therefore covariant under changes of scale but not general linear
transformations.

With `exact=false`, the pattern of nonzeros in `A` is approximated as `u * u'`,
where `sum(u) * u[i] = nz[i]` is the number of nonzero in row `i`. This results
in an `O(n^2)` rather than `O(n^3)` algorithm. `regularize=true` adds a small
offset to the diagonal (relevant only when `exact=true`), which handles all-zero
rows of `A`.
"""
function symcover(A::AbstractMatrix; exact::Bool=false, regularize::Bool=false)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    sumlogA, nz = _symcover(A, ax)
    n = length(ax)
    if !exact || all(==(n), nz)
        # Sherman-Morrison formula for efficiency
        offset = sum(sumlogA) / (2 * sum(nz))
        divsafe!(sumlogA, nz)
        return exp.(sumlogA ./ nz .- offset)
    end
    τ = regularize ? sqrt(eps(eltype(sumlogA))) : zero(eltype(sumlogA))
    W = isnz(A)
    divsafe!(sumlogA, vec(sum(W; dims=2)); sentinel=-1/τ)
    return exp.(cholesky(Diagonal(nz) + isnz(A) + τ * I) \ sumlogA)
end

function symcover_objective(α, A::AbstractMatrix, logA::AbstractMatrix{T}) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_objective requires a square matrix"))
    axes(A) == axes(logA) || throw(ArgumentError("symcover_objective requires logA to have the same shape as A"))
    val = zero(T)
    for j in ax
        αj = α[j]
        for i in j:last(ax)
            iszero(A[i, j]) && continue
            val += (logA[i, j] - α[i] - αj)^2
        end
    end
    val / 2
end

function symcover_tropical(A::AbstractMatrix; exact::Bool=false, itermax=10, rtol=1e-4, pre=1//2, post=1//2)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    z = log(oneunit(eltype(A)))
    logA = [iszero(aij) ? z : log(abs(aij)) for aij in A]
    nz = vec(sum(!iszero, A; dims=2))
    sumlogA = vec(sum(logA; dims=2))
    n = length(ax)
    T = typeof(z)
    B = if !exact || all(==(n), nz)
        u = nz / sqrt(sum(nz))
        ShermanMorrisonMatrix{T}(Diagonal(nz), u, u)
    else
        PDMat(Diagonal(nz) + isnz(A))
    end

    # Start α at the unconstrained minimizer of the objective function
    α = αstar = B \ sumlogA
    α .= max.(α, diag(logA)/2)
    objval0 = symcover_objective(α, A, logA)
    atol = rtol * sqrt(objval0)
    pr = first(ax) - 1 => typemin(T)
    boundidx = fill((pr, pr), ax)
    iter = 0
    while iter < itermax
        tightest_sym!(boundidx, logA, α, A)
        Δα = zero(T)
        for j in ax
            for i in first(ax):j-1
                αibound, αjbound, αijbound = pairbounds(boundidx, logA, i, j)
                # FINISH ME
            end
        end
        # maxviol = typemin(T)
        # # @show α objval0
        # # display(logA)
        # # display(α .+ α')
        # for j in ax
        #     violpre, violpost = maxviolation(logA, α, A, j)
        #     # @show j violpre violpost
        #     Δα = max(pre * violpre, post * violpost)
        #     # abs(Δα) > 1e-8 && @show j violpre violpost Δα
        #     Δα > typemin(T) && (maxviol = max(maxviol, abs(Δα)))
        #     α[j] += Δα
        #     # display(α .+ α')
        # end
        # # @show α symcover_objective(α, A, logA)
        # abs(maxviol) <= atol && break
        # # iter >= 3 && error("stop")
        # iter += 1
    end
    return exp.(α)
end

function tightest_sym!(boundidx, logA, α, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("tightest_sym! requires a square matrix"))
    axes(A) == axes(logA) || throw(ArgumentError("tightest_sym! requires logA to have the same shape as A"))
    eachindex(boundidx) == ax || throw(ArgumentError("tightest_sym! requires boundidx to have the same indices as the axes of A"))
    length(ax) >= 3 || throw(ArgumentError("tightest_sym! requires A to be at least 3x3"))
    sentinel = typemin(eltype(logA))
    for j in ax
        pr1 = pr2 = first(ax) - 1 => sentinel
        for i = first(ax):j-1
            Aij = A[i, j]
            iszero(Aij) && continue
            logAij = logA[i, j]
            Δ = logAij - α[i]
            if Δ > pr1.second
                pr2 = pr1
                pr1 = i => Δ
            elseif Δ > pr2.second
                pr2 = i => Δ
            end
        end
        boundidx[j] = (pr1, pr2)
    end
    return boundidx
end

function pairbounds(boundidx, logA, A, i, j)
    pr1i, pr2i = boundidx[i]
    pr1j, pr2j = boundidx[j]
    αibound = iszero(A[i, i]) ? typemin(eltype(logA)) : logA[i, i] / 2
    αibound = pr1i.first == j ? max(αibound, pr2i.second) : max(αibound, pr1i.second)
    αjbound = iszero(A[j, j]) ? typemin(eltype(logA)) : logA[j, j] / 2
    αjbound = pr1j.first == i ? max(αjbound, pr2j.second) : max(αjbound, pr1j.second)
    return αibound, αjbound, iszero(A[i, j]) ? typemin(eltype(logA)) : logA[i, j]
end

function maxviolation(logA, α, A, j)
    ax = axes(A, 1)
    violpre = typemin(eltype(logA))
    violpost = typemin(eltype(logA))
    αj = α[j]
    for i in first(ax):j
        Aij = A[i, j]
        violpre = max(violpre, iszero(Aij) ? typemin(eltype(logA)) : logA[i, j] - α[i] - αj)
    end
    for i in j+1:last(ax)
        Aij = A[i, j]
        violpost = max(violpost, iszero(Aij) ? typemin(eltype(logA)) : logA[i, j] - α[i] - αj)
    end
    return violpre, violpost
end

function symcover_barrier(A::AbstractMatrix; exact::Bool=false, τ=1.0, μ=1, τminfrac = 1//8, rtol=2*sqrt(eps(float(eltype(A)))), atol=0, itermax=50, btmax=5)
    @assert issymmetric(A)  # will generalize later
    ax = axes(A, 1)
    n = length(ax)
    W = isnz(A)
    z = log(oneunit(eltype(A)))
    T = typeof(z)
    logA = [iszero(aij) ? z : log(abs(aij)) for aij in A]
    nz = vec(sum(W; dims=2, init=0))
    B = if !exact || all(==(n), nz)
        u = nz / sqrt(sum(nz))
        ShermanMorrisonMatrix{T}(Diagonal(nz), u, u)
    else
        PDMat(Diagonal(nz) + W)
    end

    sumlogA = vec(sum(logA; dims=2))
    # Start α at the unconstrained minimizer of the objective function
    α = B \ sumlogA
    r = residual(logA, α, W)
    N = length(r)
    J = JacLNXS(jopsym(A, N), zeros(T, 0, length(α)))
    λ, τ = warmstart(r, J.Ji, B)
    # @show λ τ
    iszero(τ) && return exp.(α)
    α = B \ (sumlogA - J.Ji' * λ)   # correct α to satisfy the constraints
    s = τ ./ λ
    xs = XS(α, s)
    λν = LN(λ, T[])
    H = HessXS(B, xs, λν)
    g₀ = J.Ji'*r
    gxs = TopBottomVector(g₀, -τ ./ s)
    cviol = TopBottomVector(r + s, T[])
    Δxs0 = zero(TopBottomVector(xs))
    ws = TrimrWorkspace(KrylovConstructor(gxs, cviol))
    gxstmp, cvioltmp, Δxs = similar(gxs), similar(cviol), similar(gxs)
    xstmp, rtmp = similar(xs), similar(r)
    λνnew, Δλν = TopBottomVector(similar(λ), T[]), TopBottomVector(similar(λ), T[])
    iter = 0
    # @show α
    while iter < itermax
        # println("\nIteration $iter:")
        objval = sum(abs2, r) / 2
        gxstmp .= .-gxs
        cvioltmp .= .-cviol
        trimr!(ws, J', gxstmp, cvioltmp, Δxs0, λν; ν=0.0, τ=1.0, M=H, ldiv=true, atol = 0.0, rtol = 10 * sqrt(eps(T))) #, verbose=100)
        # println("  trimr iter = $(ws.stats.niter), solved = $(ws.stats.solved)")
        Δxs, λνnew = ws.x, ws.y
        mul!(gxstmp, J', λνnew, -1, true)
        # Compute convergence criteria
        δpar = abs(dot(gxstmp, Δxs))
        δperp = dotabs(cviol, λνnew)
        # @show (δpar, δperp, τ * abs(sum(log, s)), rtol * objval + atol)
        max(δpar, δperp, τ * abs(sum(log, s))) <= rtol * objval + atol && break

        # Diagonal backtracking
        Δλν .= λνnew .- λν
        merit0 = objval - τ * sum(log, s) + dot(λνnew, cviol) + μ * dotabs(λνnew, cviol)
        γ = γ0 = maxstep(top(Δλν), top(λν), maxstep(bottom(Δxs), bottom(xs)))
        iterbt = 0
        while iterbt < btmax
            xstmp .= xs .+ γ .* Δxs
            residual!(rtmp, logA, top(xstmp), W)
            merit = sum(abs2, rtmp) / 2 - τ * sum(log, bottom(xstmp))
            rtmp .+= bottom(xstmp)
            merit += dot(λνnew, rtmp) + μ * dotabs(λνnew, rtmp)
            merit < merit0 && break
            iterbt += 1
            γ /= (1 + iterbt)
        end
        iterbt == btmax && (γ = zero(γ))
        iszero(γ) && break
        # println("  γ = $γ, γ0 = $γ0")
        # Update the solution and barrier parameter
        α .+= γ .* top(Δxs)
        s .+= γ .* bottom(Δxs)
        λ .= (1 - γ) .* top(λν) .+ γ .* top(λνnew)
        @assert all(>(0), s) && all(>(0), λ) "Nonpositivity of s or λ: s = $s, λ = $λ"
        τ *= sqrt(1 - min(γ, 1 - τminfrac^2))
        # @show τ
        # @show α s λ τ
        iter += 1
        residual!(r, logA, α, W)
        mul!(top(gxs), J.Ji', r)
        bottom(gxs) .= -τ ./ s
        H.λsratio .= λ ./ s
        top(cviol) .= r .+ s
    end
    # @show iter λ τ
    # @show α
    return exp.(α)
end

function symnz(W)
    ax = axes(W, 1)
    k = 0
    for j in ax
        for i in j:last(ax)
            W[i, j] && (k += 1)
        end
    end
    return k
end

function residual!(r, logA, α::AbstractVector, W::AbstractMatrix{Bool})
    ax = axes(logA, 1)
    @assert axes(logA, 2) == ax
    k = firstindex(r) - 1
    for j in ax
        for i in j:last(ax)
            W[i, j] || continue
            r[k += 1] = logA[i, j] - α[i] - α[j]
        end
    end
    @assert k == lastindex(r)
    return r
end
function residual!(r, logA, α::AbstractVector, β::AbstractVector, W::AbstractMatrix{Bool})
    k = firstindex(r) - 1
    for j in axes(logA, 2)
        βj = β[j]
        for i in axes(logA, 1)
            W[i, j] || continue
            r[k += 1] = logA[i, j] - α[i] - βj
        end
    end
    @assert k == lastindex(r)
    return r
end

function residual(logA, α::AbstractVector, W::AbstractMatrix{Bool}, N::Int=symnz(W))
    T = promote_type(eltype(logA), eltype(α))
    return residual!(Vector{T}(undef, N), logA, α, W)
end
function residual(logA, α::AbstractVector, β::AbstractVector, W::AbstractMatrix{Bool}, N::Int=sum(W))
    T = promote_type(eltype(logA), eltype(α), eltype(β))
    return residual!(Vector{T}(undef, N), logA, α, β, W)
end

function jopsym(A, N)
    ax = axes(A, 1)
    n = length(ax)
    function jmul!(out::AbstractVector{T}, v, α, β) where T    # mul!(out, J, v, α, β)
        if iszero(β)
            fill!(out, zero(T))
        elseif β != 1
            out .*= β
        end
        k = 0
        for j in ax
            for i in j:last(ax)
                iszero(A[i, j]) && continue
                out[k += 1] -= α * (v[i] + v[j])
            end
        end
        @assert k == length(out)
        return out
    end
    function jmult!(out::AbstractVector{T}, v, α, β) where T    # mul!(out, J', v, α, β)
        if iszero(β)
            fill!(out, zero(T))
        elseif β != 1
            out .*= β
        end
        k = 0
        for j in ax
            for i in j:last(ax)
                iszero(A[i, j]) && continue
                k += 1
                vk = α * v[k]
                out[i] -= vk
                out[j] -= vk
            end
        end
        @assert k == length(v)
        return out
    end
    S = float(eltype(A))
    return LinearOperator{S,Vector{S}}(N, n, false, false, jmul!, jmult!, jmult!)
end

function warmstart(r::AbstractVector{T}, J, B; itermax=5, rtol=cbrt(eps(float(T)))) where T<:Real
    all(<=(zero(T)), r) && return fill(zero(T), length(r)), zero(T)
    # Initially, assume all λᵢ = λ for scalar λ, and solve for λ and τ
    w = r .> zero(T)
    h = -(w'*r)
    Δx = J' * w
    H = dot(Δx, B \ Δx)
    mI, nw = length(r), sum(w)
    λs, τ = warmstart1d(h, H, mI, nw)
    λ = fill(λs, mI)
    # Quasi-Newton refinement of λ and τ
    λτ = vcat(λ, τ)
    dualg, dualgtmp = similar(λτ), similar(λτ)
    λτtmp = similar(λτ)
    Jtλtmp, BdivJtλtmp, Jxtmp = similar(Δx), similar(Δx), similar(r)
    h = -r
    iter = 0
    while iter < itermax
        E0 = dualobjective!(dualg, λτ, h, J, B, Jtλtmp, BdivJtλtmp, Jxtmp)
        λ, τ = @view(λτ[begin:end-1]), last(λτ)
        Δλτ = - vcat(@view(dualg[begin:end-1]) .* λ.^2 ./ τ, dualg[end] * τ / length(h))  # preconditioned gradient step
        ϕϕ′ = let λτ = λτ, Δλτ = Δλτ, h = h, J = J, B = B, dualgtmp = dualgtmp, λτtmp = λτtmp
            function(t)
                λτtmp .= λτ .+ t .* Δλτ
                ϕt = dualobjective!(dualgtmp, λτtmp, h, J, B, Jtλtmp, BdivJtλtmp, Jxtmp)
                ϕt′ = dot(dualgtmp, Δλτ)
                return ϕt, ϕt′
            end
        end
        E0′ = dot(dualg, Δλτ)
        abs(E0′) <= rtol * abs(E0) && break
        t, converged = armijo_wolfe(ϕϕ′, E0, E0′, maxstep(Δλτ, λτ, typemax(one(eltype(λτ)))))
        # println("Iteration $iter: E = $E0, t = $t")
        converged || break
        λτ .+= t .* Δλτ
        iter += 1
    end
    return λτ[begin:end-1], last(λτ)
end

function dualobjective!(dualg, λτ, h, J, B, Jtλtmp, BdivJtλtmp, Jxtmp)
    λ, τ = @view(λτ[begin:end-1]), last(λτ)
    any(<=(zero(eltype(λ))), λ) && return typemax(eltype(λ))
    mul!(Jtλtmp, J', λ, true, false)
    ldiv!(BdivJtλtmp, B, Jtλtmp)
    if dualg !== nothing
        mul!(Jxtmp, J, BdivJtλtmp, true, false)
        dualg[begin:end-1] .= h .+ Jxtmp .- τ ./ λ
        dualg[end] = length(h) * log(τ) - sum(log, λ)
    end
    return h' * λ + dot(Jtλtmp, BdivJtλtmp) / 2 + length(h) * τ * (log(τ) - 1) - τ * sum(log, λ)
end

# This is copied, with modifications, from SolverTools.jl
function armijo_wolfe(
    ϕ,
    ϕ₀::T,
    ϕ₀′::T,
    tmax::T;
    t::T = one(T),
    τ₀::T = max(T(1.0e-4), sqrt(eps(T))),
    τ₁::T = T(0.9999),
    bk_max::Int = 10,
    bW_max::Int = 5,
) where {T <: AbstractFloat}
    # Perform improved Armijo linesearch.
    nbk = 0
    nbW = 0

    # First try to increase t to satisfy loose Wolfe condition
    ϕt, ϕt′ = ϕ(t)
    while (ϕt′ < τ₁ * ϕ₀′) && (ϕt <= ϕ₀ + τ₀ * t * ϕ₀′) && (nbW < bW_max)
        t *= 5
        t = min(t, tmax)
        ϕt, ϕt′ = ϕ(t)
        nbW += 1
    end

    ϕgoal = ϕ₀ + τ₀ * t * ϕ₀′
    fact = -T(8//10)
    ϵ = eps(T)^T(3 / 5)

    # Enrich Armijo's condition with Hager & Zhang numerical trick
    Armijo = (ϕt <= ϕgoal) || ((ϕt <= ϕ₀ + ϵ * abs(ϕ₀)) && (ϕt′ <= fact * ϕ₀′))
    while !Armijo && (nbk < bk_max)
        t *= 4//10
        ϕgoal = ϕ₀ + τ₀ * t * ϕ₀′
        ϕt, ϕt′ = ϕ(t)
        Armijo = (ϕt <= ϕgoal) || ((ϕt <= ϕ₀ + ϵ * abs(ϕ₀)) && (ϕt′ <= fact * ϕ₀′))
        nbk += 1
    end
    nbk < bk_max && @assert Armijo && ϕt′ >= τ₁ * ϕ₀′
    return t, Armijo
end


function warmstart1d(h::T, H::T, mI::Int, nw::Int; itermax=10, rtol=sqrt(eps(float(T)))) where T<:Real
    λsτ(τ) = (-h + sqrt(h^2 + 4 * τ * H * nw)) / (2 * H)
    function E(τ)
        λτ = λsτ(τ)
        iszero(τ) && return h*λτ + H * λτ^2 / 2
        return h*λτ + H * λτ^2 / 2 + mI * τ * (log(τ) - 1) - nw * τ * log(λτ)
    end
    # One-dimensional minimization of E with respect to τ
    τmin = zero(T)
    τ = h^2 / (4 * H * nw)   # location where τ starts to substantially affect the value of λ
    E0, Eτ = E(τmin), E(τ)
    if Eτ <= E0  # we may not have bracketed the minimum
        while true
            τlast, Elast = τ, Eτ
            τ *= 2
            Eτ = E(τ)
            Eτ > Elast && break
            τmin = τlast
        end
    end
    τmax = τ
    # Golden section search
    ϕ = (sqrt(5) + 1) / 2
    τ1 = τmax - (τmax - τmin) / ϕ
    τ2 = τmin + (τmax - τmin) / ϕ
    E1, E2 = E(τ1), E(τ2)
    iter = 0
    while iter < itermax && τmax - τmin > rtol * τmax
        if E1 < E2
            τmax, Eτ = τ2, E2
            τ2, E2 = τ1, E1
            τ1 = τmax - (τmax - τmin) / ϕ
            E1 = E(τ1)
        else
            τmin, Eτ = τ1, E1
            τ1, E1 = τ2, E2
            τ2 = τmin + (τmax - τmin) / ϕ
            E2 = E(τ2)
        end
        iter += 1
    end
    τ = (τmin + τmax) / 2
    return τ, λsτ(τ)
end


# Sherman-Morrison division
function divsm!(result::AbstractVector{T}, v::AbstractVector, nz::AbstractVector{Int}) where T<:Real
    sumnz = sum(nz)
    @assert sumnz > 0 "Cannot divide by zero: all rows are zero"
    offset = sum(v) / (2 * sumnz)
    for i in eachindex(result, v, nz)
        nzi = nz[i]
        result[i] = nzi == 0 ? typemin(T) : v[i] / nzi - offset
    end
    return result
end

function jactλ!(ξ, W, λ)  # ξ = J' * λ
    @assert issymmetric(W)  # will generalize later
    fill!(ξ, zero(eltype(ξ)))
    ax = axes(W, 1)
    for j in ax
        λj = λ[j]
        for i in ax
            wij = W[i, j]
            ξ[i] -= wij * λj
            ξ[j] -= wij * λ[i]
        end
    end
    return ξ
end

function solveΔs!(Δs, W, Δα)   # Δs = -J * Δα
    @assert issymmetric(W)  # will generalize later
    fill!(Δs, zero(eltype(Δs)))
    ax = axes(W, 1)
    for j in ax
        Δαj = Δα[j]
        for i in ax
            wij = W[i, j]
            Δs[i] += wij * Δαj
            Δs[j] += wij * Δα[i]
        end
    end
    return Δs
end

function maxstep(Δs, s, steplen=one(eltype(Δs)); η=1//16)
    for i in eachindex(s, Δs)
        Δsi = Δs[i]
        if Δsi < 0
            si = s[i]
            @assert si > zero(si) "base point is nonpositive: s = $s, Δs = $Δs"
            steplen = min(steplen, -(1 - η) * si / Δsi)
        end
    end
    @assert steplen > zero(steplen) "no positive step length: s = $s, Δs = $Δs"
    return steplen
end

"""
    a, b = cover(A; exact=false, regularize=false)

Given a matrix `A`, return vectors `a` and `b` representing the "scale of each
axis," so that `|A[i,j]| ~ a[i] * b[j]` for all `i, j`. `a[i]` and `b[j]` are
nonnegative, and are zero only if `A[i, j] = 0` for all `j` or all `i`,
respectively.

With `exact=true`, `a` and `b` solve the optimization problem

    min ∑_{i,j : A[i,j] ≠ 0} (log(|A[i,j]| / (a[i] * b[j])))²
    s.t. ∑_i nA[i] * log(a[i]) = ∑_j mA[j] * log(b[j])

where `nA` and `mA` are the number of nonzeros in each row and column,
respectively. Up to multiplication by a scalar, these vectors are covariant
under changes of scale but not general linear transformations.

With `exact=false`, the pattern of nonzeros in `A` is approximated as `u * v'`,
where `sum(u) * v[j] = mA[j]` and `sum(v) * u[i] = nA[i]`. This results in an
`O(m*n)` rather than `O((m+n)^3)` algorithm.
"""
function cover(A::AbstractMatrix; exact::Bool=false, regularize::Bool=false)
    Base.require_one_based_indexing(A)
    ax1, ax2 = axes(A, 1), axes(A, 2)
    (s, ns), (t, mt) = _cover(A, ax1, ax2)
    m, n = length(ax1), length(ax2)
    if !exact || (all(==(n), ns) && all(==(m), mt))
        z = sum(ns)
        @assert sum(mt) == z "Inconsistent nonzero counts in rows and columns"
        offsets, offsett = sum(s) / (2z), sum(t) / (2z)
        divsafe!(s, ns)
        divsafe!(t, mt)
        a = exp.(s ./ ns .- offsets)
        b = exp.(t ./ mt .- offsett)
        return a, b
    end
    p = vcat(ns, -mt)
    W = isnz(A)
    T = promote_type(eltype(s), eltype(t))
    τ = regularize ? sqrt(eps(T)) : zero(T)
    divsafe!(s, vec(sum(W; dims=2)); sentinel=-1/τ)
    divsafe!(t, vec(sum(W; dims=1)); sentinel=-1/τ)
    a12 = exp.(cholesky(Diagonal(vcat(ns, mt)) + odblocks(W) + p * p' + τ * I) \ vcat(s, t))
    return a12[begin:begin+m-1], a12[m+begin:end]
end


ratio_nz(n, d) = iszero(d) ? zero(n) / oneunit(d) : n / d

"""
    κ = condscale(A; exact=true)

Given a symmetric matrix `A`, return the condition number of the scaled matrix

    A ./ (a .* a')

where `a = symcover(A; exact)`.

This is a scale-invariant estimate of the condition number of `A`.
"""
function condscale(A; exact=true)
    a = symcover(A; exact)
    return cond(A ./ (a .* a'))
end

"""
    a, mag = divmag(A, b; cond::Bool=false, kwargs...)

Given a symmetric matrix `A` and vector `b`, for `x = A \\ b` return a pair
where `mag` is a naive estimate of the magnitude of `sum(abs.(x .* a))`. `a` and
`mag` are scale-covariant in circumstances where `A \\ b` is contravariant. With
`cond=false`, the estimate is based only on the magnitudes of the numbers in `A`
and `b`, and does not account for the conditioning of `A` or cancellation in the
solution process. Any `kwargs` are passed to [`symcover`](@ref).

This can be used to form scale-invariant estimates of roundoff errors in
computations involving `A`, `b`, and `x`.
"""
function divmag(A, b; cond::Bool=false, exact=cond)
    a = symcover(A; exact)
    κ = cond ? LinearAlgebra.cond(A ./ (a .* a')) : 1
    return a, κ * sum(abs ∘ splat(ratio_nz), zip(b, a))
end

# function diagapprox(A; iter=30)
#     ax = axes(A, 1)
#     axes(A, 2) == ax || throw(ArgumentError("diagapprox requires a square matrix"))
#     first(ax) == 1 || throw(ArgumentError("diagapprox requires 1-based indexing"))
#     n = length(ax)
#     T = float(eltype(A))
#     d = zeros(T, ax)
#     for _ in 1:iter
#         x = log.(rand(T, n)) .* (rand(Bool, n) .- T(0.5))
#         y = A * x
#         d .+= abs.(y) ./ abs.(x)
#     end
#     return d / iter
# end

end # module
