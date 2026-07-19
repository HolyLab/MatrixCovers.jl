# Shared helpers for the themed test files.

# The four penalty instances, for loops over properties that must hold for every ϕ.
const PENALTIES = (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())

"""
    isbalanced(a, b, A; atol=1e-8)

The balance convention `∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` (`nzaᵢ`, `nzbⱼ` = the
nonzero counts of row `i` and column `j`), which fixes the row/column gauge
`a → c*a`, `b → b/c`. The gauge leaves every product `a[i]*b[j]` unchanged, so no
objective and no coverage constraint can see it; without a convention the split
between `a` and `b` would be an artifact of whichever pass last touched them. It is
imposed within each connected component of the bipartite support graph of `A`
separately, since the gauge acts independently on each; this checks the sum on
every component and returns `true` only if all of them balance. Every asymmetric
cover the package returns satisfies this.

Note it is *not* scale-invariant: rescaling `A` moves the balance point, which is why
[`covaries`](@ref) compares outer products rather than the vectors themselves.
"""
function isbalanced(a, b, A; atol=1e-8)
    rowcomp, colcomp, ncomp = MatrixCovers._support_components(A)
    iszero(ncomp) && return true
    La = zeros(ncomp)
    Lb = zeros(ncomp)
    or = first(axes(A, 1)) - 1
    oc = first(axes(A, 2)) - 1
    MatrixCovers.foreach_support(A) do i, j, v
        c = rowcomp[i-or]
        La[c] += log(a[i])
        Lb[c] += log(b[j])
    end
    return all(isapprox(La[c], Lb[c]; atol=atol * max(1, abs(La[c]), abs(Lb[c]))) for c in 1:ncomp)
end

"""
    covaries(coverfn, A, d; kwargs...)
    covaries(coverfn, A, dr, dc; kwargs...)

Scale covariance under diagonal rescaling. The symmetric form checks
`coverfn(d .* A .* d') ≈ d .* coverfn(A)`; a symmetric cover has no gauge
freedom, so the vectors must match directly. The general form checks the
outer product `a .* b'`, the gauge-invariant object under the free rescaling
`(a, b) → (c*a, b/c)`. Keyword arguments are forwarded to `isapprox`.

A converged cover pins the objective to `eps` but its own entries only to
`sqrt(eps)`: the objective is stationary at the minimizer, so a displacement
`δ` along a soft direction costs only `O(δ²)`. Solvers that iterate to
convergence therefore satisfy [`covaries_objective`](@ref) far more tightly
than `covaries`, and `rtol` here must leave room for `sqrt(eps)`.
"""
function covaries(coverfn, A, d::AbstractVector; kwargs...)
    a = coverfn(A)
    aD = coverfn(d .* A .* transpose(d))
    return isapprox(aD, d .* a; kwargs...)
end
function covaries(coverfn, A, dr::AbstractVector, dc::AbstractVector; kwargs...)
    a, b = coverfn(A)
    aD, bD = coverfn(dr .* A .* transpose(dc))
    return isapprox(aD .* transpose(bD), (dr .* a) .* transpose(dc .* b); kwargs...)
end

"""
    covaries_objective(ϕ, coverfn, A, d; kwargs...)
    covaries_objective(ϕ, coverfn, A, dr, dc; kwargs...)

Scale invariance of the attained objective: the cover found in a rescaled frame
must score the same as the cover found in the original frame, since `ϕ` sees only
the ratios `|A[i,j]| / (a[i]*b[j])`, which every diagonal rescaling leaves fixed.

This is the sharp statement of covariance for an iterate driven to convergence.
"""
function covaries_objective(ϕ, coverfn, A, d::AbstractVector; kwargs...)
    E = cover_objective(ϕ, coverfn(A), A)
    AD = d .* A .* transpose(d)
    return isapprox(cover_objective(ϕ, coverfn(AD), AD), E; kwargs...)
end
function covaries_objective(ϕ, coverfn, A, dr::AbstractVector, dc::AbstractVector; kwargs...)
    a, b = coverfn(A)
    E = cover_objective(ϕ, a, b, A)
    AD = dr .* A .* transpose(dc)
    aD, bD = coverfn(AD)
    return isapprox(cover_objective(ϕ, aD, bD, AD), E; kwargs...)
end
