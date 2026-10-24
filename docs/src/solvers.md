```@meta
CurrentModule = MatrixCovers
```

# Starting points and convergence

## Starting points: initialize and refine

For the convex `AbsLog` and `PowerMean` penalties, the result does not depend
on the starting point. For the nonconvex `AbsLinear` penalty, it can. To give
control over the start, the interface separates initialization, refinement,
and selection among several starts:

- **Initializers** [`initialize_symcover`](@ref) and [`initialize_cover`](@ref)
  build a starting point from a named `strategy`. Their `feasible` keyword
  selects how the start is made into a hard cover: uniform inflation, selective
  boosting, or no feasibility step.
- **Refiners** are the `!`-suffixed forms of the solvers:
  [`symcover_min!`](@ref), [`cover_min!`](@ref), [`soft_symcover!`](@ref), and
  [`soft_cover!`](@ref) optimize a supplied point. Hard refiners require the
  supplied point to be a cover; soft refiners do not.
- **Solvers** [`symcover_min`](@ref), [`cover_min`](@ref),
  [`soft_symcover`](@ref), and [`soft_cover`](@ref) refine several starts and
  return the one with the best objective (for the convex penalties, one start
  suffices).

In general, plain forms choose their own starts, and `!` forms refine the
supplied start. The exceptions are [`symcover!`](@ref), [`cover!`](@ref), and
[`cover_transversal!`](@ref) (with or without a penalty), which write their
result in place.

The three calls below (the default multistart solver, the solver restricted to
the `:geomean` start, and an explicit initialization followed by refinement)
reach the same cover for this matrix:

```jldoctest manualstart
julia> using MatrixCovers, JuMP, Ipopt

julia> S = [4 1 0; 1 1 5; 0 5 2];

julia> a, stats = symcover_min(AbsLinear{2}(), S);

julia> round.(a; digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> a, stats = symcover_min(AbsLinear{2}(), S; strategies=(:geomean,));

julia> round.(a; digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> a0 = initialize_symcover(S; strategy=:geomean);

julia> symcover_min!(AbsLinear{2}(), a0, S);

julia> round.(a0; digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0
```

## Convergence and solver statistics

The optimizing solvers return statistics as their last output, reporting
whether the solver met its stopping criterion. The heuristics run a fixed
number of steps and return no statistics.

```jldoctest stats
julia> using MatrixCovers

julia> A = [1 2 3; 6 5 4];

julia> a, b, stats = cover_min(A);

julia> MatrixCovers.converged(stats)
true

julia> MatrixCovers.residual(stats) <= MatrixCovers.tolerance(stats)
true
```

Destructuring with `a, b = cover_min(A)` discards the statistics. For the
symmetric solvers, however, `a = symcover_min(A)` assigns the tuple
`(a, stats)` to `a`; write `a, stats = symcover_min(A)` instead.

Each solver family returns its own subtype of
[`AbstractCoverStats`](@ref MatrixCovers.AbstractCoverStats). The accessors
[`converged`](@ref MatrixCovers.converged),
[`iterations`](@ref MatrixCovers.iterations),
[`residual`](@ref MatrixCovers.residual), and
[`tolerance`](@ref MatrixCovers.tolerance) are public but not exported, and each
subtype implements those that apply to its solver.

A native solver that stops before meeting its criterion issues a warning and
returns its result; a hard cover returned this way still covers `A`. To reject
such results, test `MatrixCovers.converged(stats)`. The JuMP-backed solvers
throw a [`SolverFailure`](@ref MatrixCovers.SolverFailure) instead.
