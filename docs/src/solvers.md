```@meta
CurrentModule = MatrixCovers
```

# Starting points and convergence

## Starting points: initialize and refine

For objectives with multiple minima, the result can depend on its starting
point. The interface separates initialization, refinement, and multistart
selection:

- **Initializers** [`initialize_symcover`](@ref) and [`initialize_cover`](@ref)
  build a named `strategy`. Their `feasible` keyword selects uniform inflation,
  selective boosting, or no feasibility step.
- **Refiners** are the `!`-suffixed forms of the
  solvers: [`symcover_min!`](@ref), [`cover_min!`](@ref), [`soft_symcover!`](@ref),
  and [`soft_cover!`](@ref) optimize a supplied point. Hard refiners require a
  cover; soft refiners do not.
- **Solvers** [`symcover_min`](@ref), [`cover_min`](@ref),
  [`soft_symcover`](@ref), and [`soft_cover`](@ref) refine several starts and
  return the best objective
  (for the convex penalties, one start suffices).

Plain forms choose their starts; `!` forms refine the supplied start, except
[`symcover!`](@ref), [`cover!`](@ref), and [`cover_transversal!`](@ref), which
write their result in place.

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

For the convex `AbsLog` and `PowerMean` penalties, the start does not change the
result.

## Convergence and solver statistics

The optimizing solvers return statistics as their last output, reporting
whether the solver met its stopping criterion (the heuristics, which run a fixed
number of steps, return none):

```jldoctest stats
julia> using MatrixCovers

julia> A = [1 2 3; 6 5 4];

julia> a, b, stats = cover_min(A);

julia> MatrixCovers.converged(stats)
true

julia> MatrixCovers.residual(stats) <= MatrixCovers.tolerance(stats)
true
```

`a, b = cover_min(A)` discards the statistics, but `a = symcover_min(A)` binds
the pair; write `a, stats = symcover_min(A)`.

Each solver family returns its own subtype of
[`AbstractCoverStats`](@ref MatrixCovers.AbstractCoverStats). The accessors
[`converged`](@ref MatrixCovers.converged),
[`iterations`](@ref MatrixCovers.iterations),
[`residual`](@ref MatrixCovers.residual), and
[`tolerance`](@ref MatrixCovers.tolerance) are public but not exported, and each
subtype implements those that apply to its solver.

A native solver that stops before meeting its criterion warns and returns its
result; a hard cover returned this way still covers `A`. Test
`MatrixCovers.converged(stats)` to reject such results. The JuMP-backed solvers
throw a [`SolverFailure`](@ref MatrixCovers.SolverFailure) instead.

