# Size sweep across the cover solvers: median-of-3 wall times on lognormal dense
# symmetric matrices (σ=3) and sparse symmetric matrices (~5 nonzeros/row), from
# the fast heuristics through the native *_min solvers to the JuMP/Ipopt
# reference solvers (which hit their practical size limit around n ≈ 100).
#
# Run from the repository root with
#     julia --project=bench bench/sizesweep.jl
# First use (creates bench/Manifest*.toml, which is gitignored):
#     julia --project=bench -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
using ScaleInvariantAnalysis, SparseArrays
using JuMP, HiGHS, Ipopt   # triggers the SIAJuMP and SIAIpopt solver extensions
using StableRNGs: StableRNG
using Statistics: median
using Printf

densesym(n) = (rng = StableRNG(n); B = exp.(3 .* randn(rng, n, n)) .* randn(rng, n, n); (B + B') / 2)
sparsesym(n) = (rng = StableRNG(n); S = sprandn(rng, n, n, min(0.5, 5 / n)); S + S')

function timeit(f, A)
    f(A)                       # warmup at this size (compile paths, allocate)
    return median([@elapsed f(A) for _ in 1:3])
end

# (label, solver, family constructor, sizes)
const CASES = [
    ("symcover (heuristic)",          A -> symcover(A),                                  densesym,  (30, 100, 300, 1000)),
    ("symcover (heuristic)",          A -> symcover(A),                                  sparsesym, (300, 1000, 3000)),
    ("cover (heuristic)",             A -> cover(A),                                     densesym,  (30, 100, 300, 1000)),
    ("soft_symcover AbsLog{2}",       A -> soft_symcover(AbsLog{2}(), A),                densesym,  (30, 100, 300, 1000)),
    ("soft_symcover AbsLog{1}",       A -> soft_symcover(AbsLog{1}(), A),                densesym,  (30, 100, 300)),
    ("soft_symcover AbsLinear{2}",    A -> soft_symcover(A),                             densesym,  (30, 100, 300)),
    ("soft_cover AbsLinear{2}",       A -> soft_cover(A),                                densesym,  (30, 100, 300)),
    ("soft_cover AbsLinear{1}",       A -> soft_cover(AbsLinear{1}(), A),                densesym,  (30, 100, 300)),
    ("symcover_min AbsLog{2} dense",  A -> symcover_min(AbsLog{2}(), A),                 densesym,  (30, 100, 200)),
    ("symcover_min AbsLog{2} lsqr",   A -> symcover_min(AbsLog{2}(), A; linsolve=:lsqr), sparsesym, (100, 300, 1000)),
    ("cover_min AbsLog{2} dense",     A -> cover_min(AbsLog{2}(), A),                    densesym,  (30, 100, 200)),
    ("symcover_min AbsLog{1} HiGHS",  A -> symcover_min(AbsLog{1}(), A),                 densesym,  (30, 100)),
    ("symcover_min AbsLinear{2} Ipopt", A -> symcover_min(AbsLinear{2}(), A),            densesym,  (30, 60)),
    ("soft_symcover_min AbsLog{2} HiGHS", A -> soft_symcover_min(AbsLog{2}(), A),        densesym,  (30, 100)),
]

for (label, f, fam, sizes) in CASES
    for n in sizes
        A = fam(n)
        t = try
            timeit(f, A)
        catch err
            @printf("%-36s %-9s n=%-5d ERROR: %s\n", label, fam === densesym ? "dense" : "sparse", n, sprint(showerror, err))
            continue
        end
        @printf("%-36s %-9s n=%-5d %10.4f s\n", label, fam === densesym ? "dense" : "sparse", n, t)
    end
end
