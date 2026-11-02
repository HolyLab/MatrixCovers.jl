% Smoke test for the matrixcovers MATLAB package. Run from the build output
% directory after `build_mex`. Reference values are those of MatrixCovers.jl.

As = [4 1 0.5; 1 3 0.2; 0.5 0.2 2];
Aa = [1 2 3; 6 5 4; 0.5 0 2];
near = @(x, ref, rtol) all(abs(x(:) - ref(:)) <= rtol * abs(ref(:)));

a = matrixcovers.symcover(As);
assert(near(a, [2; sqrt(3); sqrt(2)], 1e-8), "symcover")
assert(matrixcovers.iscover_sym(a, As), "iscover_sym")

[a, b] = matrixcovers.cover(Aa);
assert(matrixcovers.iscover_ab(a, b, Aa), "cover covers")

[a, converged, iterations] = matrixcovers.symcover_min(As);
assert(near(a, [2.0; 1.7320508075688787; 1.4142135623730965], 1e-6), "symcover_min")
assert(converged && iterations > 0, "symcover_min stats")

[a, b, converged] = matrixcovers.cover_min(Aa, linsolve="dense");
assert(near(a, [1.225652917043481; 4.359106692464615; 0.8171019446954835], 1e-6) && converged, "cover_min")

[a, b, converged] = matrixcovers.soft_cover(Aa);
assert(near(a, [1.2507036306044128; 3.9635731170047523; 0.7270572991132858], 1e-6) && converged, "soft_cover")

% A solver without a stopping criterion reports [] for its statistics.
[a, converged] = matrixcovers.soft_symcover(As, penalty="abslog2");
assert(near(a, [1.363879622416882; 0.9130259591914891; 0.6330568425802215], 1e-6), "soft_symcover(abslog2)")
assert(isempty(converged), "soft_symcover(abslog2) stats")

V = [3 0 1; 0 2 5; 4 1 0];
[a, b] = matrixcovers.noise_cover(V);
assert(near(sum(V ./ (a(:) .* b(:).').^2, 2), [3; 3; 3], 1e-8), "noise_cover")

s = matrixcovers.gramcover_weighted(Aa, [1; 2; 3]);
assert(near(s, [8.587782018658846; 7.3484692283495425; 7.280109889280529], 1e-8), "gramcover_weighted")

failed = false;
try
    matrixcovers.symcover_min(As, penalty="abslinear2");
    failed = true;
catch err
    assert(contains(err.message, "Ipopt"), "extension-only penalty names Ipopt")
end
assert(~failed, "extension-only penalty raises an error")

disp("ALL CHECKS PASSED")
