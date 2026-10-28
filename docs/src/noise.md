```@meta
CurrentModule = MatrixCovers
```

# [Noise covers and biwhitening](@id noise-covers)

The covers on the other pages are computed from the entries of a matrix. A
*noise cover* is computed from the variances of the noise in a data matrix
`Y`. Given a nonnegative matrix `V` of variance estimates, the same size as
`Y`, [`noise_cover`](@ref) returns scales `a` and `b` such that the normalized
data `Y ./ (a * b')` has noise whose variance averages one over every row and
every column. This is the *biwhitening* of
[Landa, Zhang & Kluger (2022)](#noise-references). After biwhitening, the
singular values of pure noise follow the Marchenko–Pastur law, so components
of the data that rise above the noise can be identified and estimated with
standard tools.

## What `noise_cover` computes

Let `V` be `m × n`, and let `m′` and `n′` count the rows and columns of `V`
that are not entirely zero. Then

```math
\sum_j \frac{V_{ij}}{(a_i b_j)^2} = n' \quad\text{and}\quad
\sum_i \frac{V_{ij}}{(a_i b_j)^2} = m'
```

for every such row `i` and column `j`. A row or column of `V` that is entirely
zero gets a scale of zero, as in every cover. The sums run over all entries,
zeros included, so each row of the normalized noise has mean variance one
across all `n′` columns, not just across its nonzero entries.

```julia
using MatrixCovers, LinearAlgebra

a, b, stats = noise_cover(V)
Yhat = Diagonal(pinv.(a)) * Y * Diagonal(pinv.(b))
```

Dividing via `Diagonal` preserves any sparsity of `Y`, and `pinv(0) == 0`
provides the convention `0/0 = 0` for rows and columns with a zero scale.

`noise_cover` is related to, yet distinct from, soft covers with `PowerMean`:

- In terms of variances, `(a .^ 2, b .^ 2)` scales `V` to the row sums `n′` and
  column sums `m′`. The [`PowerMean`](@ref)`{1}` soft cover performs the same
  kind of scaling, but its targets count only the nonzero entries of each row
  and column. When `V` has no zeros the two coincide (up to the gauge
  `a → t a`, `b → b / t`); when `V` has zeros they differ, and the section on
  [zeros](@ref noise-zeros) explains why the difference matters for count data.
- In terms of the noise `E`, `(a, b)` satisfies the balance condition of the
  [`PowerMean`](@ref)`{2}` soft cover in expectation, with every entry
  counted: the expected mean of `(E[i,j] / (a[i] b[j]))^2` is one over each
  row and column.

`noise_cover` uses the solver of [`soft_cover`](@ref) with these targets:
power-mean sweeps, Newton steps when the sweeps are slow, and the polish of
weakly joined clusters. It accepts the same solver keywords (`maxiter`,
`newton`, `maxnewton`, `maxpolish`, `linsolve`), and its `stats` is a
[`PowerMeanStats`](@ref MatrixCovers.PowerMeanStats). The Newton steps matter
most for zero patterns that only barely admit a scaling (see
[zeros](@ref noise-zeros)), where the sweeps alone converge slowly.

With `shrink = 0`, the cover is covariant: if `V` is replaced by
`Dr^2 * V * Dc^2` for positive diagonal `Dr` and `Dc`, then `a → Dr * a` and
`b → Dc * b`. Rescaling the units of `Y`, and transforming `V` to match, leaves
`Yhat` unchanged.

## Building the variance matrix

`noise_cover` uses `V` and nothing else, so the quality of the normalization
rests on `V`. It should be an unbiased, or nearly unbiased, estimate of the
variance of each entry of `Y`, computed from the same measurements as `Y`.

**Poisson counts.** For counts with Poisson noise, such as UMI counts in
single-cell RNA sequencing, the counts are their own unbiased variance
estimate:

```julia
a, b, stats = noise_cover(Y)
```

`Y` must be the raw counts: values that have already been normalized, such as
counts per million or log-transformed counts, no longer have variance equal to
their mean, and passing them as `V` gives scales without statistical meaning.

**Quadratic variance functions.** If `Var(Y[i,j]) = α + β X[i,j] + γ X[i,j]^2`
with `X = E[Y]`, then with `γ ≠ -1`

```julia
V = (α .+ β .* Y .+ γ .* Y .^ 2) ./ (1 + γ)
```

is unbiased. The negative binomial with dispersion `θ` has `α = 0`, `β = 1`,
`γ = 1/θ`. MatrixCovers does not estimate `α`, `β`, and `γ`; Landa et al.
choose them by fitting the normalized spectrum to the Marchenko–Pastur law.

**Camera data.** For a detector with gain `g` (digital units per detected
photon), offset `o`, and read-noise variance `σr²` (in squared digital units),
a pixel value `Y` has variance `g (E[Y] - o) + σr²`, so

```julia
V = g .* (Y .- o) .+ σr²
```

is unbiased. Read noise can make a few entries of this estimate negative;
`noise_cover` requires `V ≥ 0`, and clamping such entries (for example at
`σr²`) biases them upward slightly. When `Y` has been summed over frames, sum
the variances in the same way.

**Differences such as ΔF.** When the data are the change in fluorescence upon
stimulation, `ΔF = Fs - Fb`, the variance is that of the stimulus measurement
plus that of the baseline, each computed from the raw measurements as above:

```julia
V = Vs .+ Vb   # Vs from the raw stimulus frames, Vb from the raw baseline frames
```

The variance of `ΔF` depends on the brightness of `Fs` and `Fb`, not on `ΔF`
itself, which may be small or negative where the brightness is large. A model
fit to `ΔF`, such as the Poisson or quadratic estimates applied to `ΔF`, is
therefore wrong for these data.

The independence assumption also constrains how differences are formed. If
several rows of `ΔF` share one baseline measurement, for example a single
baseline for each cell subtracted from its response to every stimulus, the
noise in that baseline appears in all of those rows. The noise matrix then
contains a rank-one term whose singular value grows like `√(m n)`, far above
the threshold `√m + √n` used below, and no diagonal scaling removes it. It
will be counted as a component of the signal. Use an independent baseline for
each row where possible; otherwise expect one spurious component for each
shared baseline.

**Other divisions.** Dividing `Y` entrywise by `sqrt.(V)` does equalize the
noise variances, but it is not a diagonal scaling: it changes the rank of the
signal, which the methods below rely on. A diagonal scaling such as `ΔF/F0`,
with `F0` measured precisely for each cell, is harmless if `V` is divided by
`F0.^2` to match; with `shrink = 0` the result is the same as without it.

## [Zero entries](@id noise-zeros)

A zero in `V` can mean two different things:

- a *structural* zero, an entry that is absent, as in a contingency table
  with cells that cannot occur; or
- a *sampling* zero, an observation of a small expected count, as in sparse
  count data.

[`soft_cover`](@ref)`(PowerMean{1}(), V)` treats zeros as structural: only the
nonzero entries enter its targets. For count data this is the wrong choice. A
Poisson count with mean `λ`, conditioned on being nonzero, has mean
`λ / (1 - exp(-λ))`, which tends to one as `λ → 0`. In sparse count data most
nonzero entries are 1 or 2 regardless of their expected value, so a scaling
based on the nonzero entries alone barely responds to the intensity of a row
or column, the noise variances remain unequal after normalization, and the
Marchenko–Pastur law does not hold. `noise_cover` treats zeros as
observations, which is correct for counts.

Counting the zeros has a cost: for some zero patterns the scaling does not
exist. A row whose nonzero entries fall in only a few columns cannot have its
variance raised to the target `n′` without violating the targets of those
columns. More precisely, after rows and columns that are entirely zero are set
aside, the scaling exists exactly when some nonnegative matrix with the same
zero pattern as `V` has row sums `n′` and column sums `m′`. Roughly, every row
needs at least `n′/m′` nonzero entries and every column at least `m′/n′`, and
groups of rows that share few columns need proportionally more. `noise_cover`
checks this condition before iterating and, when it fails, throws a
[`SolverFailure`](@ref MatrixCovers.SolverFailure) that names the rows or
entries responsible. This is the counterpart, for matrix scaling, of
the nonexistence of a maximum-likelihood estimate in a log-linear model when
the table contains sampling zeros
([Fienberg & Rinaldo, 2012](#noise-references)).

When the scaling does not exist, there are two remedies:

- *Filter.* Remove rows and columns with too few nonzero entries. Standard
  quality control in single-cell analysis often suffices.
- *Shrink.* `noise_cover(V; shrink=δ)` with `δ > 0` scales
  `(V + δ μ) / (1 + δ)` instead of `V`, where `μ[i,j] = r[i] c[j] / s` with
  `r`, `c`, and `s` the row sums, column sums, and total of `V`. For Poisson
  counts, `μ` is the expected count under a rank-one model. The shrunk matrix
  has the same total as `V` and is positive on every nonvanishing row and
  column, so the scaling always exists. `μ` is applied inside the solver and
  never stored as a dense matrix, so the memory cost stays proportional to the
  number of stored entries of `V`.

For Poisson counts, `δ` is measured in observations. Give each expected count
`X[i,j]` a gamma prior with mean `μ[i,j]` and the weight of `δ` observations
(shape `δ μ[i,j]`, rate `δ`). Given the count `Y[i,j]`, the posterior mean of
`X[i,j]`, which is also the posterior mean of its variance, is
`(Y[i,j] + δ μ[i,j]) / (1 + δ)`. So `δ = 0.1` counts the rank-one model as a
tenth of an observation of each entry, and `δ = 1` weights it equally with the
data; equivalently, `δ μ[i,j]` is a pseudocount proportional to the entry's
expected count. Each variance estimate moves toward `μ[i,j]` by the fraction
`δ / (1 + δ)`, which biases it by `δ (μ[i,j] - X[i,j]) / (1 + δ)`. At `δ = 0`
the result is biwhitening; as `δ → ∞` the products `(a[i] b[j])^2` tend to
`μ[i,j]`, the normalization used by the analytic Pearson residuals of
[Lause, Berens & Kobak (2021)](#noise-references), which equalizes the noise
variances only when the expected counts have rank one. `δ` trades the
existence of the scaling against this bias, and MatrixCovers does not choose
it. Unlike `shrink = 0`,
a positive `shrink` is not covariant under rescaling of `V`, because `μ` is
not.

Some zeros are neither structural nor sampling zeros. In mass spectrometry, a
zero often means that a signal fell below the detection limit: the value is
censored, not observed. Neither treatment is correct for censored values, and
`noise_cover` makes no attempt to model them.

The Dyson equalizer of [Landa & Kluger (2025)](#noise-references) takes a
different route: it computes the scaling from the singular values of the data,
needs no variance estimate, and has no comparable existence condition.

## Using the result

The examples below assume

```julia
using MatrixCovers, LinearAlgebra

a, b, stats = noise_cover(V)
Yhat = Diagonal(pinv.(a)) * Y * Diagonal(pinv.(b))
m, n = count(!iszero, a), count(!iszero, b)
```

**Rank.** Pure noise with unit variance has singular values at most about
`√m + √n`, so the number of singular values of `Yhat` above that threshold
estimates the rank of the signal:

```julia
σ = svdvals(Matrix(Yhat))   # or a truncated SVD for large sparse data
k = count(>(sqrt(m) + sqrt(n)), σ)
```

The threshold is asymptotic, so components just above or below it are
uncertain. With `shrink > 0`, the total variance is unchanged, but its
distribution among the entries moves toward the rank-one fit, so the noise
variances are equalized only approximately.

**Embedding and clustering.** The leading singular vectors give an embedding
of the rows or columns in units of the noise:

```julia
F = svd(Matrix(Yhat))
Zrows = F.U[:, 1:k] .* F.S[1:k]'   # one row per row of Y
Zcols = F.V[:, 1:k] .* F.S[1:k]'   # one row per column of Y
```

Euclidean distances in this embedding are measured in units of the noise. The
noise contributes roughly `k` to the squared norm of each embedded unit, so a
unit whose squared norm is not well above `k` cannot be distinguished from
noise.

When units of the same type differ only in an overall gain, for example cells
of one type that differ in brightness or in sequencing depth, their embedded
vectors differ only in length. Normalizing each to unit length removes the
gain before clustering:

```julia
Zc = Zcols ./ norm.(eachrow(Zcols))
```

This removes a pure gain and nothing else. If units of the same type also
differ in sensitivity, for example in the concentration at which they respond,
or if the measurement saturates for the brightest units, their vectors differ
in direction as well. The type then occupies more than one dimension of the
embedding, the rank estimate exceeds the number of types, and a clustering
method that allows each type to span a subspace is more appropriate.

**Returning to the original units.** The scales are part of the result.
Multiplying an estimate made from `Yhat` by the scales returns it to the units
of `Y`. For example, a rank-`k` estimate of the signal is

```julia
Lhat = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.V[:, 1:k]'
Xhat = Diagonal(a) * Lhat * Diagonal(b)
```

and rows or columns with a zero scale are zero in `Xhat`. Quantities whose
shape matters, such as a dose–response curve along the rows for responses to
one ligand at several concentrations, should be computed from `Xhat` rather
than from `Yhat`: each row of `Yhat` carries its own scale `1/a[i]`, which
changes the shape of such a curve while leaving comparisons between columns
unaffected. Do not try to make these scales equal across related rows; rows
measured at different intensities or with different numbers of repeats have
different noise, and equalizing the noise requires different scales.

For omics count data, BiPCA ([Stanley et al., 2025](#noise-references))
provides a complete pipeline built on biwhitening, including the choice of the
variance model, rank estimation, and optimal shrinkage of the singular values.

## [References](@id noise-references)

- S. E. Fienberg and A. Rinaldo, "Maximum likelihood estimation in log-linear
  models," *Annals of Statistics* 40 (2012) 996–1023.
  [doi:10.1214/12-AOS986](https://doi.org/10.1214/12-AOS986)
- B. Landa and Y. Kluger, "The Dyson equalizer: adaptive noise stabilization
  for low-rank signal detection and recovery," *Information and Inference* 14
  (2025) iaae036.
  [doi:10.1093/imaiai/iaae036](https://doi.org/10.1093/imaiai/iaae036)
- B. Landa, T. T. C. K. Zhang, and Y. Kluger, "Biwhitening reveals the rank of
  a count matrix," *SIAM Journal on Mathematics of Data Science* 4 (2022)
  1420–1446. [doi:10.1137/21M1456807](https://doi.org/10.1137/21M1456807)
- J. Lause, P. Berens, and D. Kobak, "Analytic Pearson residuals for
  normalization of single-cell RNA-seq UMI data," *Genome Biology* 22 (2021)
  258. [doi:10.1186/s13059-021-02451-7](https://doi.org/10.1186/s13059-021-02451-7)
- J. S. Stanley III, J. Yang, R. Li, O. Lindenbaum, D. Kobak, B. Landa, and
  Y. Kluger, "Principled PCA separates signal from noise in omics count data,"
  bioRxiv (2025).
  [doi:10.1101/2025.02.03.636129](https://doi.org/10.1101/2025.02.03.636129)
