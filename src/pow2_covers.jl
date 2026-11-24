# Heuristic covers whose factors are integer powers of two.
#
# A power-of-two cover of `A` is the same thing as an integer cover of the
# exponent matrix `E[i,j] = ceil(log2|A[i,j]|)`: `α[i] + α[j] >= E[i,j]` holds
# exactly when `2^α[i] * 2^α[j] >= |A[i,j]|`. `E` is computed exactly from the
# floating-point representation, and scaling `A` by a power-of-two diagonal
# `D = 2^k` shifts `E[i,j]` by exactly `k[i] + k[j]`. The solver works with
# integers wherever its result depends on the scale of `A`, so that the
# exponents shift by exactly `k`.

"""
    α = MatrixCovers.symcover_exponents(A; maxiter=3)

Return integer exponents `α` such that `a = 2 .^ α` is a hard cover of the
square, symmetric matrix `A`: `2^(α[i] + α[j]) >= abs(A[i, j])` for all `i`,
`j`. This is the solver behind [`symcover(A; pow2=true)`](@ref symcover), which
converts `α` to floating-point factors; calling it directly avoids the overflow
and underflow of `2^α[i]`.

Rows of `A` with no nonzero entry get `α[i] == typemin(Int)`, standing for the
zero factor `2^-∞`.

The method computes the exponents `E[i, j] = ceil(log2(abs(A[i, j])))` exactly
from the floating-point representation of the entries, so `A` must have real
or complex floating-point or integer entries, all finite.

`α` is exactly covariant under power-of-two rescaling whenever every connected
component of the support of `A` contains a nonzero diagonal entry: if
`D = Diagonal(2.0 .^ k)` for an integer vector `k`, and `D * A * D` is computed
without overflow or underflow, then `symcover_exponents(D * A * D) == α + k`.

# Extended help

Each row gets an integer reference exponent: `fld(E[i, i], 2)` for a row with a
nonzero diagonal entry, and otherwise, by breadth-first search outward from
those rows, the floored mean of `E[i, j] - r[j]` over neighbors `j` in the
preceding layer. Rows in a component with no nonzero diagonal entry get
reference zero. The residual matrix `|A[i, j]| / 2^(r[i] + r[j])` does not
change under power-of-two rescaling. The [`symcover`](@ref) heuristic (start,
boost, and `maxiter` tightening passes) is applied to the residual matrix, and
its factors are rounded up to powers of two. Entries that the rounded factors
miss because of floating-point round-off are then covered by raising both
endpoint exponents. A final pass visits the rows in order of how much rounding
enlarged them and lowers each exponent as far as coverage allows, so that
every supported row is tight on some entry: no single exponent can be
decreased.

Rounding each factor up to a power of two typically makes a cover product up
to four times larger than the corresponding product of [`symcover`](@ref).
Because `2^(2α[i])` is an even power of two, a diagonal entry with an odd
exponent `E[i, i]` is always covered with slack of at least a factor of two.
"""
function symcover_exponents(A::AbstractMatrix; maxiter::Int=3)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_exponents requires a square matrix"))
    require_abs_symmetric(A, :symcover_exponents)
    return _symcover_exponents(A, maxiter, :symcover_exponents)
end

# `ceil(log2(v))` for finite `v > 0`, without rounding.
_ceil_log2(v::AbstractFloat) = (e = Int(exponent(v)); isone(significand(v)) ? e : e + 1)
_ceil_log2(v::Integer) = isone(v) ? 0 : Int(ndigits(v - one(v); base=2))

function _entry_exponent(v, i, j, fname::Symbol)
    v isa Union{AbstractFloat,Integer} ||
        throw(ArgumentError("$fname requires floating-point or integer entries, got abs(A[$(string(i)), $(string(j))]) of type $(string(typeof(v)))"))
    isfinite(v) ||
        throw(ArgumentError("$fname requires finite entries, got abs(A[$(string(i)), $(string(j))]) = $(string(v))"))
    return _ceil_log2(v)
end

# Exponent support of `A`, by position, over the entries `foreach_support`
# visits (or the pairs `foreach_support_sym` visits, with `ps[k] <= qs[k]`,
# when `sym`). Entry `k` lies in row position `ps[k]` and column position
# `qs[k]`; its magnitude `v` has `es[k] = ceil(log2(v))` and
# `v = 2^fs[k] * exp(ms[k])` with `fs[k]` an integer and `0 <= ms[k] < log(2)`,
# up to the rounding of `ms[k]`.
function _exponent_support(A::AbstractMatrix, sym::Bool, fname::Symbol)
    or, oc = first(axes(A, 1)) - 1, first(axes(A, 2)) - 1
    ps, qs, es, fs, ms = Int[], Int[], Int[], Int[], Float64[]
    function record(i, j, v)
        push!(ps, i - or); push!(qs, j - oc)
        push!(es, _entry_exponent(v, i, j, fname))
        fv = float(v)
        push!(fs, Int(exponent(fv))); push!(ms, log(Float64(significand(fv))))
        return nothing
    end
    sym ? foreach_support_sym(record, A) : foreach_support(record, A)
    return ps, qs, es, fs, ms
end

# Compressed lists of every row's entries, by position; an off-diagonal pair is
# listed under both of its rows and a diagonal entry once.
function _sym_adjacency(ps::Vector{Int}, qs::Vector{Int}, es::Vector{Int}, n::Int)
    deg = zeros(Int, n)
    for k in eachindex(ps, qs)
        p, q = ps[k], qs[k]
        deg[p] += 1
        p == q || (deg[q] += 1)
    end
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    for p in 1:n
        ptr[p+1] = ptr[p] + deg[p]
    end
    nbr = Vector{Int}(undef, ptr[end] - 1)
    nex = Vector{Int}(undef, ptr[end] - 1)
    cursor = ptr[1:n]
    for k in _eachindex(ps, qs, es)
        p, q, e = ps[k], qs[k], es[k]
        nbr[cursor[p]], nex[cursor[p]] = q, e
        cursor[p] += 1
        if p != q
            nbr[cursor[q]], nex[cursor[q]] = p, e
            cursor[q] += 1
        end
    end
    return ptr, nbr, nex
end

# Integer reference exponents by position, each shifting by `k[p]` under a
# rescaling by `2^k`. A row with a nonzero diagonal entry takes `fld(E[p,p], 2)`;
# other rows reachable from one take the floored mean of `E[p,q] - r[q]` over
# neighbors `q` in the preceding breadth-first layer; all remaining rows take 0.
function _sym_reference_exponents(ptr::Vector{Int}, nbr::Vector{Int}, nex::Vector{Int}, n::Int)
    r = zeros(Int, n)
    dist = fill(-1, n)
    queue = Vector{Int}(undef, n)
    qend = 0
    for p in 1:n
        for s in ptr[p]:ptr[p+1]-1
            if nbr[s] == p
                r[p] = fld(nex[s], 2)
                dist[p] = 0
                qend += 1
                queue[qend] = p
            end
        end
    end
    # Rows leave the queue in order of distance, so the preceding layer of
    # each row has been assigned by the time it is reached.
    head = 1
    while head <= qend
        v = queue[head]
        head += 1
        if dist[v] > 0
            tot, cnt = 0, 0
            for s in ptr[v]:ptr[v+1]-1
                u = nbr[s]
                if dist[u] == dist[v] - 1
                    tot += nex[s] - r[u]
                    cnt += 1
                end
            end
            r[v] = fld(tot, cnt)
        end
        for s in ptr[v]:ptr[v+1]-1
            u = nbr[s]
            if dist[u] < 0
                dist[u] = dist[v] + 1
                qend += 1
                queue[qend] = u
            end
        end
    end
    return r
end

function _symcover_exponents(A::AbstractMatrix, maxiter::Int, fname::Symbol)
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative, got $(string(maxiter))"))
    ax = axes(A, 1)
    n = length(ax)
    ps, qs, es, fs, ms = _exponent_support(A, true, fname)
    ptr, nbr, nex = _sym_adjacency(ps, qs, es, n)
    r = _sym_reference_exponents(ptr, nbr, nex, n)
    # `lv[k] = log|A[i,j]| - (r[i] + r[j]) * log(2)` does not change under
    # power-of-two rescaling, so the floating-point heuristic below, a
    # deterministic function of it, gives the same factors before and after
    # rescaling.
    lv = [(fs[k] - r[ps[k]] - r[qs[k]]) * log(2.0) + ms[k] for k in _eachindex(fs, ms, ps, qs)]
    sup = FlatSupport(ps, qs, lv)
    x = zeros(Float64, n)
    unconstrained_min!(AbsLog{2}(), x, sup)
    boost_feasible!(x, sup)
    tighten_cover!(x, sup; maxiter)
    α = Vector{Int}(undef, n)
    loss = zeros(Float64, n)
    for p in 1:n
        if ptr[p] == ptr[p+1]
            α[p] = typemin(Int)
            continue
        end
        xp = x[p]
        floatmin(Float64) < xp < Inf ||
            throw(ArgumentError("$fname: the exponents of `A` span too wide a range relative to its diagonal for the heuristic start (row $(string(ax[p])))"))
        c = _ceil_log2(xp)
        α[p] = r[p] + c
        loss[p] = c - log2(xp)
    end
    _boost_exponents!(α, ptr, nbr, nex)
    # Each row is lowered as far as its entries allow, given the current
    # exponents of the other rows. Lowering a row only reduces the slack of
    # other rows, so after one pass every row is tight on some entry. Rows that
    # rounding raised most go first.
    for p in sortperm(loss; rev=true, alg=MergeSort)
        α[p] == typemin(Int) && continue
        d = typemax(Int)
        for s in ptr[p]:ptr[p+1]-1
            q, e = nbr[s], nex[s]
            d = min(d, q == p ? fld(2α[p] - e, 2) : α[p] + α[q] - e)
        end
        α[p] -= d
    end
    _require_exponent_cover(α, ptr, nbr, nex, ax, fname)
    out = similar(Array{Int}, ax)
    for (p, i) in enumerate(ax)
        out[i] = α[p]
    end
    return out
end

# The floating-point cover is feasible only up to round-off, which can leave
# an entry just above a power of two uncovered by one. Every endpoint of an
# entry with deficit `z` rises by at least `cld(z, 2)`, which covers it.
function _boost_exponents!(α::Vector{Int}, ptr::Vector{Int}, nbr::Vector{Int}, nex::Vector{Int})
    rise = zeros(Int, length(α))
    for p in eachindex(α)
        for s in ptr[p]:ptr[p+1]-1
            z = nex[s] - α[p] - α[nbr[s]]
            z > 0 && (rise[p] = max(rise[p], cld(z, 2)))
        end
    end
    α .+= rise
    return α
end

@noinline function _require_exponent_cover(α, ptr, nbr, nex, ax, fname)
    for p in eachindex(α)
        for s in ptr[p]:ptr[p+1]-1
            q = nbr[s]
            α[p] + α[q] >= nex[s] ||
                error("$fname: internal error, exponents fail to cover entry ($(string(ax[p])), $(string(ax[q])))")
        end
    end
    return nothing
end

# Convert cover exponents to factors of type `T`; `typemin(Int)` gives zero.
# `what` names the index in the error message.
function _pow2_factors!(a::AbstractVector, α::AbstractVector{Int}, fname::Symbol, what::String="index")
    T = eltype(a)
    T <: AbstractFloat ||
        throw(ArgumentError("$fname with pow2=true requires a floating-point factor type, got $(string(T))"))
    lo, hi = exponent(floatmin(T)), exponent(floatmax(T))
    for i in eachindex(a, α)
        e = α[i]
        if e == typemin(Int)
            a[i] = zero(T)
        else
            lo <= e <= hi ||
                throw(ArgumentError("$fname with pow2=true: the factor 2^$(string(e)) for $what $(string(i)) is not a normal $(string(T)); `MatrixCovers.$(fname === :cover ? :cover_exponents : :symcover_exponents)` returns the exponents themselves"))
            a[i] = ldexp(one(T), e)
        end
    end
    return a
end

"""
    α, β = MatrixCovers.cover_exponents(A; maxiter, cgiter, start=:auto)

Return integer exponents `α`, `β` such that `a = 2 .^ α`, `b = 2 .^ β` is a
hard cover of `A`: `2^(α[i] + β[j]) >= abs(A[i, j])` for all `i`, `j`. This is
the solver behind [`cover(A; pow2=true)`](@ref cover), which converts the
exponents to floating-point factors; calling it directly avoids the overflow
and underflow of `2^α[i]` and `2^β[j]`.

Rows and columns of `A` with no nonzero entry get exponent `typemin(Int)`,
standing for the zero factor `2^-∞`. The requirements on the entries of `A`
are those of [`symcover_exponents`](@ref).

`start`, `maxiter`, and `cgiter` have the meanings and defaults they have in
[`cover`](@ref), except that `start=:geomean` is not supported.

- `:transversal` returns a cover that is tight on every maximum-weight
  transversal of the exponent matrix `E[i, j] = ceil(log2(abs(A[i, j])))`; among
  power-of-two covers, these are the covers with the smallest `sum(α) + sum(β)`.
  It requires a square, structurally nonsingular `A`.
- `:unconstrained` applies to any `A`.
- `:auto` (the default) uses `:transversal` when `A` is square and
  structurally nonsingular, and `:unconstrained` otherwise.

The cover products are exactly covariant under power-of-two rescaling: if
`k` and `l` are integer vectors, `D1 = Diagonal(2.0 .^ k)`,
`D2 = Diagonal(2.0 .^ l)`, and `D1 * A * D2` is computed without overflow or
underflow, then `α2[i] + β2[j] == α[i] + β[j] + k[i] + l[j]` on the support of
`A`, where `α2, β2 = cover_exponents(D1 * A * D2)`. The exponents themselves
follow a balance convention that agrees with that of [`cover_min`](@ref) to
within one in each support component, so `α2 - α - k` is constant on each
support component but need not be zero.

# Extended help

The exponents `E` are computed exactly. With `start=:transversal`, a maximum-
weight transversal of `E` and integer optimal duals of the assignment problem
are found by shortest augmenting paths, which are exact on integer data. As in
[`cover`](@ref), tightness on every maximum-weight transversal leaves one
unknown per group of rows; the duals are first made canonical, independent of
the rescaling, by a layer-average start on the graph of groups with floored
means followed by the least feasible point above it. The floating-point stages
of [`cover`](@ref) then run on the slacks of these canonical duals, which do not
change under power-of-two rescaling. Their result is rounded to integers, raised
to the least feasible point above it, and given one integer Gauss–Seidel sweep.

With `start=:unconstrained`, each row and column gets an integer reference
exponent from a layer-average start on the support graph with floored means.
The residual matrix `|A[i, j]| / 2^(r[i] + c[j])` does not change under
power-of-two rescaling, apart from a shift between rows and columns that
leaves products unchanged; the [`cover`](@ref) heuristic is applied to it, and
its factors are rounded up to powers of two. Entries that the rounded factors
miss because of floating-point round-off are covered by raising their row and
column exponents, then each row exponent and after it each column exponent is
lowered as far as coverage allows, so that every supported row and column is
tight on some entry.
"""
function cover_exponents(A::AbstractMatrix; maxiter::Union{Int,Nothing}=nothing,
                         cgiter::Union{Int,Nothing}=nothing, start::Symbol=:auto)
    return _cover_exponents(A, maxiter, cgiter, start, :cover_exponents)
end

function _cover_exponents(A::AbstractMatrix, maxiter::Union{Int,Nothing}, cgiter::Union{Int,Nothing},
                          start::Symbol, fname::Symbol)
    _check_cover_start(start)
    caller = fname === :cover ? "cover with pow2=true" : string(fname)
    start === :geomean && throw(ArgumentError("$caller does not support start=:geomean"))
    if start === :auto || start === :transversal
        if length(axes(A, 1)) == length(axes(A, 2))
            tcg, tsweeps = something(cgiter, 20), something(maxiter, 5)
            tcg >= 0 || throw(ArgumentError("cgiter must be nonnegative, got $(string(tcg))"))
            tsweeps >= 0 || throw(ArgumentError("maxiter must be nonnegative, got $(string(tsweeps))"))
            r = _cover_exponents_transversal(A, tcg, tsweeps, fname)
            r === nothing || return r
            start === :transversal &&
                throw(ArgumentError("$caller with start=:transversal requires a structurally nonsingular matrix, but no transversal of nonzero entries exists"))
        elseif start === :transversal
            throw(ArgumentError("$caller with start=:transversal requires a square matrix, got size $(size(A))"))
        end
    end
    cgiter = something(cgiter, 16)
    maxiter = something(maxiter, 3)
    cgiter >= 0 || throw(ArgumentError("cgiter must be nonnegative, got $(string(cgiter))"))
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative, got $(string(maxiter))"))
    return _cover_exponents_unconstrained(A, cgiter, maxiter, fname)
end

# Exponent vectors indexed like `ax` from position-indexed `α`.
function _exponents_out(α::Vector{Int}, ax::AbstractUnitRange)
    out = similar(Array{Int}, ax)
    for (p, i) in enumerate(ax)
        out[i] = α[p]
    end
    return out
end

# Shift `α` up and `β` down by the same integer on each support component, so
# that `∑ na[i] α[i]` and `∑ nb[j] β[j]` differ by at most the component's
# entry count; this is the balance convention of `cover_min`, rounded to
# integers. `comp` labels rows `1:m` and columns `m+1:m+n`. The weighted sums
# use checked arithmetic, so exponents too large to balance throw an
# `OverflowError` instead of wrapping around.
function _balance_exponents!(α::Vector{Int}, β::Vector{Int}, na::Vector{Int}, nb::Vector{Int},
                             comp::Vector{Int}, ncomp::Int)
    m = length(α)
    Lα = zeros(Int, ncomp)
    Lβ = zeros(Int, ncomp)
    cnt = zeros(Int, ncomp)
    for p in eachindex(α, na)
        iszero(na[p]) && continue
        c = comp[p]
        Lα[c] = checked_add(Lα[c], checked_mul(na[p], α[p]))
        cnt[c] += na[p]
    end
    for q in eachindex(β, nb)
        iszero(nb[q]) && continue
        c = comp[m+q]
        Lβ[c] = checked_add(Lβ[c], checked_mul(nb[q], β[q]))
    end
    t = [iszero(cnt[c]) ? 0 : fld(checked_add(checked_sub(Lβ[c], Lα[c]), cnt[c]), 2 * cnt[c])
         for c in 1:ncomp]
    for p in eachindex(α, na)
        iszero(na[p]) || (α[p] = checked_add(α[p], t[comp[p]]))
    end
    for q in eachindex(β, nb)
        iszero(nb[q]) || (β[q] = checked_sub(β[q], t[comp[m+q]]))
    end
    return α, β
end

@noinline function _require_exponent_cover(α::Vector{Int}, β::Vector{Int}, ps, qs, es, axr, axc, fname)
    for k in _eachindex(ps, qs, es)
        p, q = ps[k], qs[k]
        α[p] + β[q] >= es[k] ||
            error("$fname: internal error, exponents fail to cover entry ($(string(axr[p])), $(string(axc[q])))")
    end
    return nothing
end

function _cover_exponents_unconstrained(A::AbstractMatrix, cgiter::Int, maxiter::Int, fname::Symbol)
    axr, axc = axes(A, 1), axes(A, 2)
    m, n = length(axr), length(axc)
    ps, qs, es, fs, ms = _exponent_support(A, false, fname)
    na = zeros(Int, m)
    nb = zeros(Int, n)
    for k in eachindex(ps, qs)
        na[ps[k]] += 1
        nb[qs[k]] += 1
    end
    rowptr, rowent = _entry_groups(ps, na, 0)
    colptr, colent = _entry_groups(qs, nb, 0)
    function foreach_neighbor(f, v)
        if v <= m
            for s in rowptr[v]:rowptr[v+1]-1
                k = rowent[s]
                f(m + qs[k], es[k])
            end
        else
            for s in colptr[v-m]:colptr[v-m+1]-1
                k = colent[s]
                f(ps[k], es[k])
            end
        end
        return nothing
    end
    # Integer reference exponents, rows `1:m` and columns `m+1:m+n`. Under a
    # rescaling by `2^k`, `2^l`, they shift by `k[i] + g` and `l[j] - g` with
    # `g` constant on each support component, so `ref[i] + ref[m+j]` shifts by
    # exactly `k[i] + l[j]`.
    ref = zeros(Int, m + n)
    comp = zeros(Int, m + n)
    ncomp = _bfs_levels!(ref, comp, na, nb, foreach_neighbor)
    # `lv[k] = log|A[i,j]| - (ref[i] + ref[m+j]) * log(2)` does not change under
    # power-of-two rescaling, so the floating-point heuristic below, a
    # deterministic function of it, gives the same factors before and after.
    lv = [(fs[k] - ref[ps[k]] - ref[m+qs[k]]) * log(2.0) + ms[k] for k in _eachindex(fs, ms, ps, qs)]
    sup = FlatSupport(ps, qs, lv)
    x = zeros(Float64, m)
    y = zeros(Float64, n)
    covariant_start!(x, y, sup; fname)
    _complete_support(sup, m, n) || cg_refine_start!(x, y, sup, cgiter; fname)
    boost_feasible!(x, y, sup)
    tighten_cover!(x, y, sup; maxiter)
    α = fill(typemin(Int), m)
    β = fill(typemin(Int), n)
    for p in 1:m
        iszero(na[p]) && continue
        floatmin(Float64) < x[p] < Inf ||
            throw(ArgumentError("$fname: the exponents of `A` span too wide a range for the heuristic start (row $(string(axr[p])))"))
        α[p] = ref[p] + _ceil_log2(x[p])
    end
    for q in 1:n
        iszero(nb[q]) && continue
        floatmin(Float64) < y[q] < Inf ||
            throw(ArgumentError("$fname: the exponents of `A` span too wide a range for the heuristic start (column $(string(axc[q])))"))
        β[q] = ref[m+q] + _ceil_log2(y[q])
    end
    # The floating-point cover is feasible only up to round-off. Splitting a
    # deficit `z` as `cld(z, 2)` to the row and `fld(z, 2)` to the column, and
    # raising each by its largest share, covers every entry.
    riseα = zeros(Int, m)
    riseβ = zeros(Int, n)
    for k in _eachindex(ps, qs, es)
        p, q = ps[k], qs[k]
        z = es[k] - α[p] - β[q]
        if z > 0
            riseα[p] = max(riseα[p], cld(z, 2))
            riseβ[q] = max(riseβ[q], fld(z, 2))
        end
    end
    α .+= riseα
    β .+= riseβ
    # Lower every row as far as the columns allow, then every column as far as
    # the rows allow. Lowering a column cannot loosen a row, so afterwards every
    # supported row and column is tight on some entry.
    newα = fill(typemin(Int), m)
    for k in _eachindex(ps, qs, es)
        p = ps[k]
        newα[p] = max(newα[p], es[k] - β[qs[k]])
    end
    α .= newα
    newβ = fill(typemin(Int), n)
    for k in _eachindex(ps, qs, es)
        q = qs[k]
        newβ[q] = max(newβ[q], es[k] - α[ps[k]])
    end
    β .= newβ
    _balance_exponents!(α, β, na, nb, comp, ncomp)
    _require_exponent_cover(α, β, ps, qs, es, axr, axc, fname)
    return _exponents_out(α, axr), _exponents_out(β, axc)
end

# One Gauss–Seidel sweep on integer group offsets: each vertex moves to the
# rounded mean of its targets, clamped to its feasible interval. A feasible `u`
# stays feasible.
function _tt_sweep_int!(u::Vector{Int}, Gg::_GroupGraph{Int})
    for v in eachindex(u)
        sl = _tt_slots(Gg, v)
        isempty(sl) && continue
        lo, hi, tot = typemin(Int), typemax(Int), 0
        for k in sl
            x = u[Gg.nbr[k]] + Gg.off[k]
            tot += x
            if Gg.lower[k]
                lo = max(lo, x)
            else
                hi = min(hi, x)
            end
        end
        len = length(sl)
        u[v] = max(min(fld(2tot + len, 2len), hi), lo)
    end
    return u
end

# Transversal-tight power-of-two cover of the square matrix `A`, as exponent
# vectors, or `nothing` when `A` has no transversal of nonzero entries. This
# follows `_cover_transversal_heuristic!` with the exponents `E` in place of
# `log|A|`; see the extended help of `cover_exponents`.
function _cover_exponents_transversal(A::AbstractMatrix, cgiter::Int, sweeps::Int, fname::Symbol)
    axr, axc = axes(A, 1), axes(A, 2)
    n = length(axr)
    oc = first(axc) - 1
    # Exponents stored as `Float64` keep the assignment solver's arithmetic
    # exact: the exponents of finite values, and their sums along a
    # transversal, are integers far below `2^53` in magnitude.
    G = _grouped_support(axr, eltype(axc), Float64) do g
        foreach_support(A) do i, j, v
            g(i, j, _entry_exponent(v, i, j, fname))
        end
    end
    mpt = _max_product_transversal(G, axc, fname; throw_singular=false, lg=identity)
    mpt === nothing && return nothing
    σ, α0f, _, _ = mpt
    τ = invperm(σ)
    ctrans = zeros(Int, n)
    nzrow = zeros(Int, n)
    nzcol = zeros(Int, n)
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            jp = G.idx[s] - oc
            jp == σ[ip] && (ctrans[jp] = Int(G.val[s]))
            nzrow[ip] += 1
            nzcol[jp] += 1
        end
    end
    # Difference system on the row exponents: entry `(i, j)` off the
    # transversal requires `α[i] - α[τ[j]] >= E[i,j] - ctrans[j]`.
    edges = Tuple{Int,Int}[]
    cvals = Int[]
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            jp = G.idx[s] - oc
            jp == σ[ip] && continue
            push!(edges, (ip, τ[jp]))
            push!(cvals, Int(G.val[s]) - ctrans[jp])
        end
    end
    comp, _, nc = _tight_components(edges, Float64.(cvals), α0f)
    relabel = zeros(Int, nc)
    k = 0
    for ip in 1:n
        c = comp[ip]
        relabel[c] == 0 && (relabel[c] = (k += 1))
        comp[ip] = relabel[c]
    end
    α0 = Int.(α0f)   # exact: the duals of integer data are integers
    eh, et, ed = Int[], Int[], Int[]
    for e in eachindex(edges, cvals)
        p, q = edges[e]
        comp[p] == comp[q] && continue
        d = cvals[e] - α0[p] + α0[q]
        d <= 0 || error("$fname: internal error, the transversal duals violate entry $(string(e))")
        push!(eh, comp[p])
        push!(et, comp[q])
        push!(ed, d)
    end
    # Optimal duals are unique only up to a feasible offset per group. The
    # floored layer-average start, raised to the least feasible point above
    # it, picks offsets that differ between any two optimal duals by a constant
    # on each support component, so the slacks below do not change under
    # power-of-two rescaling.
    U = zeros(Int, nc)
    gcomp = Vector{Int}(undef, nc)
    ncomp = _tt_layer_start!(U, gcomp, _GroupGraph(nc, eh, et, ed))
    _tt_raise!(U, _GroupGraph(nc, eh, et, ed))
    αc = [α0[ip] + U[comp[ip]] for ip in 1:n]
    edc = Int[]
    for e in eachindex(edges, cvals)
        p, q = edges[e]
        comp[p] == comp[q] || push!(edc, cvals[e] - αc[p] + αc[q])
    end
    Ggi = _GroupGraph(nc, eh, et, edc)
    Ggf = _GroupGraph(nc, eh, et, Float64.(edc))
    u = Vector{Float64}(undef, nc)
    _tt_layer_start!(u, gcomp, Ggf)
    _tt_cg!(u, Ggf, gcomp, ncomp, cgiter)
    _tt_boost!(u, Ggf)
    _tt_raise!(u, Ggf)
    _tt_sweep!(u, Ggf, sweeps)
    # Below 2^52 in magnitude, `x + 1/2` is exact, and `floor(x + 1/2)` commutes
    # with integer shifts and preserves the integer difference constraints
    # wherever `u` satisfies them.
    for x in u
        abs(x) < 2.0^52 ||
            error("$fname: internal error, the group offset $(string(x)) of the transversal start is too large to round exactly")
    end
    v = [Int(floor(x + 0.5)) for x in u]
    _tt_raise!(v, Ggi)
    _tt_sweep_int!(v, Ggi)
    α = [αc[ip] + v[comp[ip]] for ip in 1:n]
    β = [ctrans[jp] - α[τ[jp]] for jp in 1:n]
    # Every row and column of a structurally nonsingular matrix is supported;
    # the groups of a component of the group graph hold the rows and columns
    # of one support component.
    lab = Vector{Int}(undef, 2n)
    for ip in 1:n
        lab[ip] = gcomp[comp[ip]]
    end
    for jp in 1:n
        lab[n+jp] = gcomp[comp[τ[jp]]]
    end
    _balance_exponents!(α, β, nzrow, nzcol, lab, ncomp)
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            jp = G.idx[s] - oc
            e = Int(G.val[s])
            α[ip] + β[jp] >= e && (jp != σ[ip] || α[ip] + β[jp] == e) ||
                error("$fname: internal error, exponents fail to cover entry ($(string(i)), $(string(G.idx[s]))) or are not tight on the transversal")
        end
    end
    return _exponents_out(α, axr), _exponents_out(β, axc)
end
