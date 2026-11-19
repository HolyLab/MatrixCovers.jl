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
        throw(ArgumentError("$fname requires floating-point or integer entries, got abs(A[$(string(i)), $(string(j))]) of type $(typeof(v))"))
    isfinite(v) ||
        throw(ArgumentError("$fname requires finite entries, got abs(A[$(string(i)), $(string(j))]) = $(string(v))"))
    return _ceil_log2(v)
end

# Exponent support of a symmetric matrix, by position, over the pairs
# `foreach_support_sym` visits: `ps[k] <= qs[k]`, and the entry `v` there has
# `es[k] = ceil(log2(v))` and `v = 2^fs[k] * exp(ms[k])` with `fs[k]` an
# integer and `0 <= ms[k] < log(2)`, up to the rounding of `ms[k]`.
function _exponent_support_sym(A::AbstractMatrix, fname::Symbol)
    o = first(axes(A, 1)) - 1
    ps, qs, es, fs, ms = Int[], Int[], Int[], Int[], Float64[]
    foreach_support_sym(A) do i, j, v
        push!(ps, i - o); push!(qs, j - o)
        push!(es, _entry_exponent(v, i, j, fname))
        fv = float(v)
        push!(fs, Int(exponent(fv))); push!(ms, log(Float64(significand(fv))))
        return nothing
    end
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
    for k in eachindex(ps, qs, es)
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
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative, got $maxiter"))
    ax = axes(A, 1)
    n = length(ax)
    ps, qs, es, fs, ms = _exponent_support_sym(A, fname)
    ptr, nbr, nex = _sym_adjacency(ps, qs, es, n)
    r = _sym_reference_exponents(ptr, nbr, nex, n)
    # `lv[k] = log|A[i,j]| - (r[i] + r[j]) * log(2)` does not change under
    # power-of-two rescaling, so the floating-point heuristic below, a
    # deterministic function of it, gives the same factors before and after
    # rescaling.
    lv = [(fs[k] - r[ps[k]] - r[qs[k]]) * log(2.0) + ms[k] for k in eachindex(fs, ms, ps, qs)]
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
function _pow2_factors!(a::AbstractVector, α::AbstractVector{Int}, fname::Symbol)
    T = eltype(a)
    T <: AbstractFloat ||
        throw(ArgumentError("$fname with pow2=true requires a floating-point factor type, got $T"))
    lo, hi = exponent(floatmin(T)), exponent(floatmax(T))
    for i in eachindex(a, α)
        e = α[i]
        if e == typemin(Int)
            a[i] = zero(T)
        else
            lo <= e <= hi ||
                throw(ArgumentError("$fname with pow2=true: the factor 2^$e for index $(string(i)) is not a normal $T; `MatrixCovers.symcover_exponents` returns the exponents themselves"))
            a[i] = ldexp(one(T), e)
        end
    end
    return a
end
