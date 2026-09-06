# Dense-grid kernels for the heuristic covers.
#
# Full-grid storage needs only a log-magnitude array because its indices are
# implicit. Its accumulation order matches `FlatSupport`.

# Use the grid only when the support traversal is already dense and its
# allocation is worthwhile.
const DENSE_GRID_MIN = 64

_dense_grid_storage(::AbstractMatrix) = false
_dense_grid_storage(::StridedMatrix) = true
_dense_grid_storage(A::Union{Symmetric,Hermitian}) = _dense_grid_storage(parent(A))

const DENSE_GRID_FLOAT = Union{Float32,Float64}

_use_dense_grid(A::AbstractMatrix, ::Type{T}) where {T} =
    T <: DENSE_GRID_FLOAT && _dense_grid_storage(A) && minimum(size(A)) >= DENSE_GRID_MIN

# Slot of `L[1, j]` minus one in a column-packed upper triangle: column `j`
# occupies `_trioff(j)+1 : _trioff(j)+j`.
_trioff(j::Int) = (j * (j - 1)) >> 1

# Pack the upper triangle as log magnitudes, with `-Inf` for zeros.
function _tri_logabs!(Lp::Vector{T}, A::AbstractMatrix) where {T}
    ax = axes(A, 1)
    or = first(ax) - 1
    n = length(ax)
    for jp in 1:n
        j = jp + or
        o = _trioff(jp)
        for ip in 1:jp
            Lp[o+ip] = abs(A[ip+or, j])
        end
    end
    _fastlog!(Lp)
    return Lp
end

# Row sums of the diagonally normalized log magnitudes `Lp[i,j] - ρ[i] - ρ[j]`
# and the row support counts; see `_sym_unconstrained!`.
function _tri_normsums!(s::Vector{T}, cnt::Vector{Int}, Lp::Vector{T}, ρ::Vector{T}, n::Int) where {T}
    fill!(s, zero(T))
    fill!(cnt, 0)
    ninf = T(-Inf)
    for jp in 1:n
        o = _trioff(jp)
        rj = ρ[jp]
        sj = zero(T)
        cj = 0
        for ip in 1:jp-1
            l = Lp[o+ip]
            if l != ninf
                l -= ρ[ip] + rj
                s[ip] += l
                cnt[ip] += 1
                sj += l
                cj += 1
            end
        end
        s[jp] += sj
        cnt[jp] += cj
        l = Lp[o+jp]
        if l != ninf
            s[jp] += l - 2 * rj
            cnt[jp] += 1
        end
    end
    return s, cnt
end

# Fill a log-magnitude grid and the row and column summaries.
function _grid_logabs!(L::Matrix{T}, sa::Vector{T}, sb::Vector{T},
                       na::Vector{Int}, nb::Vector{Int}, A::AbstractMatrix) where {T}
    or = first(axes(A, 1)) - 1
    oc = first(axes(A, 2)) - 1
    m, n = size(A)
    for jp in 1:n
        j = jp + oc
        for ip in 1:m
            L[ip, jp] = abs(A[ip+or, j])
        end
    end
    _fastlog!(L)
    fill!(sa, zero(T))
    fill!(na, 0)
    ninf = T(-Inf)
    for jp in 1:n
        sj = zero(T)
        cj = 0
        for ip in 1:m
            l = L[ip, jp]
            if l != ninf
                sa[ip] += l
                na[ip] += 1
                sj += l
                cj += 1
            end
        end
        sb[jp] = sj
        nb[jp] = cj
    end
    return L
end

# Derive the balance summary directly for full support; traverse sparse support.
function _grid_components(A::AbstractMatrix, na::Vector{Int}, nb::Vector{Int}, m::Int, n::Int)
    for ip in 1:m
        na[ip] == n || return _support_components(A)
    end
    return ones(Int, m), ones(Int, n), 1, na, nb
end

# Use compact boost-list indices when the dimensions fit.
_grid_label(m::Int, n::Int) = max(m, n) <= typemax(Int32) ? Int32 : Int

# Select violated upper-triangle entries in traversal order without branching.
function _tri_violated(Lp::Vector{T}, lα::Vector{T}, n::Int, nviol::Int,
                       ::Type{IT}) where {T,IT}
    entries = Vector{Tuple{IT,IT,T}}(undef, nviol + 1)
    k = 1
    for jp in 1:n
        o = _trioff(jp)
        lj = lα[jp]
        for ip in 1:jp
            lv = Lp[o+ip]
            entries[k] = (ip % IT, jp % IT, lv)
            k += ifelse(lv - lα[ip] - lj > zero(T), 1, 0)
        end
    end
    resize!(entries, nviol)
    return entries
end

# The violated entries of a full grid, selected as in `_tri_violated`.
function _grid_violated(L::Matrix{T}, lα::Vector{T}, lβ::Vector{T}, m::Int, n::Int,
                        nviol::Int, ::Type{IT}) where {T,IT}
    entries = Vector{Tuple{IT,IT,T}}(undef, nviol + 1)
    k = 1
    for jp in 1:n
        lj = lβ[jp]
        for ip in 1:m
            lv = L[ip, jp]
            entries[k] = (ip % IT, jp % IT, lv)
            k += ifelse(lv - lα[ip] - lj > zero(T), 1, 0)
        end
    end
    resize!(entries, nviol)
    return entries
end

# The asymmetric start has no reference shift; see `_uncon_scale` in heuristic_covers.jl.
_uncon_scale(si::T, ni::Int, halfmu::T) where {T} = _uncon_scale(si, ni, halfmu, zero(T))

# Greedy boost that updates scales and log scales together.
function _dense_boost!(α::Vector{T}, lα::Vector{T}, entries, zmax::T) where {T}
    deficit((i, j, lv)) = lv - lα[i] - lα[j]
    function apply!((i, j, lv), z)
        h = z / 2
        lα[i] += h; α[i] = exp(lα[i])
        i == j || (lα[j] += h; α[j] = exp(lα[j]))
    end
    bucket_boost!(deficit, apply!, entries, T, zmax)
    return α
end

# `symcover!` over a packed upper-triangular log-magnitude grid.
function _symcover_dense!(a::AbstractVector, A::AbstractMatrix, ::Type{T}, maxiter::Int) where {T}
    ax = axes(A, 1)
    or = first(ax) - 1
    n = length(ax)
    Lp = Vector{T}(undef, _trioff(n) + n)
    α = Vector{T}(undef, n)
    lα = Vector{T}(undef, n)
    cnt = Vector{Int}(undef, n)
    _tri_logabs!(Lp, A)
    # Covariant reference from the diagonal; see `_sym_reference!`.
    ρ = Vector{T}(undef, n)
    for jp in 1:n
        l = Lp[_trioff(jp)+jp]
        ρ[jp] = ifelse(l == T(-Inf), T(NaN), l / 2)
    end
    _tri_normsums!(α, cnt, Lp, ρ, n)     # `α` carries the row log sums here
    nmissing = count(ip -> cnt[ip] > 0 && isnan(ρ[ip]), 1:n)
    if nmissing > 0
        # Rare path: some supported row has a zero diagonal entry.
        function foreach_entries(f)
            for jp in 1:n
                o = _trioff(jp)
                for ip in 1:jp
                    l = Lp[o+ip]
                    l == T(-Inf) || f(ip, jp, l)
                end
            end
        end
        _sym_reference!(ρ, foreach_entries, nmissing)
        _tri_normsums!(α, cnt, Lp, ρ, n)
    end
    # Unsupported rows may keep NaN references; `_uncon_scale` never reads them.
    nztotal = sum(cnt)
    halfmu = iszero(nztotal) ? zero(T) : sum(α) / (2 * nztotal)
    for ip in 1:n
        α[ip] = _uncon_scale(α[ip], cnt[ip], halfmu, ρ[ip])
        lα[ip] = log(α[ip])
    end

    # Only initially violated entries can require a boost.
    nviol = 0
    zmax = zero(T)
    for jp in 1:n
        o = _trioff(jp)
        lj = lα[jp]
        for ip in 1:jp
            z = Lp[o+ip] - lα[ip] - lj
            nviol += ifelse(z > zero(T), 1, 0)
            zmax = ifelse(z > zmax, z, zmax)
        end
    end
    # A supported zero scale produces an infinite deficit.
    isfinite(zmax) ||
        throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row"))
    _dense_boost!(α, lα, _tri_violated(Lp, lα, n, nviol, _grid_label(n, n)), zmax)

    lratio = Vector{T}(undef, n)
    for _ in 1:maxiter
        map!(log, lα, α)
        fill!(lratio, T(Inf))
        # `-Inf` slots produce `Inf` or `NaN` ratios, both ignored below.
        for jp in 1:n
            o = _trioff(jp)
            lj = lα[jp]
            mj = T(Inf)
            for ip in 1:jp-1
                lr = lα[ip] + lj - Lp[o+ip]
                lratio[ip] = ifelse(lr < lratio[ip], lr, lratio[ip])
                mj = ifelse(lr < mj, lr, mj)
            end
            lr = lj + lj - Lp[o+jp]
            mj = ifelse(lr < mj, lr, mj)
            lratio[jp] = ifelse(mj < lratio[jp], mj, lratio[jp])
        end
        for ip in 1:n
            lr = lratio[ip]
            # Infinite ratios require no update.
            isinf(lr) || (α[ip] = _tighten_shrink(α[ip], lr))
        end
    end
    for ip in 1:n
        a[ip+or] = α[ip]
    end
    return a
end

# `cover!` over a dense log-magnitude grid, up to but not including the balance
# convention.
function _cover_dense!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                       ::Type{T}, maxiter::Int, cgiter::Int) where {T}
    or = first(axes(A, 1)) - 1
    oc = first(axes(A, 2)) - 1
    m, n = size(A)
    L = Matrix{T}(undef, m, n)
    α = Vector{T}(undef, m)
    β = Vector{T}(undef, n)
    lα = Vector{T}(undef, m)
    lβ = Vector{T}(undef, n)
    na = Vector{Int}(undef, m)
    nb = Vector{Int}(undef, n)
    _grid_logabs!(L, α, β, na, nb, A)    # `α`, `β` carry the log sums here
    nztotal = sum(na)
    halfmu = iszero(nztotal) ? zero(T) : sum(α) / (2 * nztotal)
    for ip in 1:m
        α[ip] = _uncon_scale(α[ip], na[ip], halfmu)
        lα[ip] = log(α[ip])
    end
    for jp in 1:n
        β[jp] = _uncon_scale(β[jp], nb[jp], halfmu)
        lβ[jp] = log(β[jp])
    end
    if cgiter > 0
        function foreach_grid(f)
            for jp in 1:n, ip in 1:m
                l = L[ip, jp]
                isfinite(l) && f(ip, jp, l)   # -Inf marks a zero entry
            end
        end
        _cg_refine!(lα, lβ, foreach_grid, cgiter)
        for ip in 1:m
            iszero(na[ip]) || (α[ip] = max(exp(lα[ip]), floatmin(T)); lα[ip] = log(α[ip]))
        end
        for jp in 1:n
            iszero(nb[jp]) || (β[jp] = max(exp(lβ[jp]), floatmin(T)); lβ[jp] = log(β[jp]))
        end
    end

    nviol = 0
    zmax = zero(T)
    for jp in 1:n
        lj = lβ[jp]
        for ip in 1:m
            z = L[ip, jp] - lα[ip] - lj
            nviol += ifelse(z > zero(T), 1, 0)
            zmax = ifelse(z > zmax, z, zmax)
        end
    end
    isfinite(zmax) ||
        throw(ArgumentError("boost_feasible! requires a start with positive scale on every supported row/column"))
    entries = _grid_violated(L, lα, lβ, m, n, nviol, _grid_label(m, n))
    # Row and column scales require separate updates.
    deficit((i, j, lv)) = lv - lα[i] - lβ[j]
    function apply!((i, j, lv), z)
        h = z / 2
        lα[i] += h; α[i] = exp(lα[i])
        lβ[j] += h; β[j] = exp(lβ[j])
    end
    bucket_boost!(deficit, apply!, entries, T, zmax)

    ratioa = Vector{T}(undef, m)
    for _ in 1:maxiter
        map!(log, lα, α)
        map!(log, lβ, β)
        fill!(ratioa, T(Inf))
        for jp in 1:n
            lj = lβ[jp]
            mj = T(Inf)
            for ip in 1:m
                lr = lα[ip] + lj - L[ip, jp]
                ratioa[ip] = ifelse(lr < ratioa[ip], lr, ratioa[ip])
                mj = ifelse(lr < mj, lr, mj)
            end
            isinf(mj) || (β[jp] = _tighten_shrink(β[jp], mj))
        end
        for ip in 1:m
            lr = ratioa[ip]
            isinf(lr) || (α[ip] = _tighten_shrink(α[ip], lr))
        end
    end

    for ip in 1:m
        a[ip+or] = α[ip]
    end
    for jp in 1:n
        b[jp+oc] = β[jp]
    end
    # Balance the factors, then restore coverage lost to rounding.
    _balance_cover!(a, b, _grid_components(A, na, nb, m, n)...)
    for ip in 1:m
        lα[ip] = log(T(a[ip+or]))
    end
    for jp in 1:n
        lβ[jp] = log(T(b[jp+oc]))
    end
    t = zero(T)
    for jp in 1:n
        lj = lβ[jp]
        tj = zero(T)
        # Off-support entries produce `-Inf` or `NaN`, both ignored by `>`.
        for ip in 1:m
            u = (L[ip, jp] - lα[ip] - lj) / 2
            tj = ifelse(u > tj, u, tj)
        end
        t = ifelse(tj > t, tj, t)
    end
    # A supported row or column with zero scale gives `t = +Inf`.
    isfinite(t) ||
        throw(ArgumentError("inflate_feasible! requires a start with positive scale on every supported row/column"))
    iszero(t) && return a, b
    for ip in 1:m
        iszero(a[ip+or]) || (a[ip+or] = exp(lα[ip] + t))
    end
    for jp in 1:n
        iszero(b[jp+oc]) || (b[jp+oc] = exp(lβ[jp] + t))
    end
    return a, b
end
