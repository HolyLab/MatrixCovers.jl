module SIASparseArrays

using LinearAlgebra
using SparseArrays
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLog, AbsLinear, unconstrained_min!, _abslog2_greatest_curvature_eigvec, _abslinear2_linesearch, _abslinear2_iter!, _abslinear1_iter!, _symcover_min_abslog2, _cover_min_abslog2

# ============================================================
# Private helpers (operate on a plain SparseMatrixCSC parent)
# ============================================================

# Quadratic (AbsLog{2}) initialisation of the symmetric cover.
# `uplo` controls which triangle is treated as canonical ('U' → i ≤ j, 'L' → i ≥ j).
function _sparse_symcover_init_quadratic!(a::AbstractVector{T}, P::SparseMatrixCSC, uplo::Char) where T
    rv = rowvals(P)
    nzs = nonzeros(P)
    loga = zeros(T, size(P, 1))
    nza  = zeros(Int, size(P, 1))
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            (uplo == 'U' ? i > j : i < j) && continue   # canonical triangle only
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            lAij = log(Aij)
            loga[i] += lAij
            nza[i]  += 1
            if i != j
                loga[j] += lAij
                nza[j]  += 1
            end
        end
    end
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in eachindex(a)
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
    end
    # Clamp diagonal
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            i != j && continue
            Aii = abs(nzs[k])
            if a[i]^2 < Aii
                a[i] = sqrt(Aii)
            end
            break
        end
    end
    return a, nza
end

# Boost: ensure a[i]*a[j] >= |A[i,j]| for each stored off-diagonal entry in the canonical triangle.
function _sparse_symcover_boost!(a::AbstractVector, P::SparseMatrixCSC, uplo::Char)
    rv = rowvals(P)
    nzs = nonzeros(P)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            i == j && continue
            (uplo == 'U' ? i > j : i < j) && continue
            Aij = abs(nzs[k])
            ai, aj = a[i], a[j]
            if iszero(aj)
                !iszero(ai) ? (a[j] = Aij / ai) : (a[i] = a[j] = sqrt(Aij))
            elseif iszero(ai)
                a[i] = Aij / aj
            else
                aprod = ai * aj
                aprod >= Aij && continue
                s = sqrt(Aij / aprod)
                a[i] = s * ai
                a[j] = s * aj
            end
        end
    end
end

# Tighten the symmetric cover, iterating only stored nonzeros.
function _tighten_cover_sym_sparse!(a::AbstractVector{T}, P::SparseMatrixCSC; iter::Int) where T
    rv = rowvals(P)
    nzs = nonzeros(P)
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        for j in axes(P, 2)
            aratioj = aratio[j]
            aj = a[j]
            for k in nzrange(P, j)
                i = rv[k]
                Aij = T(abs(nzs[k]))
                r = ifelse(iszero(Aij), typemax(T), a[i] * aj / Aij)
                aratio[i] = min(aratio[i], r)
                aratioj   = min(aratioj, r)
            end
            aratio[j] = aratioj
        end
        a ./= sqrt.(aratio)
    end
    return a
end

# Tighten the asymmetric cover, iterating only stored nonzeros.
function _tighten_cover_asym_sparse!(a::AbstractVector{T}, b::AbstractVector{T}, P::SparseMatrixCSC; iter::Int) where T
    rv = rowvals(P)
    nzs = nonzeros(P)
    aratio = fill(typemax(T), eachindex(a))
    bratio = fill(typemax(T), eachindex(b))
    for _ in 1:iter
        fill!(aratio, typemax(T))
        fill!(bratio, typemax(T))
        for j in eachindex(b)
            bratioj = bratio[j]
            bj = b[j]
            for k in nzrange(P, j)
                i = rv[k]
                Aij = T(abs(nzs[k]))
                r = ifelse(iszero(Aij), typemax(T), a[i] * bj / Aij)
                aratio[i] = min(aratio[i], r)
                bratioj   = min(bratioj, r)
            end
            bratio[j] = bratioj
        end
        a ./= sqrt.(aratio)
        b ./= sqrt.(bratio)
    end
    return a, b
end

# ============================================================
# SparseMatrixCSC methods
# ============================================================

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, A::SparseMatrixCSC; iter::Int=3) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    return _tighten_cover_sym_sparse!(a, A; iter)
end

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::SparseMatrixCSC; iter::Int=3) where T
    eachindex(a) == axes(A, 1) || throw(DimensionMismatch("indices of a must match row-indexing of A"))
    eachindex(b) == axes(A, 2) || throw(DimensionMismatch("indices of b must match column-indexing of A"))
    return _tighten_cover_asym_sparse!(a, b, A; iter)
end

function ScaleInvariantAnalysis.symcover(ϕ::AbsLog, A::SparseMatrixCSC; iter::Int=3)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(A))
    a = zeros(T, size(A, 1))
    _sparse_symcover_init_quadratic!(a, A, 'U')
    _sparse_symcover_boost!(a, A, 'U')
    return _tighten_cover_sym_sparse!(a, A; iter)
end

function ScaleInvariantAnalysis.symcover(ϕ::AbsLinear, A::SparseMatrixCSC; iter::Int=3)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(A))
    a = zeros(T, size(A, 1))
    # Use the sparse quadratic init and extract nza for eigenvector
    _, nza = _sparse_symcover_init_quadratic!(a, A, 'U')
    v  = _abslog2_greatest_curvature_eigvec(nza)
    α₀ = log.(max.(a, eps(T)))
    # Find t_feas: smallest t ≥ 0 making a₀.*exp(t*v) feasible
    rv  = rowvals(A)
    nzs = nonzeros(A)
    t_feas = zero(T)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            i = rv[k]
            Aij = T(abs(nzs[k]))
            iszero(Aij) && continue
            s = v[i] + v[j]
            iszero(s) && continue
            deficit = log(Aij) - α₀[i] - α₀[j]
            deficit > 0 || continue
            t_feas = max(t_feas, deficit / s)
        end
    end
    for i in eachindex(a)
        a[i] = a[i] * exp(T(t_feas) * T(v[i]))
    end
    return _tighten_cover_sym_sparse!(a, A; iter)
end

ScaleInvariantAnalysis.symcover(A::SparseMatrixCSC; kwargs...) = ScaleInvariantAnalysis.symcover(AbsLinear{2}(), A; kwargs...)

function ScaleInvariantAnalysis.cover(ϕ, A::SparseMatrixCSC; iter::Int=3)
    T = float(eltype(A))
    a = zeros(T, size(A, 1))
    b = zeros(T, size(A, 2))
    loga = zeros(T, size(A, 1))
    logb = zeros(T, size(A, 2))
    nza  = zeros(Int, size(A, 1))
    nzb  = zeros(Int, size(A, 2))
    logmu   = zero(T)
    nztotal = 0
    rv  = rowvals(A)
    nzs = nonzeros(A)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            i = rv[k]
            lAij = log(Aij)
            loga[i] += lAij
            logb[j] += lAij
            nza[i]  += 1
            nzb[j]  += 1
            logmu   += lAij
            nztotal += 1
        end
    end
    halfmu = iszero(nztotal) ? zero(T) : T(logmu / (2 * nztotal))
    for i in axes(A, 1)
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
    end
    for j in axes(A, 2)
        b[j] = iszero(nzb[j]) ? zero(T) : exp(logb[j] / nzb[j] - halfmu)
    end
    for j in axes(A, 2)
        bj = b[j]
        for k in nzrange(A, j)
            i = rv[k]
            Aij, ai = abs(nzs[k]), a[i]
            aprod = ai * bj
            aprod >= Aij && continue
            s = sqrt(Aij / aprod)
            a[i] = s * ai
            b[j] = bj = s * bj
        end
    end
    return _tighten_cover_asym_sparse!(a, b, A; iter)
end
ScaleInvariantAnalysis.cover(A::SparseMatrixCSC; kwargs...) = ScaleInvariantAnalysis.cover(AbsLinear{2}(), A; kwargs...)

# Native AbsLog{2} MCM solvers on sparse supports default to the matrix-free LSQR
# inner solve, whose per-iteration cost is O(nnz) and whose accuracy tracks the
# conditioning of √W·R (≈ √κ) rather than that of the normal equations (≈ κ). This
# is the intended path when nnz ≪ n²; pass `linsolve=:auto`/`:dense` to force the
# dense factorization. Only AbsLog{2} is native; other penalties dispatch to the
# JuMP extension.
# The worker allocates its scale vectors with `similar(A, ...)`, which is a
# `SparseVector` for a sparse `A`; the scales are dense objects, so return plain
# `Vector`s, matching `cover`/`symcover` on the same input.
function ScaleInvariantAnalysis.symcover_min(ϕ::AbsLog{2}, A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(A; linsolve, kwargs...)
    return Vector(a)
end

function ScaleInvariantAnalysis.cover_min(ϕ::AbsLog{2}, A::SparseMatrixCSC; linsolve::Symbol=:lsqr, kwargs...)
    a, b, _ = _cover_min_abslog2(A; linsolve, kwargs...)
    return Vector(a), Vector(b)
end

function ScaleInvariantAnalysis.symcover_min(ϕ::AbsLog{2}, S::Symmetric{<:Any, <:SparseMatrixCSC}; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(S; linsolve, kwargs...)
    return Vector(a)
end

function ScaleInvariantAnalysis.symcover_min(ϕ::AbsLog{2}, H::Hermitian{<:Real, <:SparseMatrixCSC}; linsolve::Symbol=:lsqr, kwargs...)
    a, _ = _symcover_min_abslog2(H; linsolve, kwargs...)
    return Vector(a)
end

# ============================================================
# Symmetric{<:Any, <:SparseMatrixCSC} methods
# ============================================================

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, S::Symmetric{<:Any, <:SparseMatrixCSC}; iter::Int=3) where T
    P = parent(S)
    ax = axes(P, 1)
    axes(P, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    return _tighten_cover_sym_sparse!(a, P; iter)
end

function ScaleInvariantAnalysis.symcover(ϕ::AbsLog, S::Symmetric{<:Any, <:SparseMatrixCSC}; iter::Int=3)
    P = parent(S)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(P))
    a = zeros(T, size(P, 1))
    _sparse_symcover_init_quadratic!(a, P, S.uplo)
    _sparse_symcover_boost!(a, P, S.uplo)
    return _tighten_cover_sym_sparse!(a, P; iter)
end

function ScaleInvariantAnalysis.symcover(ϕ::AbsLinear, S::Symmetric{<:Any, <:SparseMatrixCSC}; iter::Int=3)
    P = parent(S)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(P))
    a = zeros(T, size(P, 1))
    _, nza = _sparse_symcover_init_quadratic!(a, P, S.uplo)
    v  = _abslog2_greatest_curvature_eigvec(nza)
    α₀ = log.(max.(a, eps(T)))
    rv  = rowvals(P)
    nzs = nonzeros(P)
    t_feas = zero(T)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            (S.uplo == 'U' ? i > j : i < j) && continue
            Aij = T(abs(nzs[k]))
            iszero(Aij) && continue
            s = v[i] + v[j]
            iszero(s) && continue
            # Count both (i,j) and (j,i) for off-diagonal
            deficit = log(Aij) - α₀[i] - α₀[j]
            deficit > 0 || continue
            t_feas = max(t_feas, deficit / s)
        end
    end
    for i in eachindex(a)
        a[i] = a[i] * exp(T(t_feas) * T(v[i]))
    end
    return _tighten_cover_sym_sparse!(a, P; iter)
end

ScaleInvariantAnalysis.symcover(S::Symmetric{<:Any, <:SparseMatrixCSC}; kwargs...) =
    ScaleInvariantAnalysis.symcover(AbsLinear{2}(), S; kwargs...)

# ============================================================
# Hermitian{<:Real, <:SparseMatrixCSC} methods
# (Real-valued Hermitian is equivalent to Symmetric)
# ============================================================

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, H::Hermitian{<:Real, <:SparseMatrixCSC}; iter::Int=3) where T
    P = parent(H)
    ax = axes(P, 1)
    axes(P, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    return _tighten_cover_sym_sparse!(a, P; iter)
end

function ScaleInvariantAnalysis.symcover(ϕ::AbsLog, H::Hermitian{<:Real, <:SparseMatrixCSC}; iter::Int=3)
    P = parent(H)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(P))
    a = zeros(T, size(P, 1))
    _sparse_symcover_init_quadratic!(a, P, H.uplo)
    _sparse_symcover_boost!(a, P, H.uplo)
    return _tighten_cover_sym_sparse!(a, P; iter)
end

function ScaleInvariantAnalysis.symcover(ϕ::AbsLinear, H::Hermitian{<:Real, <:SparseMatrixCSC}; iter::Int=3)
    P = parent(H)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(P))
    a = zeros(T, size(P, 1))
    _, nza = _sparse_symcover_init_quadratic!(a, P, H.uplo)
    v  = _abslog2_greatest_curvature_eigvec(nza)
    α₀ = log.(max.(a, eps(T)))
    rv  = rowvals(P)
    nzs = nonzeros(P)
    t_feas = zero(T)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            (H.uplo == 'U' ? i > j : i < j) && continue
            Aij = T(abs(nzs[k]))
            iszero(Aij) && continue
            s = v[i] + v[j]
            iszero(s) && continue
            deficit = log(Aij) - α₀[i] - α₀[j]
            deficit > 0 || continue
            t_feas = max(t_feas, deficit / s)
        end
    end
    for i in eachindex(a)
        a[i] = a[i] * exp(T(t_feas) * T(v[i]))
    end
    return _tighten_cover_sym_sparse!(a, P; iter)
end

ScaleInvariantAnalysis.symcover(H::Hermitian{<:Real, <:SparseMatrixCSC}; kwargs...) =
    ScaleInvariantAnalysis.symcover(AbsLinear{2}(), H; kwargs...)

end  # module SIASparseArrays
