# Cover methods for structured LinearAlgebra matrix types.
#
# Diagonal is handled individually.
# SymTridiagonal, Bidiagonal, and Tridiagonal share a single implementation
# via the PlusMinus1Banded union: all are square with entries only on the main
# diagonal and the ±1 off-diagonals.  Structural zeros (e.g. the sub-diagonal
# of an upper Bidiagonal) are returned as exact zero by getindex, so they are
# handled correctly by the generic algorithms.

const PlusMinus1Banded = Union{SymTridiagonal, Bidiagonal, Tridiagonal}

# ============================================================
# Diagonal
# ============================================================

function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, D::Diagonal; iter::Int=3) where T
    for i in eachindex(a, b, D.diag)
        Dii = T(abs(D.diag[i]))
        iszero(Dii) && continue
        aprod = a[i] * b[i]
        iszero(aprod) && continue
        s = sqrt(aprod / Dii)
        a[i] /= s
        b[i] /= s
    end
    return a, b
end

function _symcover_diagonal(D::Diagonal)
    T = float(eltype(D))
    a = zeros(T, length(D.diag))
    for i in eachindex(a, D.diag)
        a[i] = sqrt(T(abs(D.diag[i])))
    end
    return a
end
# Explicit ϕ-dispatch methods to resolve ambiguity with covers.jl
symcover(ϕ::AbsLog,    D::Diagonal; kwargs...) = _symcover_diagonal(D)
symcover(ϕ::AbsLinear, D::Diagonal; kwargs...) = _symcover_diagonal(D)
symcover(              D::Diagonal; kwargs...) = _symcover_diagonal(D)

function cover(ϕ, D::Diagonal; kwargs...)
    a = symcover(ϕ, D)
    return tighten_cover!(a, copy(a), D)
end
cover(D::Diagonal; kwargs...) = cover(AbsLinear{2}(), D; kwargs...)

# ============================================================
# PlusMinus1Banded  (SymTridiagonal, Bidiagonal, Tridiagonal)
# ============================================================

# Symmetric tighten: both A[i,i+1] and A[i+1,i] are checked independently.
function tighten_cover!(a::AbstractVector{T}, A::PlusMinus1Banded; iter::Int=3) where T
    n = size(A, 1)
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        for i in 1:n
            Aii = T(abs(A[i, i]))
            iszero(Aii) && continue
            aratio[i] = min(aratio[i], a[i]^2 / Aii)
        end
        for i in 1:n-1
            Aij = T(abs(A[i, i+1]))
            if !iszero(Aij)
                r = a[i] * a[i+1] / Aij
                aratio[i]   = min(aratio[i],   r)
                aratio[i+1] = min(aratio[i+1], r)
            end
            Aij = T(abs(A[i+1, i]))
            if !iszero(Aij)
                r = a[i] * a[i+1] / Aij
                aratio[i]   = min(aratio[i],   r)
                aratio[i+1] = min(aratio[i+1], r)
            end
        end
        a ./= sqrt.(aratio)
    end
    return a
end

# Asymmetric tighten for all ±1-banded types.
function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::PlusMinus1Banded; iter::Int=3) where T
    n = size(A, 1)
    aratio = similar(a)
    bratio = similar(b)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        fill!(bratio, typemax(T))
        for i in 1:n
            Aii = T(abs(A[i, i]))
            iszero(Aii) && continue
            r = a[i] * b[i] / Aii
            aratio[i] = min(aratio[i], r)
            bratio[i] = min(bratio[i], r)
        end
        for i in 1:n-1
            Aij = T(abs(A[i, i+1]))
            if !iszero(Aij)
                r = a[i] * b[i+1] / Aij
                aratio[i]   = min(aratio[i],   r)
                bratio[i+1] = min(bratio[i+1], r)
            end
            Aij = T(abs(A[i+1, i]))
            if !iszero(Aij)
                r = a[i+1] * b[i] / Aij
                aratio[i+1] = min(aratio[i+1], r)
                bratio[i]   = min(bratio[i],   r)
            end
        end
        a ./= sqrt.(aratio)
        b ./= sqrt.(bratio)
    end
    return a, b
end

function _symcover_pm1banded(A::PlusMinus1Banded; iter::Int=3)
    n = size(A, 1)
    T = float(eltype(A))
    a = zeros(T, n)
    # Quadratic (AbsLog{2}) initialization: log-norm over ±1 diagonals
    loga = zeros(T, n)
    nza  = zeros(Int, n)
    for i in 1:n
        Aii = abs(A[i, i])
        iszero(Aii) && continue
        loga[i] += log(Aii)
        nza[i]  += 1
    end
    for i in 1:n-1
        Aij = abs(A[i, i+1])   # upper triangle; caller asserts A[i+1,i] matches
        iszero(Aij) && continue
        lAij = log(Aij)
        loga[i]   += lAij;  nza[i]   += 1
        loga[i+1] += lAij;  nza[i+1] += 1
    end
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in 1:n
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
    end
    # Clamp diagonal
    for i in 1:n
        Aii = T(abs(A[i, i]))
        a[i]^2 < Aii && (a[i] = sqrt(Aii))
    end
    # Boost off-diagonal entries (upper triangle)
    for i in 1:n-1
        Aij = T(abs(A[i, i+1]))
        iszero(Aij) && continue
        ai, aj = a[i], a[i+1]
        if iszero(aj)
            iszero(ai) ? (a[i] = a[i+1] = sqrt(Aij)) : (a[i+1] = Aij / ai)
        elseif iszero(ai)
            a[i] = Aij / aj
        else
            aprod = ai * aj
            if aprod < Aij
                s = sqrt(Aij / aprod)
                a[i] *= s;  a[i+1] *= s
            end
        end
    end
    return tighten_cover!(a, A; iter)
end
# Explicit ϕ-dispatch to resolve ambiguity with covers.jl
symcover(ϕ::AbsLog,    A::PlusMinus1Banded; iter::Int=3) = _symcover_pm1banded(A; iter)
symcover(ϕ::AbsLinear, A::PlusMinus1Banded; iter::Int=3) = _symcover_pm1banded(A; iter)
symcover(              A::PlusMinus1Banded; iter::Int=3) = _symcover_pm1banded(A; iter)

function cover(ϕ, A::PlusMinus1Banded; kwargs...)
    T = float(eltype(A))
    n = size(A, 1)
    a = zeros(T, n)
    b = zeros(T, n)
    loga = zeros(T, n)
    logb = zeros(T, n)
    nza  = zeros(Int, n)
    nzb  = zeros(Int, n)
    logmu   = zero(T)
    nztotal = 0
    for i in 1:n
        Aii = abs(A[i, i])
        iszero(Aii) && continue
        lAii = log(T(Aii))
        loga[i] += lAii;  logb[i] += lAii
        nza[i]  += 1;     nzb[i]  += 1
        logmu   += lAii;  nztotal += 1
    end
    for i in 1:n-1
        Aij = abs(A[i, i+1])
        if !iszero(Aij)
            lAij = log(T(Aij))
            loga[i]   += lAij;  logb[i+1] += lAij
            nza[i]    += 1;     nzb[i+1]  += 1
            logmu     += lAij;  nztotal   += 1
        end
        Aij = abs(A[i+1, i])
        if !iszero(Aij)
            lAij = log(T(Aij))
            loga[i+1] += lAij;  logb[i] += lAij
            nza[i+1]  += 1;     nzb[i]  += 1
            logmu     += lAij;  nztotal += 1
        end
    end
    halfmu = iszero(nztotal) ? zero(T) : T(logmu / (2 * nztotal))
    for i in 1:n
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
        b[i] = iszero(nzb[i]) ? zero(T) : exp(logb[i] / nzb[i] - halfmu)
    end
    # Boost diagonal
    for i in 1:n
        Aii = T(abs(A[i, i]))
        aprod = a[i] * b[i]
        aprod >= Aii && continue
        s = sqrt(Aii / aprod)
        a[i] *= s;  b[i] *= s
    end
    # Boost off-diagonal
    for i in 1:n-1
        Aij = T(abs(A[i, i+1]))
        if !iszero(Aij)
            aprod = a[i] * b[i+1]
            if aprod < Aij
                s = sqrt(Aij / aprod)
                a[i] *= s;  b[i+1] *= s
            end
        end
        Aij = T(abs(A[i+1, i]))
        if !iszero(Aij)
            aprod = a[i+1] * b[i]
            if aprod < Aij
                s = sqrt(Aij / aprod)
                a[i+1] *= s;  b[i] *= s
            end
        end
    end
    return tighten_cover!(a, b, A; kwargs...)
end
cover(A::PlusMinus1Banded; kwargs...) = cover(AbsLinear{2}(), A; kwargs...)
