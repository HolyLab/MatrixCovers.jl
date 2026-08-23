module MatrixCoversUnitfulExt

using LinearAlgebra: LinearAlgebra, Hermitian, Symmetric
using MatrixCovers
using MatrixCovers: AbsLog, AbsLinear
using SparseArrays: SparseMatrixCSC
using Unitful: Unitful, FreeUnits, Quantity, Unit, unit, ustrip

const MC = MatrixCovers

# Exponents of a unit, keyed by atomic unit. Rational, because a symmetric cover
# halves the diagonal's exponents.
const UnitExps = Dict{FreeUnits,Rational{Int}}

# Cover objectives are dimensionless and accumulate in the quantity's numeric type.
MC.scalar_type(::Type{<:Quantity{T}}) where {T} = MC.scalar_type(T)
const QMatrix = AbstractMatrix{<:Quantity}
const QVector = AbstractVector{<:Quantity}

# ============================================================
# Unit algebra
# ============================================================

# Decompose `FreeUnits{N,D,A}` through its documented type parameters. Gauge
# selection needs per-atom rational exponents, which unit arithmetic cannot expose.

# Atomic unit at the first power; prefixes remain distinct atoms.
atomic(x::Unit{N,D}) where {N,D} = FreeUnits{(Unit{N,D}(x.tens, 1//1),), D, nothing}()

# Reject affine units because multiplicative covers require an absolute zero.
function exps(u::FreeUnits{N,D,A}) where {N,D,A}
    A === nothing || throw(ArgumentError("""
    affine units are not supported: `$u` measures from a shifted origin, so no \
    product `a[i]*b[j]` covers an entry in it. Convert to a unit with a true zero \
    first (`u"K"` for `u"°C"`, `u"Ra"` for `u"°F"`)."""))
    d = UnitExps()
    for x in N
        a = atomic(x)
        p = get(d, a, 0//1) + x.power
        iszero(p) ? delete!(d, a) : (d[a] = p)
    end
    return d
end

# Context-dependent units are unsupported.
exps(u::Unitful.Units) = throw(ArgumentError(
    "unsupported unit type $(nameof(typeof(u))) for `$u`: MatrixCovers reads `FreeUnits`. " *
    "Convert with `uconvert(FreeUnits(u), x)`."))

exps(q::Quantity) = exps(unit(q))

# Remove zero exponents so equality compares units, not representations.
function combine(f, d1::UnitExps, d2::UnitExps)
    d = UnitExps()
    for k in union(keys(d1), keys(d2))
        p = f(get(d1, k, 0//1), get(d2, k, 0//1))
        iszero(p) || (d[k] = p)
    end
    return d
end
uadd(d1::UnitExps, d2::UnitExps) = combine(+, d1, d2)
usub(d1::UnitExps, d2::UnitExps) = combine(-, d1, d2)
uhalf(d::UnitExps) = UnitExps(k => v // 2 for (k, v) in d)

freeunits(d::UnitExps) = isempty(d) ? Unitful.NoUnits : prod(k^v for (k, v) in d)

# ============================================================
# Rank-1 unit factorization
# ============================================================

# Unit exponents must form a rank-1 additive matrix. A failing 2×2 minor
# identifies a mismatch.
function throw_nofactor(lhs, rhs, lhsname, rhsname)
    throw(DimensionMismatch("""
    units of `A` do not factor: $lhsname = $lhs, but $rhsname = $rhs.
    A cover requires `unit(A[i,j]) == unit(a[i])*unit(b[j])`, which forces these two \
    products to agree. Without this factorization, the terms in a row of `A*x` \
    can have incompatible units."""))
end

# A concrete element type gives every entry, including structural zeros, one unit.
uniform_unit(A::QMatrix) = isconcretetype(eltype(A)) ? exps(unit(eltype(A))) : nothing

# Factor row and column units relative to the first row and column.
function factor_units(A::QMatrix)
    ax1, ax2 = axes(A)
    uas = similar(Array{UnitExps}, ax1)
    ubs = similar(Array{UnitExps}, ax2)
    e = uniform_unit(A)
    if e === nothing
        i0, j0 = first(ax1), first(ax2)
        e00 = exps(A[i0, j0])
        for i in ax1
            uas[i] = usub(exps(A[i, j0]), e00)
        end
        for j in ax2
            ubs[j] = exps(A[i0, j])
        end
        for j in ax2, i in ax1
            uadd(uas[i], ubs[j]) == exps(A[i, j]) ||
                throw_nofactor(unit(A[i0, j0]) * unit(A[i, j]), unit(A[i0, j]) * unit(A[i, j0]),
                               "unit(A[$i0,$j0])*unit(A[$i,$j])", "unit(A[$i0,$j])*unit(A[$i,$j0])")
        end
    else
        fill!(uas, UnitExps())
        fill!(ubs, e)
    end
    c = gauge(uas, ubs)
    ua = similar(Array{FreeUnits}, ax1)
    ub = similar(Array{FreeUnits}, ax2)
    for i in ax1
        ua[i] = freeunits(uadd(uas[i], c))
    end
    for j in ax2
        ub[j] = freeunits(usub(ubs[j], c))
    end
    return ua, ub
end

# Diagonal entries fix symmetric scale units directly.
function factor_units_sym(A::QMatrix)
    ax = axes(A, 1)
    uas = similar(Array{UnitExps}, ax)
    e = uniform_unit(A)
    if e === nothing
        for i in ax
            uas[i] = uhalf(exps(A[i, i]))
        end
        for j in ax, i in ax
            uadd(uas[i], uas[j]) == exps(A[i, j]) ||
                throw_nofactor(unit(A[i, i]) * unit(A[j, j]), unit(A[i, j])^2,
                               "unit(A[$i,$i])*unit(A[$j,$j])", "unit(A[$i,$j])^2")
        end
    else
        fill!(uas, uhalf(e))
    end
    ua = similar(Array{FreeUnits}, ax)
    for i in ax
        ua[i] = freeunits(uas[i])
    end
    return ua
end

# Choose the unit gauge by minimizing total atomic-unit powers:
#
#     minimize_c  ∑_i ‖exps(ua[i]*c)‖₁ + ∑_j ‖exps(ub[j]/c)‖₁,
#
# Each atom reduces to a median interval; its midpoint makes `cover` agree with
# `symcover` on symmetric input.
function gauge(uas, ubs)
    atoms = Set{FreeUnits}()
    for d in uas
        union!(atoms, keys(d))
    end
    for d in ubs
        union!(atoms, keys(d))
    end
    c = UnitExps()
    pts = Rational{Int}[]
    for k in atoms
        # Only the multiset of exponents matters, so gather it index-free.
        empty!(pts)
        for d in uas
            push!(pts, -get(d, k, 0 // 1))
        end
        for d in ubs
            push!(pts, get(d, k, 0 // 1))
        end
        sort!(pts)
        n = length(pts)
        t = (pts[(n + 1) ÷ 2] + pts[n ÷ 2 + 1]) // 2
        iszero(t) || (c[k] = t)
    end
    return c
end

# ============================================================
# Strip and reattach
# ============================================================

# Strip values in their written units so the balance convention follows the
# caller's parametrization. Units must factor before stripping.
strip_matrix(A::QMatrix) = ustrip.(A)

# Refiners accept dimensionally equivalent start units and convert them.
strip_start(a::QVector, ua) = map(ustrip, ua, a)

reattach(a, ua) = a .* ua

# ============================================================
# Entry points
# ============================================================

sym(f, A::QMatrix, ϕ...; kwargs...) = reattach(f(ϕ..., strip_matrix(A); kwargs...), factor_units_sym(A))

function asym(f, A::QMatrix, ϕ...; kwargs...)
    ua, ub = factor_units(A)
    a, b = f(ϕ..., strip_matrix(A); kwargs...)
    return reattach(a, ua), reattach(b, ub)
end

# Allocating scratch from `A` permits uninitialized destination vectors.
function sym!(f, a::QVector, A::QMatrix, ϕ...; kwargs...)
    ua = factor_units_sym(A)
    An = strip_matrix(A)
    x = f(ϕ..., similar(a, float(real(eltype(An)))), An; kwargs...)
    a .= reattach(x, ua)
    return a
end

function asym!(f, a::QVector, b::QVector, A::QMatrix, ϕ...; kwargs...)
    ua, ub = factor_units(A)
    An = strip_matrix(A)
    T = float(real(eltype(An)))
    x, y = f(ϕ..., similar(a, T), similar(b, T), An; kwargs...)
    a .= reattach(x, ua)
    b .= reattach(y, ub)
    return a, b
end

# `a` is a start, and is read.
function symstart!(f, a::QVector, A::QMatrix, ϕ...; kwargs...)
    ua = factor_units_sym(A)
    x = f(ϕ..., strip_start(a, ua), strip_matrix(A); kwargs...)
    a .= reattach(x, ua)
    return a
end

function asymstart!(f, a::QVector, b::QVector, A::QMatrix, ϕ...; kwargs...)
    ua, ub = factor_units(A)
    x, y = f(ϕ..., strip_start(a, ua), strip_start(b, ub), strip_matrix(A); kwargs...)
    a .= reattach(x, ua)
    b .= reattach(y, ub)
    return a, b
end

# Mirror the core penalty dispatch while specializing on unitful matrices.
const PENALTIES = (:(AbsLog{1}), :(AbsLog{2}), :(AbsLinear{1}), :(AbsLinear{2}))

# Heuristic covers and initializers: `ϕ` is checked but not consulted.
MC.symcover(A::QMatrix; kwargs...) = sym(MC.symcover, A; kwargs...)
MC.symcover(ϕ::MC.AbstractCoverPenalty, A::QMatrix; kwargs...) = sym(MC.symcover, A, ϕ; kwargs...)
MC.symcover!(a::QVector, A::QMatrix; kwargs...) = sym!(MC.symcover!, a, A; kwargs...)
MC.symcover!(ϕ::MC.AbstractCoverPenalty, a::QVector, A::QMatrix; kwargs...) = sym!(MC.symcover!, a, A, ϕ; kwargs...)

MC.cover(A::QMatrix; kwargs...) = asym(MC.cover, A; kwargs...)
MC.cover(ϕ::MC.AbstractCoverPenalty, A::QMatrix; kwargs...) = asym(MC.cover, A, ϕ; kwargs...)
MC.cover!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asym!(MC.cover!, a, b, A; kwargs...)
MC.cover!(ϕ::MC.AbstractCoverPenalty, a::QVector, b::QVector, A::QMatrix; kwargs...) = asym!(MC.cover!, a, b, A, ϕ; kwargs...)

# Core `cover`/`cover!` methods dispatch on `Adjoint`/`Transpose` without an eltype
# bound, so a wrapped `Quantity` matrix needs these to stay unambiguous.
for W in (:(LinearAlgebra.Adjoint{<:Quantity}), :(LinearAlgebra.Transpose{<:Quantity}))
    @eval begin
        MC.cover(A::$W; kwargs...) = asym(MC.cover, A; kwargs...)
        MC.cover!(a::QVector, b::QVector, A::$W; kwargs...) = asym!(MC.cover!, a, b, A; kwargs...)
    end
end

MC.initialize_symcover(A::QMatrix; kwargs...) = sym(MC.initialize_symcover, A; kwargs...)
MC.initialize_symcover!(a::QVector, A::QMatrix; kwargs...) = sym!(MC.initialize_symcover!, a, A; kwargs...)
MC.initialize_cover(A::QMatrix; kwargs...) = asym(MC.initialize_cover, A; kwargs...)
MC.initialize_cover!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asym!(MC.initialize_cover!, a, b, A; kwargs...)

# Soft covers and the `*_min` family preserve core penalty dispatch.
MC.soft_symcover(A::QMatrix; kwargs...) = sym(MC.soft_symcover, A; kwargs...)
MC.soft_cover(A::QMatrix; kwargs...) = asym(MC.soft_cover, A; kwargs...)
MC.symcover_min(A::QMatrix; kwargs...) = sym(MC.symcover_min, A; kwargs...)
MC.symcover_min!(a::QVector, A::QMatrix; kwargs...) = symstart!(MC.symcover_min!, a, A; kwargs...)
MC.cover_min(A::QMatrix; kwargs...) = asym(MC.cover_min, A; kwargs...)
MC.cover_min!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(MC.cover_min!, a, b, A; kwargs...)
MC.soft_symcover_min(A::QMatrix; kwargs...) = sym(MC.soft_symcover_min, A; kwargs...)
MC.soft_symcover_min!(a::QVector, A::QMatrix; kwargs...) = symstart!(MC.soft_symcover_min!, a, A; kwargs...)
MC.soft_cover_min(A::QMatrix; kwargs...) = asym(MC.soft_cover_min, A; kwargs...)
MC.soft_cover_min!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(MC.soft_cover_min!, a, b, A; kwargs...)

# Disambiguate sparse unitful matrices without losing sparse storage.
const QSparse = SparseMatrixCSC{<:Quantity}
const QSparseSym = Union{QSparse,
                         Symmetric{<:Quantity,<:SparseMatrixCSC},
                         Hermitian{<:Quantity,<:SparseMatrixCSC}}

MC.symcover_min(ϕ::AbsLog{2}, A::QSparseSym; kwargs...) =
    sym(MC.symcover_min, A, ϕ; kwargs...)

MC.cover_min(ϕ::AbsLog{2}, A::QSparse; kwargs...) =
    asym(MC.cover_min, A, ϕ; kwargs...)

MC.soft_symcover_min(ϕ::AbsLog{2}, A::QSparseSym; kwargs...) =
    sym(MC.soft_symcover_min, A, ϕ; kwargs...)

MC.soft_cover_min(ϕ::AbsLog{2}, A::QSparse; kwargs...) =
    asym(MC.soft_cover_min, A, ϕ; kwargs...)

MC.symcover_min!(ϕ::AbsLog{2}, a::QVector, A::QSparseSym; kwargs...) =
    symstart!(MC.symcover_min!, a, A, ϕ; kwargs...)

MC.cover_min!(ϕ::AbsLog{2}, a::QVector, b::QVector, A::QSparse; kwargs...) =
    asymstart!(MC.cover_min!, a, b, A, ϕ; kwargs...)

for P in PENALTIES
    @eval begin
        MC.soft_symcover(ϕ::$P, A::QMatrix; kwargs...) = sym(MC.soft_symcover, A, ϕ; kwargs...)
        MC.soft_cover(ϕ::$P, A::QMatrix; kwargs...) = asym(MC.soft_cover, A, ϕ; kwargs...)

        MC.symcover_min(ϕ::$P, A::QMatrix; kwargs...) = sym(MC.symcover_min, A, ϕ; kwargs...)
        MC.symcover_min!(ϕ::$P, a::QVector, A::QMatrix; kwargs...) = symstart!(MC.symcover_min!, a, A, ϕ; kwargs...)
        MC.cover_min(ϕ::$P, A::QMatrix; kwargs...) = asym(MC.cover_min, A, ϕ; kwargs...)
        MC.cover_min!(ϕ::$P, a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(MC.cover_min!, a, b, A, ϕ; kwargs...)

        MC.soft_symcover_min(ϕ::$P, A::QMatrix; kwargs...) = sym(MC.soft_symcover_min, A, ϕ; kwargs...)
        MC.soft_symcover_min!(ϕ::$P, a::QVector, A::QMatrix; kwargs...) = symstart!(MC.soft_symcover_min!, a, A, ϕ; kwargs...)
        MC.soft_cover_min(ϕ::$P, A::QMatrix; kwargs...) = asym(MC.soft_cover_min, A, ϕ; kwargs...)
        MC.soft_cover_min!(ϕ::$P, a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(MC.soft_cover_min!, a, b, A, ϕ; kwargs...)
    end
end

end  # module MatrixCoversUnitfulExt
