module SIAUnitful

using LinearAlgebra
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLog, AbsLinear
using Unitful
using Unitful: FreeUnits, Quantity, Unit

const SIA = ScaleInvariantAnalysis

# Exponents of a unit, keyed by atomic unit. Rational, because a symmetric cover
# halves the diagonal's exponents.
const UnitExps = Dict{FreeUnits,Rational{Int}}
const QMatrix = AbstractMatrix{<:Quantity}
const QVector = AbstractVector{<:Quantity}

# ============================================================
# Unit algebra
# ============================================================

# An atomic unit at the first power. A prefix belongs to the atom -- `mm` and `m`
# are distinct -- so a coordinate named in `mm` keeps `mm` in its cover.
atomic(x::Unit{N,D}) where {N,D} = FreeUnits{(Unit{N,D}(x.tens, 1//1),), D, nothing}()

function exps(u::FreeUnits)
    d = UnitExps()
    for x in typeof(u).parameters[1]
        a = atomic(x)
        p = get(d, a, 0//1) + x.power
        iszero(p) ? delete!(d, a) : (d[a] = p)
    end
    return d
end
exps(q::Quantity) = exps(unit(q))

# Zero exponents are pruned throughout so that `==` on a `UnitExps` compares
# units rather than representations.
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

# A cover needs `unit(A[i,j]) == unit(a[i])*unit(b[j])`: the unit exponents form a
# rank-1 additive matrix, and any violation is witnessed by a 2x2 minor. The message
# quotes that minor, so it names only entries the caller wrote.
function throw_nofactor(lhs, rhs, lhsname, rhsname)
    throw(DimensionMismatch("""
    units of `A` do not factor: $lhsname = $lhs, but $rhsname = $rhs.
    A cover requires `unit(A[i,j]) == unit(a[i])*unit(b[j])`, which forces these two \
    products to agree. Any matrix that models the physical world satisfies this: \
    without it the terms of a row of `A*x` do not share units, so `A*x` is undefined \
    for every `x`."""))
end

# A concrete element type names one unit for every entry, structural zeros
# included, so there is nothing to verify and nothing to read: `A` contributes only
# `unit(eltype(A))`. This is the only shape a sparse `A` can take, since sparse
# storage synthesizes its structural zeros with `zero(eltype(A))`.
uniform_unit(A::QMatrix) = isconcretetype(eltype(A)) ? exps(unit(eltype(A))) : nothing

# `unit(a[i])` and `unit(b[j])` up to the gauge `a -> a*c`, `b -> b/c`, taken
# relative to the first row and column.
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

# `a[i]*a[i] == A[i,i]` pins `unit(a[i])` outright: the symmetric gauge `a -> a*c`
# would scale every product by `c^2`, so only `c = 1` preserves them.
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

# The gauge `a -> a*c`, `b -> b/c` leaves every product `a[i]*b[j]` unchanged, so
# the factorization fixes the units only up to `c`. Pin it by minimizing the total
# atomic-unit powers carried by the two scale vectors,
#
#     minimize_c  ∑_i ‖exps(ua[i]*c)‖₁ + ∑_j ‖exps(ub[j]/c)‖₁,
#
# which separates over atoms into subproblems ∑_i |t + αᵢ| + ∑_j |t - βⱼ|, each
# minimized on the median interval of {-αᵢ} ∪ {βⱼ}.
#
# That interval is a single point only when the atom's exponents pin it; otherwise
# the objective is flat across it and the choice within it is the whole content of
# the convention. Take its midpoint, which makes `cover` reproduce `symcover` on
# symmetric input. There, `unit(A[i,j])` has exponents `dᵢ + dⱼ`, so `αᵢ = dᵢ - d₀`
# and `βⱼ = d₀ + dⱼ` relative to the reference row `i0`, and the points
# `{d₀ - dᵢ} ∪ {d₀ + dⱼ}` are distributed symmetrically about `d₀`. The midpoint is
# therefore `d₀` exactly, which is the shift that returns `unit(a[i])` with
# exponents `dᵢ` -- what `a[i]*a[i] == A[i,i]` demands.
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

# `A` is stripped in the units the caller wrote, not in a canonical system. The
# cover itself is scale-invariant, but the balance convention that splits `a` from
# `b` is not, so the strip scale selects the parametrization; the caller's units
# are their statement of the scale they want it pinned to. Stripping requires the
# units to factor as written -- otherwise entries on incommensurate scales
# (`1.0mm^-2` and `1.0m^-2` both strip to `1.0`) would be covered as if comparable.
strip_matrix(A::QMatrix) = ustrip.(A)

# The `*_min!` family reads `a` as a start, so its units must be the cover's. Any
# dimensionally equivalent spelling is accepted and converted; `ustrip` raises on
# a start that is not.
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

# `a` is overwritten, so neither its values nor its units are read: the scratch it
# is stripped into takes its element type from `A`, matching what the unitless
# methods allocate. An `a` of undefined references is a valid destination.
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

# Every penalty slot below mirrors ScaleInvariantAnalysis's own method table: where
# it leaves `ϕ` untyped these do too, and where it dispatches on concrete penalties
# these enumerate the same ones. Each method is then strictly more specific than the
# one it shadows -- including those in the JuMP and Ipopt extensions, which leave the
# matrix slot untyped -- so no ambiguity arises. A penalty the package does not
# support raises a `MethodError` here exactly as it does on a unitless matrix.
const PENALTIES = (:(AbsLog{1}), :(AbsLog{2}), :(AbsLinear{1}), :(AbsLinear{2}))

# Heuristic covers and initializers: `ϕ` is untyped upstream and ignored.
SIA.symcover(A::QMatrix; kwargs...) = sym(SIA.symcover, A; kwargs...)
SIA.symcover(ϕ, A::QMatrix; kwargs...) = sym(SIA.symcover, A, ϕ; kwargs...)
SIA.symcover!(a::QVector, A::QMatrix; kwargs...) = sym!(SIA.symcover!, a, A; kwargs...)
SIA.symcover!(ϕ, a::QVector, A::QMatrix; kwargs...) = sym!(SIA.symcover!, a, A, ϕ; kwargs...)

SIA.cover(A::QMatrix; kwargs...) = asym(SIA.cover, A; kwargs...)
SIA.cover(ϕ, A::QMatrix; kwargs...) = asym(SIA.cover, A, ϕ; kwargs...)
SIA.cover!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asym!(SIA.cover!, a, b, A; kwargs...)
SIA.cover!(ϕ, a::QVector, b::QVector, A::QMatrix; kwargs...) = asym!(SIA.cover!, a, b, A, ϕ; kwargs...)

# `cover`/`cover!` dispatch on `Adjoint`/`Transpose` upstream without an eltype
# bound, so a wrapped `Quantity` matrix needs these to stay unambiguous.
for W in (:(LinearAlgebra.Adjoint{<:Quantity}), :(LinearAlgebra.Transpose{<:Quantity}))
    @eval begin
        SIA.cover(A::$W; kwargs...) = asym(SIA.cover, A; kwargs...)
        SIA.cover!(a::QVector, b::QVector, A::$W; kwargs...) = asym!(SIA.cover!, a, b, A; kwargs...)
    end
end

SIA.initialize_symcover(A::QMatrix; kwargs...) = sym(SIA.initialize_symcover, A; kwargs...)
SIA.initialize_symcover!(a::QVector, A::QMatrix; kwargs...) = sym!(SIA.initialize_symcover!, a, A; kwargs...)
SIA.initialize_cover(A::QMatrix; kwargs...) = asym(SIA.initialize_cover, A; kwargs...)
SIA.initialize_cover!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asym!(SIA.initialize_cover!, a, b, A; kwargs...)

# Soft covers and the `*_min` family: `ϕ` is dispatched on upstream.
SIA.soft_symcover(A::QMatrix; kwargs...) = sym(SIA.soft_symcover, A; kwargs...)
SIA.soft_cover(A::QMatrix; kwargs...) = asym(SIA.soft_cover, A; kwargs...)
SIA.symcover_min(A::QMatrix; kwargs...) = sym(SIA.symcover_min, A; kwargs...)
SIA.symcover_min!(a::QVector, A::QMatrix; kwargs...) = symstart!(SIA.symcover_min!, a, A; kwargs...)
SIA.cover_min(A::QMatrix; kwargs...) = asym(SIA.cover_min, A; kwargs...)
SIA.cover_min!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(SIA.cover_min!, a, b, A; kwargs...)
SIA.soft_symcover_min(A::QMatrix; kwargs...) = sym(SIA.soft_symcover_min, A; kwargs...)
SIA.soft_symcover_min!(a::QVector, A::QMatrix; kwargs...) = symstart!(SIA.soft_symcover_min!, a, A; kwargs...)
SIA.soft_cover_min(A::QMatrix; kwargs...) = asym(SIA.soft_cover_min, A; kwargs...)
SIA.soft_cover_min!(a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(SIA.soft_cover_min!, a, b, A; kwargs...)

for P in PENALTIES
    @eval begin
        SIA.soft_symcover(ϕ::$P, A::QMatrix; kwargs...) = sym(SIA.soft_symcover, A, ϕ; kwargs...)
        SIA.soft_cover(ϕ::$P, A::QMatrix; kwargs...) = asym(SIA.soft_cover, A, ϕ; kwargs...)

        SIA.symcover_min(ϕ::$P, A::QMatrix; kwargs...) = sym(SIA.symcover_min, A, ϕ; kwargs...)
        SIA.symcover_min!(ϕ::$P, a::QVector, A::QMatrix; kwargs...) = symstart!(SIA.symcover_min!, a, A, ϕ; kwargs...)
        SIA.cover_min(ϕ::$P, A::QMatrix; kwargs...) = asym(SIA.cover_min, A, ϕ; kwargs...)
        SIA.cover_min!(ϕ::$P, a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(SIA.cover_min!, a, b, A, ϕ; kwargs...)

        SIA.soft_symcover_min(ϕ::$P, A::QMatrix; kwargs...) = sym(SIA.soft_symcover_min, A, ϕ; kwargs...)
        SIA.soft_symcover_min!(ϕ::$P, a::QVector, A::QMatrix; kwargs...) = symstart!(SIA.soft_symcover_min!, a, A, ϕ; kwargs...)
        SIA.soft_cover_min(ϕ::$P, A::QMatrix; kwargs...) = asym(SIA.soft_cover_min, A, ϕ; kwargs...)
        SIA.soft_cover_min!(ϕ::$P, a::QVector, b::QVector, A::QMatrix; kwargs...) = asymstart!(SIA.soft_cover_min!, a, b, A, ϕ; kwargs...)
    end
end

end  # module SIAUnitful
