# Workarounds for `juliac --trim=safe`, which compiles only calls it can
# resolve statically. A few `MatrixCovers` calls are not resolvable in their
# natural form from an `@api` entrypoint; each function here reshapes one of
# them — allocating an output so a mutating method can be called, resolving an
# optional keyword to a concrete sentinel, or isolating a branch behind
# `@noinline` — and carries the constraint that forces the shape. The
# entrypoints themselves are in `matrixcovers.jl`.

# --- Hard covers ------------------------------------------------------------

# `MatrixCovers.symcover` allocates its own output and forwards `maxiter`
# through two layers of `kwargs...` before reaching a concretely-typed callee
# (`tighten_cover!`); `--trim=safe` cannot statically resolve that chain. The
# mutating `symcover!` removes one layer, which is enough to make the call
# resolvable, so the output is allocated here and that method called instead.
function _symcover(A::Matrix{Float64}, maxiter::Union{Int64, Nothing})
    a = Vector{Float64}(undef, size(A, 1))
    return isnothing(maxiter) ? MatrixCovers.symcover!(a, A) : MatrixCovers.symcover!(a, A; maxiter = Int(maxiter))
end

# `cover!`'s own `tighten_cover!(a, b, A; kwargs...)` call is not in tail
# position (it balances and inflates the result afterward), and `--trim=safe`
# cannot resolve a `kwargs...`-forwarded call there, even through the mutating
# form. `cover!`'s five-step body is reproduced here instead, with an explicit
# (non-forwarded) `maxiter` on the one step that takes it.
function _cover_ab(A::Matrix{Float64}, maxiter::Union{Int64, Nothing})
    a = Vector{Float64}(undef, size(A, 1))
    b = Vector{Float64}(undef, size(A, 2))
    MatrixCovers.unconstrained_min!(AbsLog{2}(), a, b, A)
    MatrixCovers.boost_feasible!(a, b, A)
    isnothing(maxiter) ? MatrixCovers.tighten_cover!(a, b, A) :
        MatrixCovers.tighten_cover!(a, b, A; maxiter = Int(maxiter))
    MatrixCovers._balance_cover!(a, b, A)
    MatrixCovers.inflate_feasible!(a, b, A)
    return vcat(a, b)
end

# --- Soft covers -------------------------------------------------------------
#
# `MatrixCovers.soft_symcover_min`/`soft_cover_min` (`AbsLog{2}`) reduce to
# `_symcover_min_abslog2`/`_cover_min_abslog2` (the native hard-cover kernel)
# with `κs = ()`, `boost = false` — no penalty continuation, no boost to
# feasibility, giving the unconstrained minimum with no coverage constraint.
# That kernel is not exported, but calling it directly removes two layers of
# `kwargs...` forwarding `--trim=safe` cannot resolve. Three further
# conditions are needed for the call to resolve when reached (even
# transitively, even through further `@noinline` layers) from a function
# whose own signature carries an optional (`Union{T,Nothing}`) keyword — as
# every `@api` entrypoint does: `maxiter` must always be supplied (a
# local copy of the kernel's own default, 40, standing in when the caller
# omits it, since keyword *presence* varying across call sites is itself
# unresolvable); `κs` must be a concretely and consistently typed empty
# container (`Float64[]`, not `()` — the untyped empty tuple forces a second,
# mismatched specialization of the kernel alongside the one its
# `NTuple{4,Float64}` default already requires); and each of `soft_symcover_min`'s
# reachable `@api` entrypoints must resolve the optional keyword to a
# concrete sentinel (`-1` for "omitted") and delegate to its *own* `@noinline`
# dispatcher and kernel, never sharing either with another entrypoint —
# reachability from more than one `@api` root, or a shared kernel across
# entrypoints whose own keyword counts differ, reintroduces the failure.
#
# `soft_symcover`/`soft_cover` have three optional keywords (`maxiter`,
# `starts`, `sigma`), and `--trim=safe` cannot resolve a call into the
# `_soft_*_abslog2`/`AbsLinear` machinery from a function with *that
# many* optional keywords of its own, independent of which branch is taken and
# regardless of intervening `@noinline` layers — unlike `soft_symcover_min`
# (one optional keyword), which resolves the identical call fine. Every
# keyword is therefore resolved to a concrete sentinel value (`-1`/`NaN` for
# "omitted"; a real `maxiter`/`starts` is always non-negative, a real `sigma`
# always finite) in the entrypoint, and the branch on `penalty` and the
# sentinels is pulled into its own `@noinline` dispatcher — the entrypoint's
# own body touches nothing but that resolution, so no optional-keyword type
# reaches any call.

@noinline function _soft_symcover_dispatch(A::Matrix{Float64}, penalty::Penalty,
                                           mi::Int64, st::Int64, sg::Float64, seed::Int64)
    if penalty === abslog1
        return mi < 0 ? MatrixCovers.soft_symcover(AbsLog{1}(), A) :
            MatrixCovers.soft_symcover(AbsLog{1}(), A; maxiter = mi)
    end
    if penalty === abslog2
        a, _ = _soft_symcover_min_dispatch_kernel(A, mi < 0 ? 40 : mi)
        return a
    end
    if penalty === abslinear1
        m = mi < 0 ? 20 : mi
        s = st < 0 ? 5 : st
        return isnan(sg) ? _soft_symcover_call(AbsLinear{1}(), A, m, s, seed) :
            _soft_symcover_call(AbsLinear{1}(), A, m, s, sg, seed)
    end
    m = mi < 0 ? 32 : mi
    s = st < 0 ? 5 : st
    return isnan(sg) ? _soft_symcover_call(AbsLinear{2}(), A, m, s, seed) :
        _soft_symcover_call(AbsLinear{2}(), A, m, s, sg, seed)
end

@noinline _soft_symcover_min_dispatch_kernel(A::Matrix{Float64}, mi::Int64) =
    MatrixCovers._symcover_min_abslog2(A; maxiter = mi, κs = Float64[], boost = false, fname = :soft_symcover)

@noinline _soft_symcover_call(::AbsLinear{1}, A::Matrix{Float64}, mi::Int64, st::Int64, seed::Int64) =
    MatrixCovers.soft_symcover(AbsLinear{1}(), A; maxiter = mi, starts = st, rng = MersenneTwister(seed))
@noinline _soft_symcover_call(::AbsLinear{1}, A::Matrix{Float64}, mi::Int64, st::Int64, sg::Float64, seed::Int64) =
    MatrixCovers.soft_symcover(AbsLinear{1}(), A; maxiter = mi, starts = st, sigma = sg, rng = MersenneTwister(seed))
@noinline _soft_symcover_call(::AbsLinear{2}, A::Matrix{Float64}, mi::Int64, st::Int64, seed::Int64) =
    MatrixCovers.soft_symcover(AbsLinear{2}(), A; maxiter = mi, starts = st, rng = MersenneTwister(seed))
@noinline _soft_symcover_call(::AbsLinear{2}, A::Matrix{Float64}, mi::Int64, st::Int64, sg::Float64, seed::Int64) =
    MatrixCovers.soft_symcover(AbsLinear{2}(), A; maxiter = mi, starts = st, sigma = sg, rng = MersenneTwister(seed))

@noinline function _soft_cover_dispatch(A::Matrix{Float64}, penalty::Penalty,
                                        mi::Int64, st::Int64, sg::Float64, seed::Int64)
    if penalty === abslog1
        a, b = mi < 0 ? MatrixCovers.soft_cover(AbsLog{1}(), A) :
            MatrixCovers.soft_cover(AbsLog{1}(), A; maxiter = mi)
        return vcat(a, b)
    end
    if penalty === abslog2
        a, b, _ = _soft_cover_min_dispatch_kernel(A, mi < 0 ? 40 : mi)
        return vcat(a, b)
    end
    if penalty === abslinear1
        m = mi < 0 ? 100 : mi
        s = st < 0 ? 4 : st
        a, b = isnan(sg) ? _soft_cover_call(AbsLinear{1}(), A, m, s, seed) :
            _soft_cover_call(AbsLinear{1}(), A, m, s, sg, seed)
        return vcat(a, b)
    end
    m = mi < 0 ? 200 : mi
    s = st < 0 ? 4 : st
    a, b = isnan(sg) ? _soft_cover_call(AbsLinear{2}(), A, m, s, seed) :
        _soft_cover_call(AbsLinear{2}(), A, m, s, sg, seed)
    return vcat(a, b)
end

@noinline _soft_cover_min_dispatch_kernel(A::Matrix{Float64}, mi::Int64) =
    MatrixCovers._cover_min_abslog2(A; maxiter = mi, κs = Float64[], boost = false)

@noinline _soft_cover_call(::AbsLinear{1}, A::Matrix{Float64}, mi::Int64, st::Int64, seed::Int64) =
    MatrixCovers.soft_cover(AbsLinear{1}(), A; maxiter = mi, starts = st, rng = MersenneTwister(seed))
@noinline _soft_cover_call(::AbsLinear{1}, A::Matrix{Float64}, mi::Int64, st::Int64, sg::Float64, seed::Int64) =
    MatrixCovers.soft_cover(AbsLinear{1}(), A; maxiter = mi, starts = st, sigma = sg, rng = MersenneTwister(seed))
@noinline _soft_cover_call(::AbsLinear{2}, A::Matrix{Float64}, mi::Int64, st::Int64, seed::Int64) =
    MatrixCovers.soft_cover(AbsLinear{2}(), A; maxiter = mi, starts = st, rng = MersenneTwister(seed))
@noinline _soft_cover_call(::AbsLinear{2}, A::Matrix{Float64}, mi::Int64, st::Int64, sg::Float64, seed::Int64) =
    MatrixCovers.soft_cover(AbsLinear{2}(), A; maxiter = mi, starts = st, sigma = sg, rng = MersenneTwister(seed))

@noinline function _soft_symcover_min_dispatch(A::Matrix{Float64}, penalty::Penalty, mi::Int64)
    penalty === abslog2 && return _soft_symcover_min_only_kernel(A, mi < 0 ? 40 : mi)
    penalty === abslog1 && throw(ArgumentError("MatrixCovers does not implement AbsLog{1} for soft_symcover_min"))
    throw(ArgumentError(_EXT_IPOPT))
end

@noinline _soft_symcover_min_only_kernel(A::Matrix{Float64}, mi::Int64) =
    MatrixCovers._symcover_min_abslog2(A; maxiter = mi, κs = Float64[], boost = false, fname = :soft_symcover_min)[1]

@noinline function _soft_cover_min_dispatch(A::Matrix{Float64}, penalty::Penalty, mi::Int64)
    if penalty === abslog2
        a, b, _ = _soft_cover_min_only_kernel(A, mi < 0 ? 40 : mi)
        return vcat(a, b)
    end
    penalty === abslog1 && throw(ArgumentError("MatrixCovers does not implement AbsLog{1} for soft_cover_min"))
    throw(ArgumentError(_EXT_IPOPT))
end

@noinline _soft_cover_min_only_kernel(A::Matrix{Float64}, mi::Int64) =
    MatrixCovers._cover_min_abslog2(A; maxiter = mi, κs = Float64[], boost = false)
