@testset "Unitful" begin

    # A Hessian-like matrix whose coordinates carry units: A[i,j] has units
    # 1/(u[i]*u[j]), so a cover of it has units 1/u[i] per coordinate.
    L, V, F = u"m", u"m/s", u"N"
    A = [1e6/L^2   1e3/(L*V)  1.0/(L*F);
         1e3/(L*V) 1.0/V^2    1e-3/(V*F);
         1.0/(L*F) 1e-3/(V*F) 1e-6/F^2]
    UA = [u"m^-1", u"s/m", u"N^-1"]

    @testset "cover carries the coordinate units" begin
        a = symcover(A)
        @test unit.(a) == UA
        @test ustrip.(a) ≈ [1e3, 1.0, 1e-3]
        @test iscover(a, A; rtol=8eps())
        # The units are read off `A` as written: `N` stays `N` rather than
        # decaying to the SI base units it is defined from.
        @test unit(a[3]) == u"N^-1" != upreferred(u"N^-1")
    end

    @testset "same cover in every unit system" begin
        # The same physical matrix, named in mm, km/hr, kN.
        U2 = (u"mm", u"km/hr", u"kN")
        A2 = [uconvert(unit(1 / (u * v)), A[i, j]) for (i, u) in enumerate(U2), (j, v) in enumerate(U2)]
        a, a2 = symcover(A), symcover(A2)
        @test unit.(a2) == [u"mm^-1", u"hr/km", u"kN^-1"]
        # Scale invariance: renaming the units renames the answer and changes
        # nothing physical.
        @test all(isapprox(ustrip(upreferred(a2[i] / a[i])), 1; rtol=1e-12) for i in eachindex(a))
        @test iscover(a2, A2; rtol=8eps())
    end

    @testset "coordinates on mixed scales" begin
        # Coordinate 1 in m, coordinate 2 in mm. Julia promotes this to a common
        # unit whenever the entries share a dimension, so an eltype that admits
        # heterogeneous units is what preserves the scales as written.
        H = Quantity[4.0u"m^-2"        1.0u"m^-1*mm^-1";
                     1.0u"m^-1*mm^-1" 4.0u"mm^-2"]
        a = symcover(H)
        @test unit.(a) == [u"m^-1", u"mm^-1"]
        @test ustrip.(a) ≈ [2.0, 2.0]
        @test iscover(a, H; rtol=8eps())
    end

    @testset "asymmetric covers" begin
        # Rows indexed by (m, s), columns by (kg, K, m/s).
        ru, cu = (u"m", u"s"), (u"kg", u"K", u"m/s")
        B = [1.0 / (r * c) for r in ru, c in cu] .* [1e3 1e-2 5.0; 2.0 1e4 1e-1]
        a, b = cover(B)
        @test unit.(a) == [u"m^-1", u"s^-1"]
        @test unit.(b) == [u"kg^-1", u"K^-1", u"s/m"]
        @test iscover(a, b, B; rtol=8eps())

        # The unit gauge `a -> a*c`, `b -> b/c` is pinned by minimizing the total
        # atomic-unit powers the two vectors carry, so a factor shared by every
        # entry lands on whichever side has fewer of them.
        aJ, bJ = cover(B .* u"J")
        @test unit.(aJ) == [u"J/m", u"J/s"]
        @test unit.(bJ) == unit.(b)
        aJ, bJ = cover(permutedims(B) .* u"J")
        @test unit.(aJ) == unit.(b)
        @test unit.(bJ) == [u"J/m", u"J/s"]
    end

    @testset "cover reproduces symcover on symmetric input" begin
        # The gauge takes its smallest-magnitude optimizer, which is trivial here:
        # a symmetric matrix pins `unit(a[i])` outright via `a[i]^2 == A[i,i]`.
        a, b = cover(A)
        @test unit.(a) == unit.(b) == UA
        @test a == b
        @test ustrip.(a) ≈ ustrip.(symcover(A))
    end

    @testset "uniform units" begin
        # A concrete element type names one unit for every entry. The gauge's
        # median interval is not a single point here -- `∑|t| + ∑|t+2|` is flat
        # across `t ∈ [-2, 0]` -- so only its midpoint reproduces symcover.
        Uni = [4.0 1.0; 1.0 4.0] .* u"m^-2"
        @test isconcretetype(eltype(Uni))
        @test unit.(symcover(Uni)) == [u"m^-1", u"m^-1"]
        a, b = cover(Uni)
        @test unit.(a) == unit.(b) == [u"m^-1", u"m^-1"]
        @test iscover(a, b, Uni; rtol=8eps())

        # An odd exponent puts a fractional one on the cover: `a[i]^2 == A[i,i]`
        # leaves symcover no choice, so cover must agree.
        Odd = [4.0 1.0; 1.0 4.0] .* u"m^-1"
        @test unit.(symcover(Odd)) == [u"m"^(-1 // 2), u"m"^(-1 // 2)]
        @test all(unit.(v) == [u"m"^(-1 // 2), u"m"^(-1 // 2)] for v in cover(Odd))
    end

    @testset "sparse storage" begin
        # Sparse storage synthesizes structural zeros with `zero(eltype)`, so a
        # sparse matrix of quantities carries one unit throughout.
        S = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0] .* u"m^-2")
        a = symcover(S)
        @test unit.(a) == fill(u"m^-1", 3)
        @test ustrip.(a) ≈ symcover(ustrip.(S))
        @test iscover(a, S; rtol=8eps())
        @test all(unit.(v) == fill(u"m^-1", 3) for v in cover(S))
        @test unit.(symcover(Symmetric(S))) == fill(u"m^-1", 3)

        # The refiners are where MatrixCoversSparseArraysExt and MatrixCoversUnitfulExt overlap.
        a = initialize_symcover(S)
        @test unit.(symcover_min!(AbsLog{2}(), a, S)) == fill(u"m^-1", 3)
        @test unit.(symcover_min!(AbsLog{2}(), a, Symmetric(S))) == fill(u"m^-1", 3)
        a, b = initialize_cover(S)
        @test all(unit.(v) == fill(u"m^-1", 3) for v in cover_min!(AbsLog{2}(), a, b, S))
    end

    @testset "balance convention holds in the caller's units" begin
        # `A` is stripped as written rather than in a canonical system, so the
        # (non-scale-invariant) balance that splits `a` from `b` is pinned to the
        # scale the caller named.
        ru, cu = (u"m", u"s"), (u"kg", u"K", u"m/s")
        B = [1.0 / (r * c) for r in ru, c in cu] .* [1e3 1e-2 5.0; 2.0 1e4 1e-1]
        a, b = cover(B)
        @test isbalanced(ustrip.(a), ustrip.(b), ustrip.(B))
    end

    @testset "units must factor" begin
        # No `a`, `b` exist with unit(A[i,j]) == unit(a[i])*unit(b[j]).
        E = Quantity[1.0u"m^2" 1.0u"s"; 1.0u"s" 1.0u"kg^2"]
        @test_throws "units of `A` do not factor" symcover(E)
        @test_throws "unit(A[2,2])*unit(A[1,1]) = kg^2 m^2" symcover(E)
        @test_throws "unit(A[2,1])^2 = s^2" symcover(E)
        @test_throws "`A*x` is undefined for every `x`" symcover(E)
        @test_throws "units of `A` do not factor" cover(E)
        @test_throws DimensionMismatch cover(E)

        # Dimensionally consistent, but the off-diagonal is named at a scale the
        # diagonal contradicts: every entry is 𝐋^-2, yet no `a` covers them.
        M = Quantity[1.0u"m^-2" 1.0u"m^-2"; 1.0u"m^-2" 1.0u"mm^-2"]
        @test all(==(dimension(u"m^-2")), dimension.(M))
        @test_throws "units of `A` do not factor" symcover(M)
    end

    @testset "offset axes" begin
        Ao = OffsetArray(A, 0:2, 0:2)
        a = symcover(Ao)
        @test axes(a) == (0:2,)
        @test unit.(a) == OffsetArray(UA, 0:2)
        @test ustrip.(parent(a)) ≈ ustrip.(symcover(A))
        @test iscover(a, Ao; rtol=8eps())

        ao, bo = cover(Ao)
        @test axes(ao) == axes(bo) == (0:2,)
        @test iscover(ao, bo, Ao; rtol=8eps())
    end

    @testset "transposed views" begin
        ru, cu = (u"m", u"s"), (u"kg", u"K", u"m/s")
        B = [1.0 / (r * c) for r in ru, c in cu] .* [1e3 1e-2 5.0; 2.0 1e4 1e-1]
        for Bt in (transpose(B), adjoint(B))
            a, b = cover(Bt)
            @test unit.(a) == [u"kg^-1", u"K^-1", u"s/m"]
            @test unit.(b) == [u"m^-1", u"s^-1"]
            @test iscover(a, b, Bt; rtol=8eps())

            a, b = Vector{Quantity{Float64}}(undef, 3), Vector{Quantity{Float64}}(undef, 2)
            @test cover!(a, b, Bt) == (a, b)
            @test unit.(a) == [u"kg^-1", u"K^-1", u"s/m"]
            @test unit.(b) == [u"m^-1", u"s^-1"]
            @test iscover(a, b, Bt; rtol=8eps())
        end
    end

    @testset "the whole family" begin
        for f in (symcover, initialize_symcover, soft_symcover, symcover_min, soft_symcover_min)
            a = f(A)
            @test unit.(a) == UA
            @test eltype(ustrip.(a)) <: AbstractFloat
        end
        for f in (cover, initialize_cover, soft_cover, cover_min, soft_cover_min)
            a, b = f(A)
            @test unit.(a) == unit.(b) == UA
        end
        for ϕ in PENALTIES
            @test unit.(symcover(ϕ, A)) == UA
            @test unit.(soft_symcover(ϕ, A)) == UA
            @test unit.(symcover_min(ϕ, A)) == UA
            @test all(unit.(v) == UA for v in cover(ϕ, A))
            @test all(unit.(v) == UA for v in cover_min(ϕ, A))
        end
        for ϕ in (AbsLinear{1}(), AbsLinear{2}())
            @test all(unit.(v) == UA for v in soft_cover(ϕ, A))
            @test unit.(soft_symcover_min(ϕ, A)) == UA
            @test all(unit.(v) == UA for v in soft_cover_min(ϕ, A))
        end
        # `soft_cover_min` also takes the log-domain penalty, which has an
        # analytic minimum.
        @test all(unit.(v) == UA for v in soft_cover_min(AbsLog{2}(), A))
    end

    @testset "mutating forms" begin
        # `a` is written, not read: undefined references are a valid destination.
        a = Vector{Quantity{Float64}}(undef, 3)
        @test symcover!(a, A) === a
        @test unit.(a) == UA
        @test iscover(a, A; rtol=8eps())

        a, b = Vector{Quantity{Float64}}(undef, 3), Vector{Quantity{Float64}}(undef, 3)
        @test cover!(a, b, A) == (a, b)
        @test unit.(a) == unit.(b) == UA

        # The heuristics ignore `ϕ`, so the forms that take one agree with those
        # that do not.
        a = Vector{Quantity{Float64}}(undef, 3)
        @test symcover!(AbsLog{2}(), a, A) === a
        @test unit.(a) == UA
        @test a == symcover(A)

        a, b = Vector{Quantity{Float64}}(undef, 3), Vector{Quantity{Float64}}(undef, 3)
        @test cover!(AbsLog{2}(), a, b, A) == (a, b)
        @test unit.(a) == unit.(b) == UA
        @test (a, b) == cover(A)

        a = Vector{Quantity{Float64}}(undef, 3)
        @test initialize_symcover!(a, A) === a
        @test unit.(a) == UA

        a, b = Vector{Quantity{Float64}}(undef, 3), Vector{Quantity{Float64}}(undef, 3)
        @test initialize_cover!(a, b, A) == (a, b)
        @test unit.(a) == unit.(b) == UA

        # The `*_min!` family reads `a` as a start, so `initialize_*` feeds them
        # directly: both sides name the units the same way.
        a = initialize_symcover(A)
        @test symcover_min!(AbsLog{2}(), a, A) === a
        @test unit.(a) == UA
        @test iscover(a, A; rtol=1e-8)

        a, b = initialize_cover(A)
        @test cover_min!(AbsLog{2}(), a, b, A) == (a, b)
        @test unit.(a) == unit.(b) == UA

        # The no-ϕ forms of the `*_min` family default their penalty, and carry
        # the units the same way.
        a = initialize_symcover(A)
        @test symcover_min!(a, A) === a
        @test unit.(a) == UA
        @test iscover(a, A; rtol=1e-8)

        a, b = initialize_cover(A)
        @test cover_min!(a, b, A) == (a, b)
        @test unit.(a) == unit.(b) == UA

        a = initialize_symcover(A)
        @test soft_symcover_min!(a, A) === a
        @test unit.(a) == UA

        a, b = initialize_cover(A)
        @test soft_cover_min!(a, b, A) == (a, b)
        @test unit.(a) == unit.(b) == UA

        a = initialize_symcover(A)
        @test soft_symcover_min!(AbsLinear{2}(), a, A) === a
        @test unit.(a) == UA

        a, b = initialize_cover(A)
        @test soft_cover_min!(AbsLog{2}(), a, b, A) == (a, b)
        @test unit.(a) == unit.(b) == UA

        # A start in a dimensionally equivalent spelling is converted, not rejected.
        a = initialize_symcover(A)
        a2 = [uconvert(u, x) for (u, x) in zip((u"mm^-1", u"hr/km", u"kN^-1"), a)]
        @test ustrip.(symcover_min!(AbsLog{2}(), a2, A)) ≈ ustrip.(symcover_min!(AbsLog{2}(), a, A))

        # A start whose dimensions are wrong is not.
        a3 = Quantity{Float64}[1.0u"m", 1.0u"s", 1.0u"N"]
        @test_throws Unitful.DimensionError symcover_min!(AbsLog{2}(), a3, A)
    end

    @testset "scoring a dimensional cover" begin
        # `unit(A[i,j]) == unit(a[i])*unit(b[j])` makes every ratio the objective
        # forms dimensionless, so the score is a plain number and is identical to
        # the score of the unit-stripped problem.
        ϕs = (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())

        a = symcover(A)
        for ϕ in ϕs
            s = cover_objective(ϕ, a, A)
            @test s isa Float64
            @test s == cover_objective(ϕ, ustrip.(a), ustrip.(A))
        end

        # Heterogeneous units across coordinates, not just a single uniform unit.
        H = Quantity[4.0u"m^-2"        1.0u"m^-1*mm^-1";
                     1.0u"m^-1*mm^-1" 4.0u"mm^-2"]
        h = symcover(H)
        for ϕ in ϕs
            @test cover_objective(ϕ, h, H) == cover_objective(ϕ, ustrip.(h), ustrip.(H))
        end

        # The asymmetric form, whose row and column scales carry different units.
        ru, cu = (u"m", u"s"), (u"kg", u"K", u"m/s")
        B = [1.0 / (r * c) for r in ru, c in cu] .* [1e3 1e-2 5.0; 2.0 1e4 1e-1]
        ab, bb = cover(B)
        for ϕ in ϕs
            s = cover_objective(ϕ, ab, bb, B)
            @test s isa Float64
            @test s == cover_objective(ϕ, ustrip.(ab), ustrip.(bb), ustrip.(B))
        end

        # Scale invariance: renaming the units cannot change a dimensionless score.
        # `A`'s cover is exact, so both scores sit at rounding level and only an
        # absolute tolerance is meaningful here.
        U2 = (u"mm", u"km/hr", u"kN")
        A2 = [uconvert(unit(1 / (u * v)), A[i, j]) for (i, u) in enumerate(U2), (j, v) in enumerate(U2)]
        for ϕ in ϕs
            @test cover_objective(ϕ, symcover(A2), A2) ≈ cover_objective(ϕ, a, A) atol = 1e-12
        end
    end

end
