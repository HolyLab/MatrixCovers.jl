# Cross-notion invariants: conventions documented for every cover notion,
# checked uniformly across all of them. Each entry supplies the solver as
# `A -> a` (symmetric) or `A -> (a, b)` (general), whether it promises hard
# feasibility, and the tolerance its algorithm warrants. The themed files pin
# each notion's algorithm-specific precision; this file pins the shared
# conventions:
#   - repeated calls return identical results
#   - hard covers are feasible
#   - results co-vary with diagonal rescaling of A
#   - an entirely unsupported row/column gets scale exactly 0
#   - results depend only on entry magnitudes (complex input ≡ abs.(A))
#   - offset axes propagate from A to the scale vectors

const SYM_NOTIONS = (
    (name = "symcover",                        f = A -> symcover(A),                          hard = true,  rtol = 1e-9),
    (name = "symcover_min(AbsLog{2})",         f = A -> symcover_min(AbsLog{2}(), A),         hard = true,  rtol = 1e-5),
    (name = "symcover_min(AbsLog{1})",         f = A -> symcover_min(AbsLog{1}(), A),         hard = true,  rtol = 1e-5),
    (name = "symcover_min(AbsLinear{1})",      f = A -> symcover_min(AbsLinear{1}(), A),      hard = true,  rtol = 1e-5),
    (name = "symcover_min(AbsLinear{2})",      f = A -> symcover_min(AbsLinear{2}(), A),      hard = true,  rtol = 1e-5),
    (name = "soft_symcover(AbsLog{2})",        f = A -> soft_symcover(AbsLog{2}(), A),        hard = false, rtol = 1e-9),
    (name = "soft_symcover(AbsLog{1})",        f = A -> soft_symcover(AbsLog{1}(), A),        hard = false, rtol = 1e-8),
    (name = "soft_symcover(AbsLinear{2})",     f = A -> soft_symcover(AbsLinear{2}(), A),     hard = false, rtol = 1e-8),
    (name = "soft_symcover(AbsLinear{1})",     f = A -> soft_symcover(AbsLinear{1}(), A),     hard = false, rtol = 1e-8),
    (name = "soft_symcover_min(AbsLog{2})",    f = A -> soft_symcover_min(AbsLog{2}(), A),    hard = false, rtol = 1e-5),
    (name = "soft_symcover_min(AbsLinear{1})", f = A -> soft_symcover_min(AbsLinear{1}(), A), hard = false, rtol = 1e-5),
    (name = "soft_symcover_min(AbsLinear{2})", f = A -> soft_symcover_min(AbsLinear{2}(), A), hard = false, rtol = 1e-5),
)

const GEN_NOTIONS = (
    (name = "cover",                        f = A -> cover(A),                        hard = true,  rtol = 1e-9),
    (name = "cover_min(AbsLog{2})",         f = A -> cover_min(AbsLog{2}(), A),       hard = true,  rtol = 1e-5),
    (name = "cover_min(AbsLog{1})",         f = A -> cover_min(AbsLog{1}(), A),       hard = true,  rtol = 1e-5),
    (name = "soft_cover(AbsLinear{2})",     f = A -> soft_cover(AbsLinear{2}(), A),   hard = false, rtol = 1e-8),
    (name = "soft_cover(AbsLinear{1})",     f = A -> soft_cover(AbsLinear{1}(), A),   hard = false, rtol = 1e-8),
    (name = "soft_cover_min(AbsLog{2})",    f = A -> soft_cover_min(AbsLog{2}(), A),  hard = false, rtol = 1e-9),
)

@testset "cross-notion invariants" begin
    Asym  = [2.0 1.0 0.5; 1.0 3.0 1.0; 0.5 1.0 2.5]
    Azsym = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 3.0]     # row/column 2 unsupported
    Hc    = [2.0 1.0+1.0im 0.0; 1.0-1.0im 3.0 0.5im; 0.0 -0.5im 2.5]  # Hermitian values
    d     = [2.0, 0.5, 4.0]

    @testset "sym: $(nt.name)" for nt in SYM_NOTIONS
        a = nt.f(Asym)
        @test a == nt.f(Asym)
        if nt.hard
            @test iscover(a, Asym; rtol=1e-8, atol=1e-6)
        end
        @test covaries(nt.f, Asym, d; rtol=nt.rtol)
        az = nt.f(Azsym)
        @test az[2] == 0
        @test nt.f(Hermitian(Hc)) ≈ nt.f(abs.(Hc)) rtol=nt.rtol
        Ao = OffsetArray(Asym, -1:1, -1:1)
        ao = nt.f(Ao)
        @test axes(ao, 1) == axes(Ao, 1)
        @test collect(ao) ≈ a rtol=nt.rtol
    end

    Agen  = [1.0 2.0 0.5; 0.25 3.0 1.0]                  # rectangular
    Azgen = [1.0 0.0 2.0; 0.0 0.0 0.0; 3.0 0.0 4.0]      # row/column 2 unsupported
    Gc    = [1.0+1.0im 2.0 0.5; 0.25im 3.0 1.0-2.0im]
    dr, dc = [2.0, 0.5], [3.0, 0.25, 1.5]

    @testset "gen: $(nt.name)" for nt in GEN_NOTIONS
        a, b = nt.f(Agen)
        @test (a, b) == nt.f(Agen)
        if nt.hard
            @test iscover(a, b, Agen; rtol=1e-8, atol=1e-6)
        end
        @test covaries(nt.f, Agen, dr, dc; rtol=nt.rtol)
        az, bz = nt.f(Azgen)
        @test az[2] == 0
        @test bz[2] == 0
        # Gauge-invariant comparison: complex input must reproduce abs.(A).
        aC, bC = nt.f(Gc)
        aR, bR = nt.f(abs.(Gc))
        @test aC .* transpose(bC) ≈ aR .* transpose(bR) rtol=nt.rtol
        Ao = OffsetArray(Agen, 0:1, -1:1)
        ao, bo = nt.f(Ao)
        @test axes(ao, 1) == axes(Ao, 1)
        @test axes(bo, 1) == axes(Ao, 2)
        @test collect(ao) .* transpose(collect(bo)) ≈ a .* transpose(b) rtol=nt.rtol
    end
end
