using PhotonicTightBinding, Test
using Crystalline

# `m` and `brs` need only refer to *equal* irreps, not identical ones (cf. issue #10).
@testset "`m` and `brs` need only refer to equal irreps" begin
    sgnum, D = 2, 3
    brs = calc_bandreps(sgnum, Val(D))
    brs′ = calc_bandreps(sgnum, Val(D)) # equal to `brs`, but a distinct object
    @test irreps(brs) == irreps(brs′)
    @test irreps(brs) !== irreps(brs′)

    s = "[-Γ₁⁺+3Γ₁⁻,2R₁⁻, 2T₁⁻, 2U₁⁻, V₁⁺+V₁⁻, X₁⁺+X₁⁻,Y₁⁺+Y₁⁻,2Z₁⁺]"
    n′ = parse(SymmetryVector, s, irreps(brs′))
    @test !isempty(find_bandrep_decompositions(n′, brs; μᴸ_min = 1))
end
