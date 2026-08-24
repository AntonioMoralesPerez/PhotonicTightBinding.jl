using PhotonicTightBinding, Test
using Crystalline
using PhotonicTightBinding: PhotonicBandConnectivity as PBC

@testset "EBR decomposition" begin
    @testset "SG #2" begin
        sgnum, D = 2, 3
        brs = calc_bandreps(sgnum, Val(D))
        lgirsv = irreps(brs)
        ns = SymmetryVector{D}[]

        # do some checks for interesting symmetry vectors
        s1 = "[-Γ₁⁺+3Γ₁⁻,2R₁⁻, 2T₁⁻, 2U₁⁻, V₁⁺+V₁⁻, X₁⁺+X₁⁻,Y₁⁺+Y₁⁻,2Z₁⁺]"
        push!(ns, parse(SymmetryVector, s1, lgirsv))

        s2 = "[-Γ₁⁺+3Γ₁⁻, 2R₁⁻, T₁⁺ + T₁⁻, U₁⁺ + U₁⁻, V₁⁺ + V₁⁻, 2X₁⁻, 2Y₁⁻, 2Z₁⁺]"
        push!(ns, parse(SymmetryVector, s2, lgirsv))

        s3 = "[-Γ₁⁺+3Γ₁⁻, R₁⁺+R₁⁻, T₁⁺ + T₁⁻, 2 U₁⁻, 2 V₁⁻, X₁⁺+X₁⁻, 2Y₁⁻, 2Z₁⁺]"
        push!(ns, parse(SymmetryVector, s3, lgirsv))

        s4 = "[-Γ₁⁺+3Γ₁⁻, R₁⁺+R₁⁻, 2 T₁⁻, U₁⁺ + U₁⁻, 2 V₁⁻, 2X₁⁻, Y₁⁺+Y₁⁻, 2Z₁⁺]"
        push!(ns, parse(SymmetryVector, s4, lgirsv))

        s5 = "[-Γ₁⁺+3Γ₁⁻, 2R₁⁻, T₁⁺ + T₁⁻, 2 U₁⁻, 2 V₁⁺, 2X₁⁻, Y₁⁺+Y₁⁻, Z₁⁺+Z₁⁻]"
        push!(ns, parse(SymmetryVector, s5, lgirsv))

        for n in ns
            # `find_bandrep_decompositions` returns at the first μᴸ ≥ μᴸ_min admitting a
            # decomposition, so each μᴸ below exercises a distinct number of longitudinal
            # modes. The ceiling is purely a cost cap: for these vectors μᴸ = 3 is ~30×
            # slower than μᴸ = 2, and μᴸ = 4 takes ~8 min apiece.
            for μᴸ in 1:2
                candidatesv = find_bandrep_decompositions(n, brs; μᴸ_min = μᴸ)

                # we should find at least one decomposition
                isempty(candidatesv) && continue

                for candidates in candidatesv

                    # the decomposition should match the symmetry vector
                    @test !isnothing(candidates.longitudinal)
                    @test !isnothing(candidates.apolarv)

                    for nᵀ⁺ᴸ in candidates.apolarv
                        vᵀ = SymmetryVector(nᵀ⁺ᴸ - candidates.longitudinal) # SymVec of nᵀ

                        @test occupation(n) == occupation(vᵀ)
                        @test irreps(n) == irreps(vᵀ)

                        for (i, mult) in enumerate(multiplicities(n))
                            klabel(irreps(n)[i][1]) == "Γ" && continue

                            @test mult == multiplicities(vᵀ)[i]
                        end
                    end

                    # the decompositions should be physical 
                    @test !isnothing(candidates.ps)
                    for p in candidates.ps
                        @test all(isinteger, p)
                    end
                end
            end
        end
    end # SG 2

    @testset "transverse solutions from PhotonicBandConnectivity" begin
        # `transverse_symmetry_vectors` w/ its default `separate_vrep = true` returns
        # vectors carrying a synthetic virtual irrep at Γ, which
        # `find_bandrep_decompositions` rejects with an informative error; the vrep-free
        # `separate_vrep = false` form must decompose fine

        # SG 1
        brs = calc_bandreps(1, Val(3))
        m_t = PBC.transverse_symmetry_vectors(1, Val(3))[1] # μᵀ = 2 solution
        m_f = PBC.transverse_symmetry_vectors(1, Val(3); separate_vrep = false)[1]
        @test_throws "virtual irrep at Γ" find_bandrep_decompositions(m_t, brs)
        c = only(find_bandrep_decompositions(m_f, brs))
        @test iszero(occupation(c.longitudinal)) # μᴸ = 0
        @test only(c.apolarv).coefs == [2]       # 2(1a|A)

        # SG 2 (centrosymmetric; exercises the unpinned Γ-irrep path at ω=0)
        brs = calc_bandreps(2, Val(3))
        sols_t = PBC.transverse_symmetry_vectors(2, Val(3))
        sols_f = PBC.transverse_symmetry_vectors(2, Val(3); separate_vrep = false)
        # pick a solution (by content, as sort order may vary) that decomposes at μᴸ = 1
        s = "[2Z₁⁺, 2Y₁⁺, 2U₁⁻, X₁⁺+X₁⁻, T₁⁺+T₁⁻, -Γ₁⁺+3Γ₁⁻, 2V₁⁺, R₁⁺+R₁⁻]"
        n_ref = parse(SymmetryVector, s, irreps(brs))
        i = something(findfirst(n -> multiplicities(n) == multiplicities(n_ref), sols_f))
        @test_throws "virtual irrep at Γ" find_bandrep_decompositions(sols_t[i], brs)
        cv_f = find_bandrep_decompositions(sols_f[i], brs; μᴸ_max = 1)
        @test !isempty(cv_f)
        @test all(all(isinteger, p) for c in cv_f for p in c.ps)
    end # transverse solutions from PhotonicBandConnectivity
end # EBR decomposition
