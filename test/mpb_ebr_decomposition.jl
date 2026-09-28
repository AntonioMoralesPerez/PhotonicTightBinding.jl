using PhotonicTightBinding, Test
using Crystalline # for bandreps

@testset "EBR decomposition" begin
    @testset "SG #221" begin

        # construct the structure under study
        R = 0.2 # cylinder radius
        mat = mp.Medium(; epsilon = 12)
        geometry = [
            mp.Cylinder(;
                radius = R,
                center = [0, 0, 0],
                axis = [0, 0, 1],
                height = 1,
                material = mat,
            ),
            mp.Cylinder(;
                radius = R,
                center = [0, 0, 0],
                axis = [0, 1, 0],
                height = 1,
                material = mat,
            ),
            mp.Cylinder(;
                radius = R,
                center = [0, 0, 0],
                axis = [1, 0, 0],
                height = 1,
                material = mat,
            ),
        ]

        # solve the system
        ms = mpb.ModeSolver(;
            num_bands = 8,
            geometry_lattice = mp.Lattice(;
                basis1 = [1, 0, 0],
                basis2 = [0, 1, 0],
                basis3 = [0, 0, 1],
                size = [1, 1, 1],
            ),
            geometry = pylist(geometry), # must be a genuine `list`; mpb aborts otherwise
            resolution = 16,
        )
        ms.init_params(; p = mp.ALL, reset_fields = true)

        # obtain the symmetry vectors of the bands computed above; `brs` is shared with the
        # decomposition below, whose `n` must be built against this collection
        sgnum = 221
        brs = primitivize(bandreps(sgnum, Val(3)))
        ns, topos = obtain_symmetry_vectors(ms, brs)

        for n in ns
            for μᴸ in 1:2 # limited range cf. cost (see μᴸ note in `ebr_decomposition.jl`)
                candidatesv = find_bandrep_decompositions(n, brs; μᴸ_min = μᴸ)

                # we should find at least one decomposition
                isempty(candidatesv) && continue

                for candidates in candidatesv
                    # the decomposition should match the symmetry vector
                    @test !isnothing(candidates.longitudinal)
                    @test !isnothing(candidates.apolarv)

                    for nᵀ⁺ᴸ in candidates.apolarv
                        nᵀ = SymmetryVector(nᵀ⁺ᴸ - candidates.longitudinal) # SymVec of nᵀ

                        @test occupation(n) == occupation(nᵀ)
                        @test irreps(n) == irreps(nᵀ)

                        for (i, mult) in enumerate(multiplicities(n))
                            klabel(irreps(n)[i][1]) == "Γ" && continue
                            @test mult == multiplicities(nᵀ)[i]
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
    end # SG 221
end # EBR decomposition
