using PhotonicTightBinding, Test

@testset "PhotonicTightBinding" begin
    # decompositions into EBRs
    include("bandrep_provenance.jl")
    include("ebr_decomposition.jl")
    include("mpb_ebr_decomposition.jl")

    # fitting against stored MPB references
    include("fit_single_gyroid.jl")
    include("fit_inverse_opal.jl")
    include("weighted_fit.jl")
end
