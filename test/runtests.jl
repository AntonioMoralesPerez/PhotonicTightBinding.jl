using PhotonicTightBinding, Test

@testset "PhotonicTightBinding" begin
    # decompositions into EBRs
    include("ebr_decomposition.jl")
    include("mpb_ebr_decomposition.jl")
end
