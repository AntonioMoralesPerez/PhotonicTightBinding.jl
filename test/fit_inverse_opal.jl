using PhotonicTightBinding, Test
using Crystalline
using DelimitedFiles: readdlm
using LinearAlgebra: norm
using Random: seed!
using Statistics: mean

# As `fit_single_gyroid.jl`, but for the inverse opal (SG 225), cf.
# `examples/fit_inverse_opal.jl`. Two deliberate departures from that example: fewer
# multi-starts, since the fit dominates the runtime here, and the `:strict` longitudinal
# penalty, which places the longitudinal bands at ω ≈ 0.007 rather than the 0.19 left by the
# default `:hinge`. Together with the gyroid test, which uses `:hinge`, this covers both
# penalties.
@testset "photonic fit: inverse opal (SG 225)" begin
    datadir = joinpath(@__DIR__, "..", "examples", "data")
    freqs = readdlm(joinpath(datadir, "mpb_inverse_opal_freqs.csv"), ',', Float64)
    kvs = collect(eachrow(readdlm(joinpath(datadir, "mpb_inverse_opal_kvs.csv"), ',', Float64)))
    @test size(freqs, 1) == length(kvs)

    sgnum = 225
    Rm = stack(primitivize(directbasis(sgnum, Val(3)), centering(sgnum)))
    cbrs = calc_bandreps(sgnum, Val(3))
    cbr = @composite cbrs[12] # (8a|T₂)

    _tbm = tb_hamiltonian(cbr, [[0, 0, 0], [2, 1, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0], [1, 1, 1]])
    sort!(_tbm.terms, by = t -> norm(Rm * t.block.h_orbit.representative())) # order by range
    tbm = TightBindingModel(_tbm.terms[2:31], _tbm.cbr, _tbm.positions, _tbm.N)
    @test length(tbm) == 30
    @test tbm.N == 6

    μᵀ = 2
    μᴸ = tbm.N - μᵀ
    freqs_r = freqs[:, 1:μᵀ]

    seed!(1234)
    ptbm = photonic_fit(tbm, freqs_r, kvs; max_multistarts = 10,
                        longitudinal_penalty = :strict, longitudinal_weight = 10,
                        longitudinal_width = 1e-2)
    freqs_fit = spectrum(ptbm, kvs; transform = energy2frequency)

    Δ = freqs_fit[:, (μᴸ+1):end] - freqs_r
    @test sqrt(mean(abs2, Δ)) < 0.06            # 3.6e-2 c/a as measured
    @test maximum(freqs_fit[:, 1:μᴸ]) < 0.05    # 7.4e-3 as measured
end
