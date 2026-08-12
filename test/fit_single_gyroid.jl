using PhotonicTightBinding, Test
using Crystalline
using DelimitedFiles: readdlm
using LinearAlgebra: norm
using Random: seed!
using Statistics: mean

# Fits the two lowest bands of the single gyroid (SG 214) to the stored MPB reference in
# `examples/data/`, i.e. the `examples/fit_single_gyroid.jl` pipeline without the plotting.
@testset "photonic fit: single gyroid (SG 214)" begin
    datadir = joinpath(@__DIR__, "..", "examples", "data")
    freqs = readdlm(joinpath(datadir, "mpb_single_gyroid_freqs.csv"), ',', Float64)
    kvs = collect(eachrow(readdlm(joinpath(datadir, "mpb_single_gyroid_kvs.csv"), ',', Float64)))
    @test size(freqs, 1) == length(kvs)

    sgnum = 214
    Rm = stack(primitivize(directbasis(sgnum, Val(3)), centering(sgnum)))
    cbrs = calc_bandreps(sgnum, Val(3))
    cbr = @composite cbrs[6] # (12c|B₁)

    _tbm = tb_hamiltonian(cbr, [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]])
    sort!(_tbm.terms, by = t -> norm(Rm * t.block.h_orbit.representative())) # order by range
    tbm_full = TightBindingModel(_tbm.terms, _tbm.cbr, _tbm.positions, _tbm.N)
    tbm = tbm_full[[2, 3, 4, 5, 6, 8, 10, 11, 12, 21, 27, 29, 31, 32, 33]]
    @test length(tbm) == 15
    @test tbm.N == 6

    μᵀ = 2
    μᴸ = tbm.N - μᵀ
    freqs_r = freqs[:, 1:μᵀ]

    seed!(1234) # the multi-start search draws random initial points
    ptbm = photonic_fit(tbm, freqs_r, kvs; max_multistarts = 25, atol = 1e-8)
    freqs_fit = spectrum(ptbm, kvs; transform = energy2frequency)

    Δ = freqs_fit[:, (μᴸ+1):end] - freqs_r
    @test sqrt(mean(abs2, Δ)) < 5e-3            # 1.7e-3 c/a in the example
    far = freqs_r .> 0.05                       # relative error is meaningless as ω → 0
    @test maximum(abs.(Δ[far]) ./ freqs_r[far]) < 0.15  # 6.6% in the example
    @test maximum(freqs_fit[:, 1:μᴸ]) < 0.1     # longitudinal bands pushed toward 0
end
