# examples/fit_inverse_opal.jl
#
# Fitting a symmetry-constrained tight-binding model to the photonic band structure of an
# inverse opal (air spheres in ε = 13, space group 225).
#
# Unlike `mpb_inverse_opal.jl` — which computes the reference bands with MPB and doubles as
# a scratchpad for the symmetry analysis — this example loads a *stored* MPB reference from
# `data/` and focuses solely on the fitting step. It therefore needs no MPB/Python stack and
# runs anywhere. Regenerate the stored reference with `benchmark/generate_mpb_data.jl`.
#
# Expect ~20 minutes end-to-end: `tb_hamiltonian` needs a couple of minutes for a space group
# this large (cf. SymmetricTightBinding.jl issue #44) and the multi-start fit dominates the
# rest. Lower `max_multistarts` below for a quicker, correspondingly rougher fit.

using Pkg
Pkg.activate(@__DIR__)

using PhotonicTightBinding
using Crystalline           # space groups, band representations
using Brillouin             # k-paths through the Brillouin zone
using DelimitedFiles        # to read the stored MPB reference
using LinearAlgebra: norm
using Statistics: mean
using GLMakie               # for plotting

## ---------------------------------------------------------------------------------------- #
# load the stored MPB reference

sgnum = 225
Rs′ = directbasis(sgnum, Val(3))            # conventional basis
Rs = primitivize(Rs′, centering(sgnum))     # primitive basis (the TB model's setting)
Rm = stack(Rs)

# the reference was sampled on the standard k-path, so we can rebuild the k-points from
# Brillouin rather than storing them: keeping the `KPathInterpolant` gives us labelled axes
# when plotting (the stored k-point file is retained as a consistency check)
kpi = interpolate(irrfbz_path(sgnum, Rs′), 100)
freqs = readdlm(joinpath(@__DIR__, "data", "mpb_inverse_opal_freqs.csv"), ',', Float64)

kvs_stored = eachrow(readdlm(joinpath(@__DIR__, "data", "mpb_inverse_opal_kvs.csv"), ',', Float64))
@assert length(kpi) == size(freqs, 1) == length(kvs_stored)
@assert all(norm(k - kv) < 1e-10 for (k, kv) in zip(kpi, kvs_stored)) "stored k-points do \
    not match the reconstructed k-path (Brillouin version skew?)"

## ---------------------------------------------------------------------------------------- #
# build the tight-binding model

# the two lowest photonic bands of this structure transform as the (8a|T₂) EBR, less the
# longitudinal modes; cf. the symmetry analysis in `mpb_inverse_opal.jl`, which we take as
# given here
cbrs = calc_bandreps(sgnum, Val(3))
cbr = @composite cbrs[12] # (8a|T₂)

# hoppings out to a handful of neighbour shells. Sorting the terms by physical hopping
# distance (`tb_hamiltonian` orders them block-major, not by range) lets us truncate the
# model by range; we keep the 30 shortest-range terms and drop the very first one, an
# overall on-site shift that merely sets the energy reference
_tbm = tb_hamiltonian(cbr, [[0, 0, 0], [2, 1, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0], [1, 1, 1]])
sort!(_tbm.terms, by = t -> norm(Rm * t.block.h_orbit.representative()))
tbm = TightBindingModel(_tbm.terms[2:31], _tbm.cbr, _tbm.positions, _tbm.N)

# the model carries more bands than we fit: the lowest `μᴸ` are longitudinal modes, absent
# from the photonic spectrum, which `photonic_fit` pushes toward zero/imaginary frequency
μᵀ = 2                # transverse bands, i.e. the physical bands we fit
μᴸ = tbm.N - μᵀ       # longitudinal bands, penalized rather than fitted
freqs_r = freqs[:, 1:μᵀ]

println("model: $(length(tbm)) terms, $(tbm.N) bands ($μᴸ longitudinal, $μᵀ transverse)")
println("reference: $(size(freqs_r, 2)) bands × $(length(kpi)) k-points")

## ---------------------------------------------------------------------------------------- #
# fit the model to the reference bands

# `longitudinal_weight` sets how hard the longitudinal bands are pushed below zero energy.
# The penalty is one-sided (`λ·max(0, E)²`), so it only pushes them *toward* zero and larger
# weights buy a lower longitudinal band at the cost of a slightly worse transverse fit
ptbm = photonic_fit(tbm, freqs_r, kpi; max_multistarts = 25, verbose = true)

freqs_fit = spectrum(ptbm, kpi; transform = energy2frequency)
freqs_fitᵀ = freqs_fit[:, (μᴸ+1):end] # drop the longitudinal bands
Eᴸ_max = maximum(spectrum(ptbm, kpi)[:, 1:μᴸ])

println("\ntransverse frequency RMS error: ", sqrt(mean(abs2, freqs_fitᵀ - freqs_r)), " c/a")
println("largest longitudinal energy:    ", Eᴸ_max, " (≤ 0 desired)")

## ---------------------------------------------------------------------------------------- #
# plot the fit against the reference

plot(
    kpi,
    freqs[:, 1:μᵀ],
    freqs_fitᵀ;
    color = [:black, :red],
    linewidth = [3, 2],
    linestyle = [:solid, :dash],
    ylabel = "Frequency (c/a)",
)
