# benchmark/fit_mpb_benchmark.jl
#
# Benchmark / performance indicator for `photonic_fit` (built on SymmetricTightBinding's
# fitting machinery) on the SG 225 inverse-opal MPB reference; cf. examples/mpb_inverse_opal.jl
# for the underlying structure and examples/fit_inverse_opal.jl for the same fit as a plain
# example. [Historical baseline, July 2026: the pre-revision, self-contained `photonic_fit`
# scored loss 2.6e-2 in 240 s at the 50-multistart budget — the revision reached loss 8.0e-3
# in 105 s.]
#
# Requires the stored MPB reference data in examples/data/ (checked in; regenerate with
# benchmark/generate_mpb_data.jl). All rows use identical problems and RNG seeds. Run:
#
#     julia benchmark/fit_mpb_benchmark.jl

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "examples"); io = devnull)

using Crystalline
using PhotonicTightBinding
using LinearAlgebra: norm
using DelimitedFiles, Random, Statistics, Printf

# --- load the stored MPB reference -------------------------------------------------------
const DATA = joinpath(dirname(@__DIR__), "examples", "data")
freqs_path = joinpath(DATA, "mpb_inverse_opal_freqs.csv")
kvs_path   = joinpath(DATA, "mpb_inverse_opal_kvs.csv")
isfile(freqs_path) && isfile(kvs_path) ||
    error("MPB reference data not found in examples/data/: run benchmark/generate_mpb_data.jl once first")
freqs = readdlm(freqs_path, ',', Float64)
kvs   = [collect(kv) for kv in eachrow(readdlm(kvs_path, ',', Float64))]

# --- TB model: (8a|T₂) EBR of SG 225 (as in examples/mpb_inverse_opal.jl) ----------------
sgnum = 225
Rs′ = directbasis(sgnum, Val(3))
Rs  = primitivize(Rs′, centering(sgnum))
Rm  = stack(Rs)
cbrs = calc_bandreps(sgnum, Val(3))
cbr′ = @composite cbrs[12] # (8a|T₂)
_tbm = tb_hamiltonian(cbr′, [[0, 0, 0], [2, 1, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0], [1, 1, 1]])
sort!(_tbm.terms, by = t -> norm(Rm * t.block.h_orbit.representative()))
tbm = TightBindingModel(_tbm.terms[2:31], _tbm.cbr, _tbm.positions, _tbm.N)

const μᵀ = 2 # the two lowest (transverse) bands (cf. the example's symmetry analysis)
const μᴸ = tbm.N - μᵀ
freqs_r = freqs[:, 1:μᵀ]

@printf("model: %d terms, %d bands (%d longitudinal) | reference: %d bands × %d k-points\n\n",
        length(tbm), tbm.N, μᴸ, μᵀ, length(kvs))

# --- benchmark ---------------------------------------------------------------------------
# equal budgets for the head-to-head rows (the fitters' defaults differ); each row reports
# the actual objective value ("loss": transverse MSE + λ·longitudinal penalty, the single
# fair comparison scalar), the transverse frequency RMS (in units of c/a, vs. the MPB
# reference), and the largest longitudinal band energy (≤ 0 desired, i.e., imaginary
# frequency)
const MAX_MULTISTARTS = 50

Em_r = sort(freqs_r .^ 2; dims = 2)
cache = TightBindingCache(tbm, kvs) # for scoring `objective` (re-exported from SymmetricTightBinding)
objective(cs) = PhotonicTightBinding.photonic_fgh!(
    0.0, nothing, nothing, cs, cache, Em_r, μᴸ;
    λ = PhotonicTightBinding.DEFAULT_LONGITUDINAL_WEIGHT)

function report(name, t, ptbm)
    E = spectrum(ptbm, kvs)
    f_fit = sqrt.(max.(E[:, (μᴸ+1):end], 0.0))
    f_rms = sqrt(mean(abs2, f_fit .- freqs_r))
    E_L_max = maximum(E[:, 1:μᴸ])
    @printf("%-28s | %7.1f s | loss %.4e | transv. freq RMS %.3e | max longit. E %+.3e\n",
            name, t, objective(ptbm.cs), f_rms, E_L_max)
end

# NB (July 2026): two landscape-simplification prototypes were tried here and dropped, both
# having failed to improve on plain `photonic_fit` for this inverse-opal problem: k-point
# dropout (tunneling through the loss landscape by fitting subsets of k-points) ties at
# best, the multimodality being broad rather than a matter of narrow walls; and hopping-range
# continuation (fitting short-range terms first, then releasing longer-range ones) is much
# faster but violates the longitudinal constraint, since the short-range submodel cannot push
# the longitudinal bands negative.
fitters = [
    "photonic_fit"             =>
        (tbm, fr, ks) -> photonic_fit(tbm, fr, ks; max_multistarts = MAX_MULTISTARTS),
    "photonic_fit (4× budget)" =>
        (tbm, fr, ks) -> photonic_fit(tbm, fr, ks; max_multistarts = 4MAX_MULTISTARTS),
    "photonic_fit (`:strict`)" =>
        (tbm, fr, ks) -> photonic_fit(tbm, fr, ks; max_multistarts = MAX_MULTISTARTS,
                                      longitudinal_penalty = :strict,
                                      longitudinal_weight = 0.1),
]

for (name, fitter) in fitters
    Random.seed!(1234)
    t = @elapsed ptbm = fitter(tbm, copy(freqs_r), kvs)
    report(name, t, ptbm)
end

# yardstick (CAVEAT): the hand-tuned coefficients from examples/mpb_inverse_opal.jl
# (`cs_best2`, the outcome of several manually-seeded fitting rounds). NB: term ordering in
# the model construction is not stable across Crystalline/SymmetricTightBinding versions,
# so historically saved coefficient vectors may map onto the wrong terms and score
# arbitrarily badly here (observed: loss ~18 vs ~5e-3 for fresh fits) — treat this row as
# a staleness indicator, not as the by-hand fit's historical quality
cs_hand = [-0.052232, 0.07637, -0.085527, 0.059834, 0.03153, 0.02869, -0.063746, -0.02989,
           0.027744, 0.0015414, -0.0011028, 8.3267e-5, 0.00054636, 0.00017783, -0.00063429,
           -0.0011246, 0.00172, 0.00039308, 6.6726e-5, 7.387e-5, 0.00059735, 0.00072184,
           3.0234e-7, 0.00020861, 0.00011531, 0.00011129, -0.00031123, 3.0333e-7,
           -9.1388e-5, -6.8531e-5]
report("hand-tuned (example)", 0.0, tbm(cs_hand))
