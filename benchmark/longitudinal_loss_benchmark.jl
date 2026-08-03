# benchmark/longitudinal_loss_benchmark.jl
#
# Multi-seed (λ, δ) scan for the longitudinal-band penalty ψ of `photonic_fit`. Every setting
# is run over `SEEDS` and reported as median [min–max]: a single-seed version of this scan
# proved actively misleading, its apparent accuracy differences lying well within the
# run-to-run scatter of the multi-start search (the leakage differences, by contrast, are
# real — they span orders of magnitude).
#
# ## What the scan is for
#
# - Is the incumbent λ well placed for the `:hinge` penalty?
# - For `:strict`, where should the smoothing width δ sit? Theory says the two axes trade
#   off: inside the quadratic core ψ′ = E/δ, so the equilibrium leakage is E* = gδ/λ (g
#   being the competing transverse gradient) — linear in δ and inversely proportional to λ —
#   while a *smaller* δ makes the penalty less quadratic and its curvature ψ″ = 1/δ stiffer,
#   which should cost trust-region iterations. Exactness additionally needs λ ≳ g, or the
#   band never reaches the core at all.
#
# `calls`/`∇calls` (objective evaluations, and those also requesting a gradient/Hessian)
# quantify that cost directly, independently of machine load; they are collected through
# `photonic_fit`'s `objective_callback` hook.
#
# Requires the stored MPB references in examples/data/. Run:
#
#     julia benchmark/longitudinal_loss_benchmark.jl

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "examples"); io = devnull)

using Crystalline
using PhotonicTightBinding
using Brillouin: irrfbz_path, interpolate
using LinearAlgebra: norm
using DelimitedFiles, Random, Statistics, Printf

const DATA = joinpath(dirname(@__DIR__), "examples", "data")
const MAX_MULTISTARTS = 10
const SEEDS = (1234, 1, 2, 3, 4, 5, 6, 7, 8, 9)

readcsv(name) = readdlm(joinpath(DATA, name), ',', Float64)

# --- test cases ---------------------------------------------------------------------------

function inverse_opal_case()
    sgnum = 225
    Rs′ = directbasis(sgnum, Val(3))
    Rm = stack(primitivize(Rs′, centering(sgnum)))
    freqs = readcsv("mpb_inverse_opal_freqs.csv")
    kvs = [collect(kv) for kv in eachrow(readcsv("mpb_inverse_opal_kvs.csv"))]
    cbrs = calc_bandreps(sgnum, Val(3))
    cbr = @composite cbrs[12]
    _tbm = tb_hamiltonian(cbr, [[0,0,0], [2,1,0], [1,1,0], [1,0,0], [2,0,0], [1,1,1]])
    sort!(_tbm.terms, by = t -> norm(Rm * t.block.h_orbit.representative()))
    tbm = TightBindingModel(_tbm.terms[2:31], _tbm.cbr, _tbm.positions, _tbm.N)
    return ("inverse opal (SG 225)", tbm, freqs[:, 1:2], kvs, tbm.N - 2)
end

function sg221_case()
    freqs = readcsv("mpb_sg221_freqs.csv")
    kvs = [collect(kv) for kv in eachrow(readcsv("mpb_sg221_kvs.csv"))]
    brs = primitivize(calc_bandreps(221, Val(3)))
    cbr = @composite brs[15]
    tbm = tb_hamiltonian(cbr, [[0,0,0], [1,0,0], [1,1,0], [1,1,1], [2,0,0]])
    return ("crossed cylinders (SG 221)", tbm, freqs[:, 1:2], kvs, tbm.N - 2)
end

# --- instrumented fit ----------------------------------------------------------------------

function counted_fit(tbm, freqs_r, ks; λ, penalty, δ_rel)
    calls, ∇calls = Ref(0), Ref(0)
    ptbm = photonic_fit(tbm, freqs_r, ks;
                        max_multistarts = MAX_MULTISTARTS,
                        longitudinal_weight = λ,
                        longitudinal_penalty = penalty,
                        longitudinal_width = δ_rel,
                        objective_callback = (F, G, H, cs) -> begin
                            calls[] += 1
                            (isnothing(G) && isnothing(H)) || (∇calls[] += 1)
                        end)
    return ptbm, calls[], ∇calls[]
end

# --- settings under test ------------------------------------------------------------------
# (penalty, λ, δ_rel); δ_rel is ignored by `:hinge`/`:zero`, which take no smoothing width
const CONFIGS = [
    (:hinge,  0.05, 1e-3),  # ┐
    (:hinge,  0.15, 1e-3),  # ├ is the default λ = 0.15 well placed?
    (:hinge,  0.50, 1e-3),  # ┘
    (:zero,   5.0,  1e-3),  # the two-sided variant, for the record
    (:strict,  0.1,  1e-3),  # ┐
    (:strict,  0.1,  1e-2),  # ├ δ scan at fixed λ: leakage ∝ δ vs. iteration cost
    (:strict,  0.1,  1e-1),  # ┘
    (:strict,  1.0,  1e-2),  # ┐ λ scan at fixed δ: exactness needs λ ≳ g
    (:strict,  0.5,  1e-2),  # ┘
]

fmt(xs, f) = @sprintf("%.2e [%.2e–%.2e]", median(f.(xs)), minimum(f.(xs)), maximum(f.(xs)))

for (name, tbm, freqs_r, kvs, μᴸ) in (sg221_case(), inverse_opal_case())
    @printf("\n=== %s: %d terms, %d bands (%d longitudinal), %d k-points, %d seeds ===\n",
            name, length(tbm), tbm.N, μᴸ, length(kvs), length(SEEDS))
    @printf("%-10s %6s %7s | %-26s | %-26s | %8s | %9s\n",
            "penalty", "λ", "δ_rel", "transv. RMS  med [min–max]", "max Eᴸ  med [min–max]",
            "time (s)", "calls")
    for (penalty, λ, δ_rel) in CONFIGS
        results = map(SEEDS) do seed
            Random.seed!(seed)
            t = @elapsed ((ptbm, calls, ∇calls) =
                counted_fit(tbm, copy(freqs_r), kvs; λ, penalty, δ_rel))
            E = spectrum(ptbm, kvs)
            f_fit = sqrt.(max.(E[:, (μᴸ+1):end], 0.0))
            (rms = sqrt(mean(abs2, f_fit .- freqs_r)), Eᴸ = maximum(E[:, 1:μᴸ]),
             t = t, calls = calls)
        end
        @printf("%-10s %6.2f %7s | %-26s | %-26s | %8.1f | %9d\n",
                penalty, λ, penalty === :strict ? string(δ_rel) : "–",
                fmt(results, r -> r.rms), fmt(results, r -> r.Eᴸ),
                median(r.t for r in results), round(Int, median(r.calls for r in results)))
    end
end
