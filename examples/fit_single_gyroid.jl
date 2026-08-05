# examples/fit_single_gyroid.jl
#
# Fitting a symmetry-constrained tight-binding model to the two lowest photonic bands of a
# single gyroid (ε = 16 network at 30% filling, space group 214) — the structure of issue #1,
# where the longitudinal bands stubbornly refused to be pushed below zero frequency.
#
# Two things have to go right here, and they are easy to confuse:
#
#  1. **The model must be small.** The six-band model carries *four* longitudinal bands
#     against only two fitted transverse ones, so it is badly under-determined: given enough
#     hopping terms the fit will happily buy a low longitudinal band with transverse bands
#     that oscillate wildly between the reference points. Pruning to 15 well-chosen terms
#     (below) is what actually resolves issue #1 — it takes the transverse RMS from ~5e-2
#     down to ~2e-3 c/a. This matters far more than the choice of penalty.
#  2. **The longitudinal penalty then decides where the longitudinal bands sit.** With the
#     pruned model the default `:hinge` already fits the transverse bands well, but leaves
#     the longitudinal bands at ω ≈ 0.036 c/a: its gradient vanishes as E → 0⁺, so nothing
#     pushes them below zero. `:strict` is linear beyond a width δ, so its restoring gradient
#     does not vanish and E = 0 becomes a genuine minimum; the leakage drops ~20×, to
#     ω ≈ 0.002 c/a, with no loss of transverse accuracy.
#
# Accordingly we fit twice: once with `:hinge`, then again with `:strict` started from that
# result. Seeding the second fit this way matters — `:strict`'s stiff exact-penalty gradient
# distorts the landscape enough that a *cold* `:strict` multi-start lands in a bad basin
# (RMS 3e-2, transverse bands off by 65%) in 2 of 6 seeds even at 100 multi-starts, whereas
# `:hinge` reaches the good basin from the moment seed in 6 of 6 — and four times faster.
#
# Like `fit_inverse_opal.jl`, this example loads a *stored* MPB reference from `data/` and so
# needs no MPB/Python stack; regenerate it with `benchmark/generate_mpb_data.jl`. Expect well
# under a minute for the two fits.

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

sgnum = 214
Rs′ = directbasis(sgnum, Val(3))            # conventional basis
Rs = primitivize(Rs′, centering(sgnum))     # primitive basis (the TB model's setting)
Rm = stack(Rs)

# as in `fit_inverse_opal.jl` we rebuild the k-points from Brillouin rather than reading them
# back, so that plotting gets labelled axes; the stored file is kept as a cross-check
kpi = interpolate(irrfbz_path(sgnum, Rs′), 100)
freqs = readdlm(joinpath(@__DIR__, "data", "mpb_single_gyroid_freqs.csv"), ',', Float64)

kvs_stored = eachrow(readdlm(joinpath(@__DIR__, "data", "mpb_single_gyroid_kvs.csv"), ',', Float64))
@assert length(kpi) == size(freqs, 1) == length(kvs_stored)
@assert all(norm(k - kv) < 1e-10 for (k, kv) in zip(kpi, kvs_stored)) "stored k-points do \
    not match the reconstructed k-path (Brillouin version skew?)"

## ---------------------------------------------------------------------------------------- #
# build the tight-binding model

# the two lowest bands carry the symmetry content [P₂, N₂+N₄, H₃, -Γ₁+Γ₄], for which
# `find_bandrep_decompositions` offers (12c|B₁) = (8b|A₁) ⊕ nᵀ among others: a six-band model
# with four longitudinal bands
cbrs = calc_bandreps(sgnum, Val(3))
cbr = @composite cbrs[6] # (12c|B₁)

# hoppings out to five neighbour shells (33 terms), sorted by physical range — `tb_hamiltonian`
# orders them block-major, not by range
_tbm = tb_hamiltonian(cbr, [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]])
sort!(_tbm.terms, by = t -> norm(Rm * t.block.h_orbit.representative()))
tbm_full = TightBindingModel(_tbm.terms, _tbm.cbr, _tbm.positions, _tbm.N)

# ... and then keep only the terms that actually earn their place. Truncating by range alone
# is too blunt here: what matters is not how far a term reaches but whether the fit needs it,
# and carrying the rest is what produces the oscillating transverse bands of issue #1 — the
# full 33-term model fits to 26% maximum deviation with its longitudinal bands at ω = 0.17,
# against 6% and 0.036 for the 15 terms kept below.
#
# The selection was found by fitting with a `lasso` penalty, dropping the terms whose
# amplitudes it drove to ~0, refitting, and repeating over several fits until it stabilised;
# term 1, an overall on-site shift that merely sets the energy reference, goes first. To redo
# it, start from `keep = collect(2:length(tbm_full))` and iterate
#
#   ptbm = photonic_fit(tbm_full[keep], freqs_r, kpi; max_multistarts = 25, lasso = 1e-3)
#   keep = keep[findall(>(1e-4), abs.(ptbm.cs))]   # the terms worth keeping
#
# Two things are worth knowing before reaching for `lasso` here. Keep it *small*: at 1e-3 the
# procedure converges sensibly, whereas ≥5e-3 deletes terms the model genuinely needs and the
# refitted result is far worse than what it started from. And it earns its keep only while
# the model is still overgrown — it roughly halves the iteration count on the 33-term model,
# but on the pruned one it merely adds bias and costs iterations, so the fits below use none.
# Finally, running the loop mechanically is no substitute for looking at the result: a greedy
# automated version of exactly this converges happily onto supports that fit ~3× worse
tbm = tbm_full[[2, 3, 4, 5, 6, 8, 10, 11, 12, 21, 27, 29, 31, 32, 33]]

μᵀ = 2                # transverse bands, i.e. the physical bands we fit
μᴸ = tbm.N - μᵀ       # longitudinal bands, penalized rather than fitted
freqs_r = freqs[:, 1:μᵀ]

println("model: $(length(tbm)) terms, $(tbm.N) bands ($μᴸ longitudinal, $μᵀ transverse)")

## ---------------------------------------------------------------------------------------- #
# fit the model, first with the default penalty, then with the strict one

# RMS alone hides the failure mode of issue #1 — a few badly overshooting k-points barely move
# it — so we report the largest deviation as well, relative to the reference away from Γ
# (where ω → 0 makes a relative measure meaningless)
function summarize(label, ptbm)
    freqs_fit = spectrum(ptbm, kpi; transform = energy2frequency)
    Δ = freqs_fit[:, (μᴸ+1):end] - freqs_r
    far = freqs_r .> 0.05
    println(label, ": transverse RMS = ", round(sqrt(mean(abs2, Δ)); sigdigits = 3), " c/a, ",
            "max deviation = ", round(100*maximum(abs.(Δ[far]) ./ freqs_r[far]); sigdigits = 3), "%, ",
            "highest longitudinal frequency = ",
            round(maximum(freqs_fit[:, 1:μᴸ]); sigdigits = 3), " c/a (0 desired)")
    return freqs_fit
end

# `atol` is the mean *energy* error (E = ω²) at which the multi-start search returns early.
# The ~2e-3 c/a RMS this model reaches corresponds to |ΔE| ≈ 2ω·Δω ≈ 8e-4, i.e. already
# inside the default 1e-3 — so the search would stop at whichever local minimum crossed the
# line first and hand back a needlessly variable answer. We disable it and let the full
# multi-start budget run; it costs a couple of seconds
ptbm_hinge = photonic_fit(tbm, freqs_r, kpi; max_multistarts = 25, atol = 1e-8)
freqs_hinge = summarize("`:hinge` (default)", ptbm_hinge)

# `:strict`, started from the `:hinge` optimum. Unlike `:hinge` it responds strongly to
# `longitudinal_weight`; λ = 1 suffices here, and raising it much further only trades
# transverse accuracy away for longitudinal leakage we no longer care about
ptbm_strict = photonic_fit(tbm, freqs_r, kpi; max_multistarts = 25, atol = 1e-8,
                           init = ptbm_hinge.cs,
                           longitudinal_penalty = :strict, longitudinal_weight = 1,
                           longitudinal_width = 1e-2)
freqs_strict = summarize("`:strict`         ", ptbm_strict)

# `ptbm_strict` is the end product: a 15-term, 6×6 parameterized model of the gyroid's two
# lowest bands, from which band structure, group velocities, Berry curvature etc. follow
display(ptbm_strict)

## ---------------------------------------------------------------------------------------- #
# plot both fits against the reference; the longitudinal bands are shown as well, since where
# they end up is the whole point here

for (title, freqs_fit) in (("hinge", freqs_hinge), ("strict", freqs_strict))
    display(plot(
        kpi,
        freqs[:, 1:μᵀ],                 # black: MPB reference
        freqs_fit[:, (μᴸ+1):end],       # red:   fitted transverse bands
        freqs_fit[:, 1:μᴸ];             # blue:  longitudinal bands (should sit at 0)
        color = [:black, :red, :royalblue],
        linewidth = [3, 2, 2],
        linestyle = [:solid, :dash, :solid],
        ylabel = "Frequency (c/a) — $title",
        ylims = (-0.01, 0.35),
    ))
end
