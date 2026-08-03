# Photonic band fitting, delegating all generic machinery to SymmetricTightBinding:
# `TightBindingCache` (up-front hᵢ(k) caching + work arrays for Hamiltonian assembly and
# Feynman–Hellmann gradients) and `multistart_fit`/`make_fit_objective`/`spectralmoments`
# (implemented in its Optim extension, which is guaranteed loaded since both
# SymmetricTightBinding and Optim are dependencies here). Only the genuinely
# photonics-specific parts live in this file:
# - the frequency↔energy convention (the model's energies are squared frequencies, E = ω²);
# - the band-split loss: the model's `μᴸ` lowest (longitudinal) bands are penalized toward
#   non-positive energies via `λ·max(0, E)²`, while the remaining (transverse) bands are
#   fit to the ω²-reference by least squares.

using SymmetricTightBinding: ReciprocalPointLike, energy_gradient_wrt_hopping
using SymmetricTightBinding: TightBindingCache, multistart_fit, make_fit_objective,
                             spectralmoments
using Optim
using LinearAlgebra: eigen!, eigvals!

# photonic loss, its gradient, and its Gauss–Newton Hessian, following the calling
# convention of `SymmetricTightBinding.make_fit_objective`: with sorted model energies `Es`
# split as `Esᴸ = Es[1:μᴸ]` (longitudinal) & `Esᵀ = Es[μᴸ+1:end]` (transverse),
#   F = ∑ₖ [ ∑ₙ (Eₙʳ − Eₙᵀ)² + λ ∑ₗ max(0, Eₗᴸ)² ]  (+ optional LASSO term),
# i.e., least-squares matching of the transverse bands to the reference plus a one-sided
# quadratic penalty pushing longitudinal bands to non-positive energies (imaginary
# frequencies). Both terms are squared residuals, so the Gauss–Newton Hessian is
# `2∑∇E∇Eᵀ + 2λ∑_{E>0}∇E∇Eᵀ`.
function photonic_fgh!(
    F, G, H, cs, cache::TightBindingCache, Em_r, μᴸ::Integer;
    λ::Real = 1, lasso::Union{Nothing,Real} = nothing
)
    isnothing(G) || fill!(G, zero(eltype(G)))
    isnothing(H) || fill!(H, zero(eltype(H)))

    for (κ, Es_r) in enumerate(eachrow(Em_r))
        Hₖ = cache(cs, κ) # assembled into `cache`'s work array (mutated by eigen below)
        if isnothing(G) && isnothing(H)
            # fast-path, avoiding eigenvector eval. if this is a residuals-only calculation
            Es = eigvals!(Hₖ)
            us = nothing
        else
            Es, us = eigen!(Hₖ) # no Bloch phases, deliberately
        end
        Esᴸ = @view Es[1:μᴸ]     # longitudinal bands
        Esᵀ = @view Es[μᴸ+1:end] # regular, transverse bands

        # photonic loss
        if !isnothing(F)
            F += sum(abs2∘splat(-), zip(Es_r, Esᵀ); init = zero(F))   # transverse
            F += λ * sum(E -> max(zero(E), E)^2, Esᴸ; init = zero(F)) # longitudinal
        end

        # loss gradient & Gauss–Newton approx. of Hessian
        if !isnothing(G) || !isnothing(H)
            ∇Es = energy_gradient_wrt_hopping(cache, κ, (Es, us))
            ∇Esᴸ = @view ∇Es[1:μᴸ]
            ∇Esᵀ = @view ∇Es[μᴸ+1:end]
            for (E_r, E, ∇E) in zip(Es_r, Esᵀ, ∇Esᵀ) # transverse
                isnothing(G) || (G .+= (-2 * (E_r - E)) .* ∇E)
                isnothing(H) || (H .+= 2 .* ∇E .* ∇E')
            end
            for (E, ∇E) in zip(Esᴸ, ∇Esᴸ)            # longitudinal
                E > 0 || continue
                isnothing(G) || (G .+= (2λ * E) .* ∇E)
                isnothing(H) || (H .+= (2λ) .* ∇E .* ∇E')
            end
        end
    end

    # lasso penalty term & gradient
    if !isnothing(lasso)
        lasso *= size(Em_r, 1) # rescaling `lasso` weight to ensure relative contributions
                               # of LSE vs LASSO are invariant to number of k-points
        !isnothing(F) && (F += lasso * sum(abs, cs))
        !isnothing(G) && (G .+= lasso .* sign.(cs))
    end

    return F
end

"""
    photonic_fit(tbm::TightBindingModel{D},
                 freqs_r::AbstractMatrix{<:Real},
                 ks::AbstractVector{<:ReciprocalPointLike{D}},
                 kws...)                          --> ParameterizedTightBindingModel{D}

Fit the hopping amplitudes of a tight-binding model `tbm` to the reference frequencies
`freqs_r`, assumed sampled over **k**-points `ks`. `freqs_r[i,n]` denotes the band
frequency at `ks[i]` in band `n` (and bands are assumed energetically sorted).

The model may have more bands than the reference: the `μᴸ = tbm.N - size(freqs_r, 2)`
lowest model bands are treated as longitudinal bands, penalized toward non-positive
energies (i.e., imaginary frequencies) with weight `longitudinal_weight`; the remaining
(transverse) bands are fit to the reference by least squares.

Fitting is performed using a local optimizer (configurable via `optimizer` from Optim.jl),
used as the basis of a moment-seeded, basin-hopping multi-start global optimization — the
machinery of `SymmetricTightBinding.fit` & `SymmetricTightBinding.multistart_fit`, reused
here with the photonic loss. The global search returns early if the mean fit error, per
band and per **k**-point, is less than `atol`.

## Keyword arguments
- `longitudinal_weight` (default, `$DEFAULT_LONGITUDINAL_WEIGHT`): a weighting factor `λ`
  used to scale the loss term from longitudinal bands. Increase to promote longitudinal
  bands having imaginary frequencies (i.e., negative energies).
- `optimizer` (default, `Optim.NewtonTrustRegion()`): a local optimizer from Optim.jl.
  First-order optimizers exploit the analytic (Feynman–Hellmann) gradient of the loss;
  second-order optimizers additionally exploit its Gauss–Newton Hessian (both the
  transverse and the longitudinal loss terms are squared residuals), thereby acting as
  Gauss–Newton (line-search) or Levenberg–Marquardt-like (trust-region) least-squares
  solvers.
- `atol` (default, `1e-3`): threshold for early return, specifying the minimum required
  mean energetic error (averaged over bands and **k**-points).
- `lasso` (default, `nothing`): if set to a positive number, applies a LASSO penalty to the
  hopping amplitudes, encouraging model sparsity (i.e., small hopping amplitudes to
  vanish). Setting to `nothing` disables the LASSO penalty.
- `options` (default, `Optim.Options(g_abstol = 5e-3, f_reltol = 1e-5)`): a
  `Optim.Options(…)` structure of optimization options, used during the local optimization
  of the multi-start search (i.e., low tolerances, suitable for the low precision demands
  of the multi-start search).
- Remaining keyword arguments (`max_multistarts`, `restart_every`, `verbose`, `polish`,
  `init`, …) are forwarded to `SymmetricTightBinding.multistart_fit`; see its docstring and
  `SymmetricTightBinding.fit`'s for their meaning and defaults.

## Notes
The frequencies are provided by the user but the energies are used internally to do the
fitting. The tight-binding model energies (E) are compared to squared frequencies (ω²), so
the provided frequencies are squared before fitting.
"""
function photonic_fit(
    tbm::TightBindingModel{D},
    freqs_r::AbstractMatrix{<:Real},
    ks::AbstractVector{<:ReciprocalPointLike{D}};
    optimizer::Optim.AbstractOptimizer = NewtonTrustRegion(),
    atol::Real = 1e-3, # minimum threshold error, per k-point & per band, averaged over both
    longitudinal_weight::Real = DEFAULT_LONGITUDINAL_WEIGHT,
    lasso::Union{Nothing,Real} = nothing,
    options::Optim.Options = Optim.Options(;
        g_abstol = 5e-3,
        f_reltol = 1e-5,
    ),
    kws..., # remaining kwargs (`max_multistarts`, `restart_every`, `verbose`, `polish`,
            # `init`, …) are forwarded to `SymmetricTightBinding.multistart_fit`
) where D
    # convert frequencies to energies and sort them
    Em_r = freqs_r .^ 2
    sort!(Em_r; dims = 2)

    μᴸ = tbm.N - size(Em_r, 2) # number of longitudinal bands
    μᴸ ≥ 0 || error(lazy"model has fewer bands ($(tbm.N)) than the reference ($(size(Em_r, 2)))")

    λ = longitudinal_weight
    cache = TightBindingCache(tbm, ks) # hᵢ(k) tabulated once, shared by objective & moments
    obj = make_fit_objective(
        (F, G, H, cs) -> photonic_fgh!(F, G, H, cs, cache, Em_r, μᴸ; λ, lasso))
    # moment seeding from the transverse reference alone: the longitudinal bands are absent
    # from `Em_r`, so the trace fit `c₀` & scales are biased slightly high — but since the
    # longitudinal target is merely E ≤ 0, they remain apt seeding heuristics
    moments = spectralmoments(cache, Em_r)
    tol = length(ks) * size(Em_r, 2) * atol^2 # sum of absolute squares tolerance
    best_cs, _ = multistart_fit(obj, moments; optimizer, tol, options, kws...)
    return tbm(best_cs)
end
