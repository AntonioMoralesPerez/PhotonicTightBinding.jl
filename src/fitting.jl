# Photonic band fitting, delegating all generic machinery to SymmetricTightBinding:
# `TightBindingCache` (up-front hᵢ(k) caching + work arrays for Hamiltonian assembly and
# Feynman–Hellmann gradients) and `multistart_fit`/`make_fit_objective`/`spectralmoments`
# (implemented in its Optim extension, which is guaranteed loaded since both
# SymmetricTightBinding and Optim are dependencies here). Only the genuinely
# photonics-specific parts live in this file:
# - the frequency↔energy convention (the model's energies are squared frequencies, E = ω²);
# - the band-split loss: the model's `μᴸ` lowest (longitudinal) bands are penalized toward
#   non-positive energies via `λ·ψ(E)`, while the remaining (transverse) bands are fit to
#   the ω²-reference by least squares. See the penalty variants below for the choice of ψ.

using SymmetricTightBinding: ReciprocalPointLike, energy_gradient_wrt_hopping
using SymmetricTightBinding: TightBindingCache, multistart_fit, make_fit_objective,
                             spectralmoments
using Optim
using LinearAlgebra: eigen!, eigvals!

# --- longitudinal band penalties ---------------------------------------------------------
# The longitudinal bands have no reference data attached: they are instead shaped by a
# penalty ψ(E), whose choice encodes what we believe about them (namely, that they ought to
# sit at zero or imaginary frequency, i.e. E ≤ 0). `_longitudinal_penalty` returns the
# triple `(ψ, ψ′, ψ″)`, with ψ″ supplying the curvature used for the Hessian; `δ` is the
# smoothing width, and is ignored by the variants that do not need it.
#
# `:hinge`   ψ = max(0,E)²  one-sided quadratic; the default. Its gradient vanishes at
#                           E = 0, so nothing pushes a band below zero: the minimum sits at
#                           E* ≈ g/(2λ) > 0 (g being the competing transverse gradient), and
#                           zero is approached as 1/λ but never attained.
# `:strict`  ψ = one-sided, the linear tail exerts a constant restoring gradient, so once
#            linear beyond  λ ≳ g, E = 0 becomes a genuine minimum rather than an asymptote
#            δ, quadratic   (an exact penalty): the residual leakage falls to E* ≈ gδ/λ. The
#            within it      quadratic core keeps ψ twice differentiable for the trust-region
#                           solver — but only within |E| < δ, so the quadratic model is poor
#                           in the linear branch and the solver needs ~4–10× more iterations.
# `:zero`    ψ = E²         two-sided; treats "longitudinal ⇒ ω = 0" as a datum to be fitted
#                           rather than an inequality to be satisfied. It also penalizes the
#                           harmless E < 0, so with several longitudinal bands it
#                           over-constrains the model badly (measured: RMS an order of
#                           magnitude worse on a 4-longitudinal-band model).
#
# Empirically (cf. benchmark/longitudinal_loss_benchmark.jl), E* ≈ gδ/λ is well obeyed, and
# `:strict` lowers the leakage by 2–3 orders of magnitude relative to `:hinge` without
# measurably changing the transverse fit quality either way.
const LONGITUDINAL_PENALTIES = (:hinge, :strict, :zero)

@inline function _longitudinal_penalty(::Val{:hinge}, E::T, δ::T) where T<:Real
    return E > zero(T) ? (E^2, 2E, T(2)) : (zero(T), zero(T), zero(T))
end
@inline function _longitudinal_penalty(::Val{:zero}, E::T, δ::T) where T<:Real
    return (E^2, 2E, T(2))
end
@inline function _longitudinal_penalty(::Val{:strict}, E::T, δ::T) where T<:Real
    if E ≤ zero(T)
        return (zero(T), zero(T), zero(T))
    elseif E < δ
        return (E^2/(2δ), E/δ, inv(δ))
    else
        return (E - δ/2, one(T), zero(T))
    end
end

# photonic loss, its gradient, and its Gauss–Newton Hessian, following the calling
# convention of `SymmetricTightBinding.make_fit_objective`: with sorted model energies `Es`
# split as `Esᴸ = Es[1:μᴸ]` (longitudinal) & `Esᵀ = Es[μᴸ+1:end]` (transverse),
#   F = ∑ₖ [ ∑ₙ (Eₙʳ − Eₙᵀ)² + λ ∑ₗ ψ(Eₗᴸ) ]  (+ optional LASSO term),
# i.e., least-squares matching of the transverse bands to the reference plus a penalty ψ
# pushing the longitudinal bands toward zero/imaginary frequency (see above). The transverse
# term is a squared residual, contributing the Gauss–Newton Hessian `2∑∇E∇Eᵀ`; the
# longitudinal term contributes its exact curvature `λ∑ψ″∇E∇Eᵀ`.
function photonic_fgh!(
    F, G, H, cs, cache::TightBindingCache, Em_r, μᴸ::Integer;
    λ::Real = 1, lasso::Union{Nothing,Real} = nothing,
    penalty::Val = Val(:hinge), δ::Real = 0.0
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
            F += sum(abs2∘splat(-), zip(Es_r, Esᵀ); init = zero(F))          # transverse
            F += λ * sum(E -> _longitudinal_penalty(penalty, E, δ)[1], Esᴸ;
                         init = zero(F))                                     # longitudinal
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
                _, ψ′, ψ″ = _longitudinal_penalty(penalty, E, δ)
                isnothing(G) || iszero(ψ′) || (G .+= (λ * ψ′) .* ∇E)
                isnothing(H) || iszero(ψ″) || (H .+= (λ * ψ″) .* ∇E .* ∇E')
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
  bands having imaginary frequencies (i.e., negative energies) — though with the default
  `:hinge` penalty this has surprisingly little leverage over how far above zero they
  ultimately settle (see `longitudinal_penalty`).
- `longitudinal_penalty` (default, `:hinge`): the shape `ψ` of the penalty applied to the
  longitudinal bands, whose target is `E ≤ 0` (zero or imaginary frequency):
  - `:hinge` (`ψ = max(0,E)²`): a one-sided quadratic. Cheap, but its gradient vanishes at
    `E = 0`, so the longitudinal bands settle slightly *above* zero — typically at a few
    percent of the reference energy scale, and not much improvable by raising `λ`.
  - `:strict` (one-sided; linear beyond a width `δ`, quadratic within it): the linear tail
    makes `E = 0` a genuine minimum rather than an asymptote, lowering the residual
    longitudinal energy by 2–3 orders of magnitude, at ~4–10× more iterations. Worth trying
    if a longitudinal band strays into the transverse manifold, where it can corrupt the
    energy-ordered band assignment.
  - `:zero` (`ψ = E²`): fits the longitudinal bands *to* zero, two-sided. Only advisable
    with a single longitudinal band; with several it over-constrains the model badly.
- `longitudinal_width` (default, `1e-3`): the smoothing width `δ` of the `:strict` penalty,
  relative to the mean reference energy. Leakage scales as `δ`, while smaller `δ` costs
  iterations; loosening it much beyond the default is only safe for easy problems. Unused by
  the other penalties.
- `optimizer` (default, `Optim.NewtonTrustRegion()`): a local optimizer from Optim.jl.
  First-order optimizers exploit the analytic (Feynman–Hellmann) gradient of the loss;
  second-order optimizers additionally exploit its Hessian, which is Gauss–Newton for the
  transverse term (a squared residual) and exact in `ψ` for the longitudinal one, thereby
  acting as Gauss–Newton (line-search) or Levenberg–Marquardt-like (trust-region)
  least-squares solvers. Note that this is a good model of the loss only where `ψ` is itself
  quadratic, which is why `:strict` needs more iterations than `:hinge`.
- `atol` (default, `1e-3`): threshold for early return, specifying the minimum required
  mean energetic error (averaged over bands and **k**-points).
- `lasso` (default, `nothing`): if set to a positive number, applies a LASSO penalty to the
  hopping amplitudes, encouraging model sparsity (i.e., small hopping amplitudes to
  vanish). Setting to `nothing` disables the LASSO penalty.
- `objective_callback` (default, `nothing`): if set to a function, it is called as
  `objective_callback(F, G, H, cs)` immediately before every objective evaluation, with the
  same arguments the objective receives (`G`/`H` are `nothing` when the optimizer requests
  no gradient/Hessian). Intended for instrumentation — e.g. counting evaluations or tracing
  the search — and imposes no cost when left at `nothing`.
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
    longitudinal_penalty::Symbol = :hinge,
    longitudinal_width::Real = 1e-3,
    lasso::Union{Nothing,Real} = nothing,
    objective_callback::Union{Nothing,Function} = nothing,
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
    longitudinal_penalty ∈ LONGITUDINAL_PENALTIES ||
        error(lazy"unknown `longitudinal_penalty = :$longitudinal_penalty`; must be one of $LONGITUDINAL_PENALTIES")

    λ = longitudinal_weight
    # smoothing width of the `:strict` penalty, taken relative to the reference energy scale
    # so that the default ports across structures: small enough that the linear
    # (exact-penalty) branch governs any leakage we would care about, large enough to keep
    # the ψ″ = 1/δ curvature — and hence the iteration count — in hand
    δ = longitudinal_width * (sum(Em_r) / length(Em_r))
    penalty = Val(longitudinal_penalty)
    cache = TightBindingCache(tbm, ks) # hᵢ(k) tabulated once, shared by objective & moments
    obj = make_fit_objective() do F, G, H, cs
        isnothing(objective_callback) || objective_callback(F, G, H, cs)
        photonic_fgh!(F, G, H, cs, cache, Em_r, μᴸ; λ, lasso, penalty, δ)
    end
    # moment seeding from the transverse reference alone: the longitudinal bands are absent
    # from `Em_r`, so the trace fit `c₀` & scales are biased slightly high — but since the
    # longitudinal target is merely E ≤ 0, they remain apt seeding heuristics
    moments = spectralmoments(cache, Em_r)
    tol = length(ks) * size(Em_r, 2) * atol^2 # sum of absolute squares tolerance
    best_cs, _ = multistart_fit(obj, moments; optimizer, tol, options, kws...)
    return tbm(best_cs)
end
