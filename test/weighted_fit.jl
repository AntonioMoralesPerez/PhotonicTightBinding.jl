using PhotonicTightBinding, Test
using Crystalline
using Random: seed!
using PhotonicTightBinding: TightBindingCache, photonic_fgh!

# Per-entry weighting of the transverse loss (`weights`), on a small model with a synthetic
# reference. The loss checks go through the `fgh!` closure directly, so that they are exact
# and deterministic; only the sorting/alignment check needs a full `photonic_fit`.
@testset "photonic fit: per-entry weights" begin
    cbrs = calc_bandreps(2, Val(2))
    cbr = @composite cbrs[3] + cbrs[5] + cbrs[7] # (1c|A) + (1b|A) + (1a|A)
    tbm = tb_hamiltonian(cbr, [[0, 0], [1, 0], [0, 1]])
    @test tbm.N == 3

    kvs = [[0.0, 0.0], [0.1, 0.2], [0.3, -0.1], [0.25, 0.25], [-0.2, 0.4]]
    cache = TightBindingCache(tbm, kvs)

    μᵀ = 2
    μᴸ = tbm.N - μᵀ
    seed!(1234)
    Em_r = sort!(abs.(randn(length(kvs), μᵀ)); dims = 2)
    css = [randn(length(tbm)) for _ in 1:5]

    fgh(cs, weights = nothing) =
        let G = zeros(length(cs)), H = zeros(length(cs), length(cs))
            F = photonic_fgh!(0.0, G, H, cs, cache, Em_r, μᴸ; weights)
            (F, G, H)
        end

    # the default `weights = nothing` reproduces the plain least-squares loss, recomputed
    # here from the model spectrum (`photonic_fgh!` defaults: λ = 1, `:hinge` penalty),
    # and unit weights match it exactly — `nothing` and `ones` are distinct code paths
    for cs in css
        F, G, H = fgh(cs)
        Es = spectrum(tbm(cs), kvs)
        @test F ≈ sum(abs2, Es[:, (μᴸ+1):end] .- Em_r) +
                  sum(E -> max(E, 0)^2, Es[:, 1:μᴸ])
        Fw, Gw, Hw = fgh(cs, ones(size(Em_r)))
        @test Fw == F
        @test Gw == G
        @test Hw == H
    end

    # F, G & H are exactly linear in `weights`: the longitudinal contribution carries no
    # weight, so it cancels from both sides and only the transverse scaling is under test
    w₁ = [0.0 2.5; 1.0 0.0; 0.3 4.0; 1.7 0.9; 0.0 0.0]
    w₂ = [1.2 0.0; 0.5 3.1; 0.0 0.0; 2.0 0.4; 0.8 1.9]
    w₀ = zeros(size(Em_r))
    for cs in css
        F₊, G₊, H₊ = fgh(cs, w₁ .+ w₂)
        F₁, G₁, H₁ = fgh(cs, w₁)
        F₂, G₂, H₂ = fgh(cs, w₂)
        F₀, G₀, H₀ = fgh(cs, w₀)
        @test F₊ - F₁ ≈ F₂ - F₀
        @test G₊ - G₁ ≈ G₂ - G₀
        @test H₊ - H₁ ≈ H₂ - H₀
    end

    # zero-weight reference points contribute nothing: perturbing them is invisible
    mask = ones(size(Em_r))
    mask[1, 2] = mask[3, 2] = mask[4, 1] = 0
    Em_r′ = copy(Em_r)
    Em_r′[1, 2] += 0.7; Em_r′[3, 2] -= 0.4; Em_r′[4, 1] += 1.3
    for cs in css
        F, G, _ = fgh(cs, mask)
        G′ = zeros(length(cs))
        F′ = photonic_fgh!(0.0, G′, nothing, cs, cache, Em_r′, μᴸ; weights = mask)
        @test F′ == F
        @test G′ == G
    end

    # finite-difference check of the gradient under non-trivial weights
    weights = w₁
    for cs in css
        _, G, _ = fgh(cs, weights)
        h = 1e-6
        G_fd = map(eachindex(cs)) do i
            cs⁺ = copy(cs); cs⁺[i] += h
            cs⁻ = copy(cs); cs⁻[i] -= h
            (photonic_fgh!(0.0, nothing, nothing, cs⁺, cache, Em_r, μᴸ; weights) -
             photonic_fgh!(0.0, nothing, nothing, cs⁻, cache, Em_r, μᴸ; weights)) / 2h
        end
        @test G ≈ G_fd rtol=1e-5
    end

    # weights are attached to the bands of `freqs_r` as given, so the row-wise energetic
    # sorting of the reference must permute them identically: fitting an unsorted reference
    # under its mask must agree with fitting the pre-sorted reference under the mask
    # permuted alike. `max_multistarts = 1` with an explicit `init` keeps both fits
    # deterministic (no random starts are drawn)
    freqs_r = [0.30 0.10; 0.15 0.25; 0.40 0.35; 0.05 0.20; 0.45 0.28]
    mask = [0.0 1.0; 1.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 0.0]
    p = sortperm.(eachrow(freqs_r))
    freqs_s = stack(getindex.(eachrow(freqs_r), p); dims = 1)
    mask_s = stack(getindex.(eachrow(mask), p); dims = 1)
    @test mask != mask_s # else the comparison below is vacuous
    ptbm = photonic_fit(tbm, freqs_r, kvs; weights = mask, init = css[1],
                        max_multistarts = 1)
    ptbm′ = photonic_fit(tbm, freqs_s, kvs; weights = mask_s, init = css[1],
                         max_multistarts = 1)
    @test ptbm.cs ≈ ptbm′.cs

    @test_throws ErrorException photonic_fit(tbm, freqs_r, kvs; weights = ones(5, 3))
end
