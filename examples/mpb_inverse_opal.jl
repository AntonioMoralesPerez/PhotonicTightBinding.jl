using Pkg
Pkg.activate(@__DIR__)

using Crystalline
using Crystalline: SVector, @SVector
using PhotonicTightBinding
using PythonCall: pylist, pyconvert
using Brillouin: irrfbz_path, interpolate
using LinearAlgebra: norm
using ProgressMeter: @showprogress
using GLMakie

## --------------------------------------------------------------------------------------- #
# construct the structure under study

sgnum = 225 # space group number
Rs′ = directbasis(sgnum, Val(3))
Rs = primitivize(Rs′, centering(sgnum))
Rm = stack(Rs)

kp = irrfbz_path(sgnum, Rs′)

# we fit against randomly sampled k-points rather than a path, so that the fit is not biased
# towards the high-symmetry lines; the high-symmetry points themselves are appended explicitly.
# A conventional path is used for plotting
kvs = [(rand(SVector{3, Float64}) - (@SVector [0.5, 0.5, 0.5])) for i in 1:100]
push!(kvs, SVector{3, Float64}([0.0, 0.0, 0.0]), SVector{3, Float64}([0.5, 0.5, 0.5]), SVector{3, Float64}([0.5, 0.5, 0.0]), SVector{3, Float64}([0.5, 0.0, 0.5]), SVector{3, Float64}([0.0, 0.5, 0.5]))
kvs_plot = interpolate(kp, 100)

# meep geometry
m = mp.Medium(epsilon=1)
geometry = [mp.Sphere(center=mp.Vector3(0,0,0), radius=1/sqrt(8), material=m)]
lattice = mp.Lattice(basis_size=norm.(Rs), # take units relative to conventional unit cell
                     basis1 = Rs[1], basis2 = Rs[2], basis3 = Rs[3])
ms = mpb.ModeSolver(
    num_bands        = 10,
    k_points         = [],
    geometry         = pylist(geometry),
    geometry_lattice = lattice,
    resolution       = 16,
    tolerance        = 1e-6,
    default_material = mp.Medium(epsilon=13),
)
ms.init_params(p = mp.NO_PARITY, reset_fields = true)

freqs = Matrix{Float64}(undef, length(kvs), pyconvert(Int, ms.num_bands))
@showprogress 0.1 for (i, kv) in enumerate(kvs)
    redirect_stdout(devnull) do
        ms.solve_kpoint(mp.Vector3(kv...))
    end
    freqs[i,:] = sort!(pyconvert(Vector{Float64}, ms.get_freqs()))
end

freqs_plot = Matrix{Float64}(undef, length(kvs_plot), pyconvert(Int, ms.num_bands))
@showprogress 0.1 for (i, kv) in enumerate(kvs_plot)
    redirect_stdout(devnull) do
        ms.solve_kpoint(mp.Vector3(kv...))
    end
    freqs_plot[i,:] = sort!(pyconvert(Vector{Float64}, ms.get_freqs()))
end

## --------------------------------------------------------------------------------------- #
# obtain the symmetry vectors of the bands computed above
ms.init_params(; p = mp.ALL, reset_fields = true)

cbrs = calc_bandreps(sgnum, Val(3)) # conventional setting; to pass to `tb_hamiltonian`
brs = primitivize(cbrs)
symvecs, symeigsv = obtain_symmetry_vectors(ms, brs);

nᵀ = symvecs[1] # pick the 2 lower bands which we are going to study
μᵀ = nᵀ.occupation # number of transverse bands

# obtain an EBR decomposition with at least one additional band
μᴸ_min = 1
candidatesv = find_bandrep_decompositions(nᵀ, brs; μᴸ_min)

## --------------------------------------------------------------------------------------- #
# plot the bands of the original system
plot(
    kvs_plot, freqs_plot;
    linewidth = 3, ylabel = "Frequency (c/a)",
    annotations = collect_irrep_annotations(symeigsv, nᵀ.lgirsv),
)

## --------------------------------------------------------------------------------------- #
# make a TB model out of one of the solutions

cbr = candidatesv[1].apolarv[1]
cbrᴸ = candidatesv[1].longitudinal
μᴸ = occupation(cbrᴸ)
μᵀ = occupation(cbr) - μᴸ

# create a parameterized tight-binding model w/ sufficiently many hopping terms
cbr′ = @composite cbrs[12] # picks out (8a|T₂) manually. HAAAACK + FIXME. Needed because `cbr` is in primitive basis
_tbm = tb_hamiltonian(cbr′, [[0, 0, 0], [2,1,0], [1,1,0], [1, 0, 0], [2,0,0], [1,1,1]])
sort!(_tbm.terms, by=t->norm(Rm * t.block.h_orbit.representative()));
tbm = TightBindingModel(_tbm.terms[2:31], _tbm.cbr, _tbm.positions, _tbm.N)

## --------------------------------------------------------------------------------------- #
# fit the TB model to the MPB results
# a cold multi-start fit suffices here — no warm start needed. The `:strict` longitudinal
# penalty is worth the extra iterations: it places the longitudinal bands distinctly closer to
# zero frequency than the default `:hinge`, whose gradient vanishes as E → 0⁺
ptbm_fit = photonic_fit(
    tbm, freqs[:, 1:μᵀ], kvs;
    max_multistarts = 100, verbose = true,
    longitudinal_penalty = :strict,
)
display(ptbm_fit)
freqs_plot_fit = spectrum(ptbm_fit, kvs_plot; transform = energy2frequency);
## --------------------------------------------------------------------------------------- #
# plot the results
plot(
    kvs_plot,
    freqs_plot,
    freqs_plot_fit;
    color = [:blue, :red],
    linewidth = [3, 2],
    linestyle = [:solid, :solid],
)
