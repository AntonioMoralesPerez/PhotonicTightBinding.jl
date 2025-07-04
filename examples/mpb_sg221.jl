using Pkg
Pkg.activate(@__DIR__)

### necessary packages
using PhotonicTightBinding
using Crystalline
using PhotonicTightBinding.PythonCall: pylist, pyconvert
using Brillouin

# ---------------------------------------------------------------------------------------- #
# construct the structure under study

# this example consist of a set of cilinders in the x,y,z directions. This structure has a 
# set of symmetries that correspond to the space group 221 (P4/mmm).

sgnum, D = 221, 3 # space group number and dimension
Rs = directbasis(sgnum, Val(3))

R1 = 0.2 #cylinder radius
N_BANDS = 5 # number of bands to compute
mat = mp.Medium(; epsilon = 12)
geometry = map([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) do axis
    mp.Cylinder(; radius = R1, center = [0, 0, 0], axis = axis, height = 1, material = mat)
end

# solve the system
ms = mpb.ModeSolver(;
    num_bands = N_BANDS,
    geometry_lattice = mp.Lattice(; basis1 = Rs[1], basis2 = Rs[2], basis3 = Rs[3]),
    geometry = pylist(geometry),
    resolution = 16,
)
ms.init_params(; p = mp.ALL, reset_fields = true)

# obtain the symmetry vectors of the bands computed above
symvecs = obtain_symmetry_vectors(ms, sgnum)

nᵀ = symvecs[1] # pick the 2 lower bands which we are going to study
μᵀ = nᵀ.occupation # number of transverse bands

# obtain an EBR decomposition for the set of bands considered
brs = calc_bandreps(sgnum)
candidatesv = find_bandrep_decompositions(nᵀ, brs)

##-----------------------------------------------------------------------------------------#
# make a TB model out of one of the solutions obtained

cbr = candidatesv[1].apolarv[1] # take one of the possible solutions
μᵀ⁺ᴸ = occupation(cbr) # number of apolar modes
μᴸ = μᵀ⁺ᴸ - μᵀ # number of longitudinal modes

# realize that if we only take intra-cell hoppings, the fitting will not converge
tbm = tb_hamiltonian(cbr, [[0, 0, 0], [1, 0, 0]]);

##-----------------------------------------------------------------------------------------#
# fit the TB model to the MPB results

# obtain the spectrum for a set of k-points on the connected standard band path
kp = irrfbz_path(sgnum, Rs)
kvs = interpolate(kp, 40)
ms = mpb.ModeSolver(;
    num_bands = N_BANDS,
    geometry_lattice = mp.Lattice(; basis1 = Rs[1], basis2 = Rs[2], basis3 = Rs[3]),
    geometry = pylist(geometry),
    k_points = pylist(map(k -> mp.Vector3(k...), kvs)),
)
ms.run()
freqs = pyconvert(Matrix{Float64}, ms.all_freqs)

ptbm_fit = photonic_fit(tbm, freqs[:, 1:μᵀ], kvs; verbose = true) # fit only the bands that are considered
freqs_fit = spectrum(ptbm_fit, kvs; transform = energy2frequency)[:, μᴸ+1:end] # remove the longitudinal bands

# ---------------------------------------------------------------------------------------- #
# plot the results

using GLMakie

plot(
    kvs,
    freqs,
    freqs_fit;
    color = [:blue, :red],
    linewidth = [3, 2],
    linestyle = [:solid, :dash],
    ylabel = "Frequency (c/a)",
)