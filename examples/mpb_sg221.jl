using Pkg
Pkg.activate(@__DIR__)

### necessary packages
using PhotonicTightBinding
using Crystalline # to access space group information, such as irreps, band reps...
using Brillouin # to obtain k-point paths in the Brillouin zone
using GLMakie # for plotting
using PythonCall: pylist, pyconvert
using ProgressMeter: @showprogress

# ---------------------------------------------------------------------------------------- #
# construct the structure under study

# this example consist of a set of cilinders in the x,y,z directions. This structure has a 
# set of symmetries that correspond to the space group 221 (Pm-3m).

sgnum, D = 221, 3 # space group number and dimension
Rs = directbasis(sgnum, Val(3))

R1 = 0.2 #cylinder radius
N_BANDS = 6 # number of bands to compute
mat = mp.Medium(; epsilon = 12)

# either circular rods, or square-cross-section rods of equal filling fraction
geometry_type = :cylinder # or `:block`
geometry = if geometry_type == :cylinder
    map([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) do axis
        mp.Cylinder(; radius = R1, center = [0, 0, 0], axis = axis, height = 1, material = mat)
    end
elseif geometry_type == :block
    block_size = [R1*√π, R1*√π, 1e20]
    [
        mp.Block(center=[0,0,0], material=mat, size=block_size, e1 = [1,0,0], e2 = [0,1,0], e3 = [0,0,1]),
        mp.Block(center=[0,0,0], material=mat, size=block_size, e1 = [0,1,0], e2 = [0,0,1], e3 = [1,0,0]),
        mp.Block(center=[0,0,0], material=mat, size=block_size, e1 = [0,0,1], e2 = [1,0,0], e3 = [0,1,0]),
    ]
else
    error(lazy"unknown `geometry_type = :$geometry_type`; must be `:cylinder` or `:block`")
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
brs = primitivize(bandreps(sgnum, Val(D))) # already primitive in SG 221, but necessary more generally
symvecs, symeigsv = obtain_symmetry_vectors(ms, brs);

nᵀ = symvecs[1] # pick the 2 lower bands which we are going to study
μᵀ = nᵀ.occupation # number of transverse bands

# obtain an EBR decomposition for the set of bands considered
candidatesv = find_bandrep_decompositions(nᵀ, brs)

##-----------------------------------------------------------------------------------------#
# make a TB model out of one of the solutions obtained

cbr = candidatesv[1].apolarv[1] # take one of the possible solutions
cbrᴸ = candidatesv[1].longitudinal
μᵀ⁺ᴸ = occupation(cbr) # number of apolar (longitudinal + transverse) modes
μᴸ = occupation(cbrᴸ)  # number of longitudinal modes
μᵀ = μᵀ⁺ᴸ - μᴸ         # number of transverse modes

# if we only take intra-cell hoppings, the fitting will not converge
tbm = tb_hamiltonian(cbr, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]);

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
ms.init_params(p = mp.NO_PARITY, reset_fields = true)

# solve across k-points
freqs = Matrix{Float64}(undef, length(kvs), pyconvert(Int, ms.num_bands))
@showprogress 0.1 for (i, kv) in enumerate(kvs)
    redirect_stdout(devnull) do
        ms.solve_kpoint(mp.Vector3(kv...))
    end
    freqs[i,:] = sort!(pyconvert(Vector{Float64}, ms.get_freqs()))
end

# plot the bands of the original system
plot(
    kvs, freqs;
    linewidth = 3, ylabel = "Frequency (c/a)",
    annotations = collect_irrep_annotations(symeigsv, nᵀ.lgirsv),
)

ptbm_fit = photonic_fit(
    tbm, freqs[:, 1:μᵀ], kvs; # fit only the bands that are considered
    max_multistarts=15, verbose = true, lasso=1e-3
)
freqs_fit = spectrum(ptbm_fit, kvs; transform = energy2frequency)[:, μᴸ+1:end] # remove the longitudinal bands

# ---------------------------------------------------------------------------------------- #
# plot fitting results
plot(
    kvs,
    freqs,
    freqs_fit;
    color = [:blue, :red],
    linewidth = [3, 2],
    linestyle = [:solid, :dash],
    ylabel = "Frequency (c/a)",
)
