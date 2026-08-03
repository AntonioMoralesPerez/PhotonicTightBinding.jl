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

kp  = irrfbz_path(sgnum, Rs′)
kvs = interpolate(kp, 100)
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
#nᵀ = sum(symvecs) # pick all "symmetry-resolvable" (grouped) bands
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
tbm = TightBindingModel(_tbm.terms[[2:31...,]], _tbm.cbr, _tbm.positions, _tbm.N)

## --------------------------------------------------------------------------------------- #
# fit the TB model to the MPB results
#cs_init = [-0.0030304572922553335, -0.01861706917293479, 0.030543122533357774, 0.003827217428906263, 0.003786315518442052, 0.0019488157454362223, -0.005967638848001936, 0.007860952156904384, 0.000165776832794544, -0.006495203323597664]
#cs_init = [0.052989, -0.074152, 0.085709, 0.059708, 0.031709, 0.029844, -0.061744, 0.03568, -0.027736, -0.0064168]
cs_init = [-0.055666, 0.067298, -0.086655, 0.060994, 0.032677, 0.02746, -0.063016, -0.034632, 0.032715, 0.0054279, 0.00081203]
cs_init = [-0.052056, 0.077015, -0.085167, 0.060552, 0.031724, 0.028582, -0.062882, -0.030866, 0.027865, 0.0014247, -0.00059435, 0.00034027, 0.00033292, -3.6544e-8, -0.00043924, -0.00080634, 0.0011535]
cs_init = [-0.052232, 0.07637,  -0.085527, 0.059834, 0.03153,  0.02869,  -0.063746, -0.02989,  0.027744, 0.0015414, -0.0011028,  8.3267e-5, 0.00054636,  0.00017783, -0.00063429, -0.0011246, 0.00172, 0.00039308, 6.6726e-5, 7.387e-5, 0.00059735, 0.00072184, 3.0234e-7, 0.00020861, 0.00011531, 0.00011129, -0.00031123, 3.0333e-7, -9.1388e-5, -6.8531e-5]
ptbm_fit = photonic_fit(
    tbm, freqs[:, 1:μᵀ], kvs;
    max_multistarts=100,
    verbose = true, lasso=nothing, #5e-3,
    #init=vcat(cs_init, randn(length(tbm)-length(cs_init)).*0.00),
    options = PhotonicTightBinding.Optim.Options(;
        #g_abstol = 1e-4,
        #f_reltol = 1e-6,
        ),
    polish = true,
)
#cs_best =  [-0.052056, 0.077015, -0.085167, 0.060552, 0.031724, 0.028582, -0.062882, -0.030866, 0.027865, 0.0014247, -0.00059435, 0.00034027, 0.00033292, -3.6544e-8, -0.00043924, -0.00080634, 0.0011535]
cs_best2 =  [-0.052232, 0.07637,  -0.085527, 0.059834, 0.03153,  0.02869,  -0.063746, -0.02989,  0.027744, 0.0015414, -0.0011028,  8.3267e-5, 0.00054636,  0.00017783, -0.00063429, -0.0011246, 0.00172, 0.00039308, 6.6726e-5, 7.387e-5, 0.00059735, 0.00072184, 3.0234e-7, 0.00020861, 0.00011531, 0.00011129, -0.00031123, 3.0333e-7, -9.1388e-5, -6.8531e-5]
#cs_best2[abs.(cs_best2) .< 9.9e-5] .= 0.0
#ptbm_fit = tbm(cs_best2)
display(ptbm_fit)
freqs_plot_fit = spectrum(ptbm_fit, kvs_plot; transform = energy2frequency);
## --------------------------------------------------------------------------------------- #
# plot the results
plot(
    kvs_plot,
    freqs_plot,#[:, 1:μᵀ],
    freqs_plot_fit;
    color = [:blue, :red],
    linewidth = [3, 2],
    linestyle = [:solid, :solid],
)
