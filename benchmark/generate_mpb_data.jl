# benchmark/generate_mpb_data.jl
#
# Computes & stores the MPB reference data (SG 225 inverse opal; cf.
# examples/mpb_inverse_opal.jl) used by examples/fit_inverse_opal.jl and
# benchmark/fit_mpb_benchmark.jl, so that neither has to recompute it. Run **once**:
#
#     julia benchmark/generate_mpb_data.jl
#
# (takes minutes; requires the MPB/meep Python stack of the examples environment, which is
# reused here). Results are stored as plain CSV in examples/data/.

using Pkg
Pkg.activate(joinpath(dirname(@__DIR__), "examples"); io = devnull)

using Crystalline
using PhotonicTightBinding
using PythonCall: pylist, pyconvert
using Brillouin: irrfbz_path, interpolate
using LinearAlgebra: norm
using ProgressMeter: @showprogress
using DelimitedFiles

### structure under study: inverse opal (air sphere in ε = 13) in SG 225
sgnum = 225
Rs′ = directbasis(sgnum, Val(3))
Rs = primitivize(Rs′, centering(sgnum))

kp  = irrfbz_path(sgnum, Rs′)
kvs = interpolate(kp, 100)

m = mp.Medium(epsilon = 1)
geometry = [mp.Sphere(center = mp.Vector3(0, 0, 0), radius = 1/sqrt(8), material = m)]
lattice = mp.Lattice(basis_size = norm.(Rs), # units relative to conventional unit cell
                     basis1 = Rs[1], basis2 = Rs[2], basis3 = Rs[3])
ms = mpb.ModeSolver(
    num_bands        = 10,
    k_points         = [],
    geometry         = pylist(geometry),
    geometry_lattice = lattice,
    resolution       = 16,
    tolerance        = 1e-6,
    default_material = mp.Medium(epsilon = 13),
)
ms.init_params(p = mp.NO_PARITY, reset_fields = true)

freqs = Matrix{Float64}(undef, length(kvs), pyconvert(Int, ms.num_bands))
@showprogress 0.1 for (i, kv) in enumerate(kvs)
    redirect_stdout(devnull) do
        ms.solve_kpoint(mp.Vector3(kv...))
    end
    freqs[i, :] = sort!(pyconvert(Vector{Float64}, ms.get_freqs()))
end

datadir = joinpath(dirname(@__DIR__), "examples", "data")
mkpath(datadir)
writedlm(joinpath(datadir, "mpb_inverse_opal_kvs.csv"), permutedims(reduce(hcat, collect(kvs))), ',')
writedlm(joinpath(datadir, "mpb_inverse_opal_freqs.csv"), freqs, ',')
println("saved MPB reference to examples/data/ ($(length(kvs)) k-points × $(size(freqs, 2)) bands)")

### second structure: crossed dielectric cylinders (ε = 12) in SG 221; cf. examples/mpb_sg221.jl.
# Much smaller than the inverse opal (1 longitudinal + 2 transverse bands, 14-term model), so
# it serves as the cheap counterpart when scanning fitting options
sgnum221 = 221
Rs221 = directbasis(sgnum221, Val(3))
kvs221 = interpolate(irrfbz_path(sgnum221, Rs221), 40)

mat221 = mp.Medium(; epsilon = 12)
geometry221 = map([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) do axis
    mp.Cylinder(; radius = 0.2, center = [0, 0, 0], axis = axis, height = 1, material = mat221)
end
ms221 = mpb.ModeSolver(;
    num_bands = 6,
    geometry_lattice = mp.Lattice(; basis1 = Rs221[1], basis2 = Rs221[2], basis3 = Rs221[3]),
    geometry = pylist(geometry221),
    k_points = pylist(map(k -> mp.Vector3(k...), kvs221)),
    resolution = 16,
)
ms221.init_params(p = mp.NO_PARITY, reset_fields = true)

freqs221 = Matrix{Float64}(undef, length(kvs221), pyconvert(Int, ms221.num_bands))
@showprogress 0.1 for (i, kv) in enumerate(kvs221)
    redirect_stdout(devnull) do
        ms221.solve_kpoint(mp.Vector3(kv...))
    end
    freqs221[i, :] = sort!(pyconvert(Vector{Float64}, ms221.get_freqs()))
end

writedlm(joinpath(datadir, "mpb_sg221_kvs.csv"), permutedims(reduce(hcat, collect(kvs221))), ',')
writedlm(joinpath(datadir, "mpb_sg221_freqs.csv"), freqs221, ',')
println("saved MPB reference to examples/data/ ($(length(kvs221)) k-points × $(size(freqs221, 2)) bands)")
