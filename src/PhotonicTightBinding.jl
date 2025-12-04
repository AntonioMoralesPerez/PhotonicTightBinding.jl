module PhotonicTightBinding

# --- Dependencies ----------------------------------------------------------------------- #

using Crystalline
using MPBUtils
# TODO: ↓ probably unnecessary dependency
using PhotonicBandConnectivity
using Crystalline: AbstractSymmetryVector, irdim, CompositeBandRep_from_indices, translation
using Crystalline: reduce_translation_to_unitrange, constant, free, isapproxin, orbit
using PythonCall: pynew, pycopy!, pyimport, Py, pyconvert, pylist
using Reexport
@reexport using SymmetricTightBinding
using Optim # for the fitting of the photonic bands

# --- Constants -------------------------------------------------------------------------- #

const DEFAULT_LONGITUDINAL_WEIGHT = 0.1 # for controlling the penalty for extra bands in the loss function

# --- Export meep and mpb ---------------------------------------------------------------- #

export mp, mpb # loaded from MPBUtils
export pylist, pyconvert # for interacting with MPB

# --- Code loading ----------------------------------------------------------------------- #

include("types.jl")
include("show.jl")
include("constrained_nonnegative_expansions.jl")
include("utils.jl")
export find_apolar_modes
export find_auxiliary_modes
export find_bandrep_decompositions
export obtain_symmetry_vectors
export energy2frequency
include("fitting.jl")
export photonic_fit

# ---------------------------------------------------------------------------------------- #
end # module
