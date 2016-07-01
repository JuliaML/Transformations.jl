module Transformations

## Dependencies ##

using Reexport
@reexport using LearnBase

import StatsBase:
    # functions
    logistic,
    logit

import Distributions:
    # probability distributions
    Distribution,
    Normal,
    Bernoulli,
    NegativeBinomial,
    Poisson

## Exported types ##

export
    # static transformations
    IdentityTransformation,
    LogTransformation,
    ExpTransformation,
    LogisticTransformation,
    LogitTransformation,
    ScaleTransformation,
    ShiftTransformation,

    # stochastic transformations
    GaussianTransformation,
    PoissonTransformation,
    BernoulliTransformation,
    NegativeBinomialTransformation,
    GeneralizedLinearTransformation

## Exported functions ##
# TODO: should these be defined in LearnBase?
export invert, invert!,
       get_inverse, isinvertible

include("common.jl")
include("static.jl")
include("stochastic.jl")
include("glt.jl")

# TODO: Need to decide conventions for learned transformations.
# include("learned.jl")
# include("sequence.jl")

end # module
