module Transformations

## Dependencies ##

using Reexport
@reexport using LearnBase

import Base: rand
import LearnBase: transform, transform!, learn, learn!
import StatsBase: logistic, logit

# # the "transformable trait"... I and O are the shapes (num dims) of the input and output
# abstract TransformationType{I, O}
#     immutable Transformable{I,O} <: TransformationType{I,O} end
#     immutable NotTransformable{I,O} <: TransformationType{I,O} end

# transformation_type(t) = error()

# -------------------------------------------------------

# we can wrap any object which implements the interface, and it becomes a transformation
immutable Transformation{T,I,O}
    t::T
end

transformation(x) = Transformation{typeof(x), length(input_size(x)), length(output_size(x))}(x)
Base.show{T,I,O}(io::IO, t::Transformation{T,I,O}) = print(io, "T{$I-->$O}($(t.t))")

# -------------------------------------------------------

# the "randomness trait"... is it static or stochastic?
abstract Randomness
    immutable Static <: Randomness end
    immutable Stochastic <: Randomness end

# everything is static, unless we specify
randomness(x) = Static()

rand(t::Transformation, args...) = rand(randomness(t), t, args...)
rand(::Static, args...) = error()
rand(::Stochastic, t, args...) = rand(t, args...)


# -------------------------------------------------------
# some sample transformations

immutable Center{A<:AbstractArray}
    mu::A
end

input_size(t::Center) = size(t.mu)
output_size(t::Center) = size(t.mu)

immutable Affine{W<:AbstractMatrix, B<:AbstractVector}
    w::W
    b::B
end

input_size(t::Affine) = (size(t.w,2), size(t.b,2))
output_size(t::Affine) = size(t.b)
# transformation_type(aff::Affine) = Transformable{1,1}

# -------------------------------------------------------
# Distributions are generating transformations

import Distributions:
    Distribution,
    Normal,
    Bernoulli,
    NegativeBinomial,
    Poisson
    

randomness(d::Distribution) = Stochastic()


## Exported types ##

# export
#     # static transformations
#     IdentityTransformation,
#     LogTransformation,
#     ExpTransformation,
#     LogisticTransformation,
#     LogitTransformation,
#     ScaleTransformation,
#     ShiftTransformation,

#     # stochastic transformations
#     GaussianTransformation,
#     PoissonTransformation,
#     BernoulliTransformation,
#     NegativeBinomialTransformation,
#     GeneralizedLinearTransformation

# ## Exported functions ##
# # TODO: should these be defined in LearnBase?
# export invert, invert!, isinvertible

# include("common.jl")
# include("static.jl")
# include("stochastic.jl")
# include("glt.jl")

# TODO: Need to decide conventions for learned transformations.
# include("learned.jl")
# include("sequence.jl")

end # module
