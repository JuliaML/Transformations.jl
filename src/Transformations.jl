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


# TODO:
# - add input_size/output_size, tranformation/Transformation to LearnBase
# - add traits to LearnBase?

export
    transformation,
    is_learnable,
    Center,
    Affine


# -------------------------------------------------------


abstract AbstractTransformation{I,O}

# we can wrap any object which implements the interface, and it becomes a transformation
immutable Transformation{T,I,O} <: AbstractTransformation{I,O}
    t::T
end

transformation(x) = Transformation{typeof(x), length(input_size(x)), length(output_size(x))}(x)
Base.show{T,I,O}(io::IO, t::Transformation{T,I,O}) = print(io, "T{$I-->$O}{$(t.t)}")

# just pass it through.  we only need to define `transform!` now
transform(t::Transformation, x)     = (y = zeros(output_size(t.t)); transform!(y, t.t, x))
transform!(y, t::Transformation, x) = transform!(y, t.t, x)

# -------------------------------------------------------

# # the "learnable trait"... are there parameters to fit/learn?
# abstract LearnableTrait
#     immutable Learnable <: LearnableTrait end
#     immutable NotLearnable <: LearnableTrait end

# # by default, it can't be learned
# learnable(x) = NotLearnable()

# are there parameters to be learned in this tranformation?
is_learnable(x) = false

# can we take a partial derivative with respect to those parameters?
is_differentiable(x) = false


# the non-bang version must have an empty constructor
# learn{T}(::Type{T}, args...)        = learn!(transformation(T()), args...)
# learn!(t::Transformation, args...)     = learn!(learnable(t.t), t.t, args...)
# learn!(::NotLearnable, t, args...)  = error() # TODO: maybe just do nothing?
# learn!(::Learnable, t, args...)     = learn!(t, args...)

# build a new object and then learn
learn{T}(::Type{T}, args...)        = learn!(transformation(T()), args...)
learn!(t::Transformation, args...)  = learn!(t.t, args...)

# learn!(t::Transformation, args...)  = learn!(Val{is_learnable(t.t)}, t.t, args...)
# learn!(::Type{Val{false}}, args...)    = error("Called learn but is_learnable($(typeof(t.t))) == false") # TODO: maybe just do nothing?
# learn!(::Type{Val{true}}, args...)     = learn!(args...)

# -------------------------------------------------------

is_stochastic(x) = false

# rand(t::Transformation, args...) = rand(Val{is_stochastic(t.t)}, t.t, args...)
# rand(::Type{Val{false}}, t, args...) = error("Called rand but is_stochastic($(typeof(t.t))) == false")

# # the "randomness trait"... is it static or stochastic?
# abstract RandomnessTrait
#     immutable Static <: RandomnessTrait end
#     immutable Stochastic <: RandomnessTrait end

# # by default, everything is static
# randomness(x) = Static()

# # pass through to RandomnessTrait definition
# rand(t::Transformation, args...)    = rand(randomness(t.t), t.t, args...)
# rand(::Static, t, args...)          = error()
# rand(::Stochastic, t, args...)      = rand(t, args...)


# -------------------------------------------------------
# some sample transformations
# notice how we don't need to inherit from a common type tree?

abstract DemoTransform

# we can set traits for a bunch of types like this.  could also use a Union
learnable(::DemoTransform) = Learnable()

immutable Center{A<:AbstractArray} <: DemoTransform
    mu::A
end

input_size(t::Center)               = size(t.mu)
output_size(t::Center)              = size(t.mu)

transform!(y, t::Center, x)         = (y[:] = x - t.mu)
learn!(t::Center, x)                = (mean!(t.mu, x))  # recomputes for full dataset

immutable Affine{W<:AbstractMatrix, B<:AbstractVector} <: DemoTransform
    w::W
    b::B
end

input_size(t::Affine)               = (size(t.w,2), size(t.b,2))
output_size(t::Affine)              = size(t.b)

transform!(y, t::Affine, x)         = (y[:] = x * t.w + t.b)

# -------------------------------------------------------
# Distributions are generating transformations

import Distributions:
    Distribution,
    Normal,
    Bernoulli,
    NegativeBinomial,
    Poisson

is_stochastic(::Distribution) = true


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

include("algebra.jl")

# include("common.jl")
# include("static.jl")
# include("stochastic.jl")
# include("glt.jl")

# TODO: Need to decide conventions for learned transformations.
# include("learned.jl")
# include("sequence.jl")

end # module
