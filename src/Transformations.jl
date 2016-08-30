__precompile__(true)

module Transformations

using Reexport
@reexport using LearnBase
using RecipesBase

import Base: rand
import LearnBase: transform, transform!, learn, learn!
import StatsBase: logistic, logit

# ----------------------------------------------------------------

# part of a subgraph
immutable Node{TYPE,T}
    v::T  # value
    δ::T  # sensitivity
    ∇::T  # gradient
end

# y = wx + b
immutable Affine{V,M} <: Transformation
    x::Node{:input, V}
    w::Node{:param, M}
    b::Node{:param, V}
    y::Node{:output, V}
end

function Affine{T}(::Type{T}, nin::Int, nout::Int)
    Affine{T,N}(
        Node{:input}(zeros(T, nin)),
        Node{:param}(zeros(T, nout, nin)),
        Node{:param}(zeros(T, nout)),
        Node{:output}(zeros(T, nout))
    )
end


# ----------------------------------------------------------------

# We will be using the AutoGrad package for automatic differentiation of our
# transform methods



# ----------------------------------------------------------------

# export
#     logistic,
#     logit,
#     output_shape

# #
# function transform(t::Transformation, x)
#     y = Array(eltype(x), output_shape(t, x))
#     transform!(y, tfm, x)
# end
#
# "For a Transformation t, the dimensions of the output"
# function output_shape end


end # module
