__precompile__(true)

module Transformations

using Reexport
@reexport using LearnBase
using RecipesBase

import CatViews: CatView
import Base: rand
import LearnBase: transform, transform!, grad, grad!, addgrad!
import StatsBase: logistic, logit

export
    Node,
    link_nodes!,
    Affine,
    Activation

function zero!{T,N}(v::AbstractArray{T,N})
    for i in eachindex(v)
        v[i] = zero(T)
    end
end


# notes:
#   Transformations will be updated in a forward (transform) and backward (grad) pass.
#   The input nodes of a larger comp graph will be called with `transform!(t,input)` and all other
#   nodes will be called with `transform!(t)`, assuming they are properly "linked" beforehand.
#   This is because `output` is computed, which shares a reference to the arrays in the following
#   Transformation's `input`, so it's ready to compute with `transform!`.

#   The same happens in reverse.  An `input` node's `∇` is the same array as the child node's `output.∇`,
#   so the gradients can flow backwards with one call to `grad!` in the proper (reverse) order.
#   In this case, the

# TODO:
#   - implement some basic activations: logistic (sigmoid), tanh, relu
#   - simple Chainer to connect transformations (graphs can be later: general DAGs at first, then allow cycles)


# ----------------------------------------------------------------

# Most Transformations can share these methods as long as they have the required fields.


# Copy input values into the input node, then transform
function transform!(t::Transformation, input::AbstractVector)
    copy!(t.input.val, input)
    transform!(t)
end

# Copy the gradient into the output node, and propagate it back.
function grad!(t::Transformation, ∇out::AbstractVector)
    copy!(t.output.∇, ∇out)
    grad!(t)
end

# return a CatView of the param gradients
function grad(t::Transformation)
    t.∇θ
end

# update our params
# TODO: handle learning rate better
function addgrad!(t::Transformation, dθ::AbstractVector, η::Number)
    for (i,j) in zip(eachindex(t.θ), eachindex(dθ))
        t.θ[i] += η * dθ[j]
    end
end


# ----------------------------------------------------------------

include("nodes.jl")
include("affine.jl")
include("activations.jl")

# ----------------------------------------------------------------

end # module
