__precompile__(true)

module Transformations

using Reexport
@reexport using LearnBase
using RecipesBase

import CatViews: CatView
import Base: rand
import LearnBase: transform, transform!, learn, learn!
import StatsBase: logistic, logit

function zero!{T,N}(v::AbstractArray{T,N})
    for i in eachindex(v)
        v[i] = zero(T)
    end
end

# ----------------------------------------------------------------

# part of a subgraph, representing input, output, or learnable parameters
type Node{TYPE,T,N}
    val::Array{T,N}  # value
    ∇::Array{T,N}  # gradient
end

function Node{T,N}(nodetype::Symbol, val::Array{T,N})
    Node{nodetype,T,N}(val, zeros(val))
end

Base.show{TYPE}(io::IO, node::Node{TYPE}) = print(io, "$TYPE$(size(node.val))")

# two nodes can be "linked together", which means that they are the "same node"
# from the perspective of the computational graph, even though one is an output
# of a transformation(s) and the other is the input to a transformation(s).
# this reduces memory requirements and unnecessary copying
function link_nodes!{T,N}(outnode::Node{:output,T,N}, innode::Node{:input,T,N})
    innode.val = outnode.val
    innode.∇ = outnode.∇
end

# Most Transformations can share this.
# Copy the gradient into the output node, and propagate it back.
function grad!(t::Transformation, ∇out::AbstractVector)
    copy!(t.output.∇, ∇out)
    grad!(t)
end


# Most Transformations can share this.
# Copy input values into the input node, then transform
function transform!(t::Transformation, input::AbstractVector)
    copy!(t.input.val, input)
    transform!(t)
end

# ----------------------------------------------------------------

# notes:
#   Transformations will be updated in a forward (transform) and backward (grad) pass.
#   The input nodes of a larger comp graph will be called with `transform!(t,input)` and all other
#   nodes will be called with `transform!(t)`, assuming they are properly "linked" beforehand.
#   This is because `output` is computed, which shares a reference to the arrays in the following
#   Transformation's `input`, so it's ready to compute with `transform!`.

#   The same happens in reverse.  An `input` node's `∇` is the same array as the child node's `output.∇`,
#   so the gradients can flow backwards with one call to `grad!` in the proper (reverse) order.
#   In this case, the

# output = wx + b
immutable Affine{T} <: Transformation
    nin::Int
    nout::Int
    input::Node{:input,T,1}
    w::Node{:param,T,2}
    b::Node{:param,T,1}
    output::Node{:output,T,1}
    θ::CatView{2,T}
    ∇θ::CatView{2,T}
end

function Affine{T}(::Type{T}, nin::Int, nout::Int)
    input = Node(:input, zeros(T, nin)),
    w = Node(:param, zeros(T, nout, nin)),
    b = Node(:param, zeros(T, nin)),
    output = Node(:output, zeros(T, nin))
    Affine(nin, nout, input, w, b, output, CatView(w.val, b.val), CatView(w.∇, b.∇)))
end

Base.show(io::IO, t::Affine) = print(io, "Affine{$(t.nin)-->$(t.nout), input=$(t.input), w=$(t.w), b=$(t.b), output=$(t.output)}")

# compute output = wx + b
function transform!(aff::Affine)
    copy!(aff.output.val, aff.b.val)
    for i=1:aff.nout
        aff.output.val[i] += sum(aff.w.val[i,j] * input[j] for j=1:aff.nin)
    end
    aff.output.val
end

# update the partial derivatives:
#   ∇x = ∂L/∂x
#   ∇w = ∂L/∂w
#   ∇b = ∂L/∂b
# use the chain rule, assuming that we've already updated ∇out = ∂L/∂y
function grad!(aff::Affine)
    # ∇x, ∇w
    for i=1:aff.nin
        ∇xᵢ = zero(eltype(aff.input.∇))
        for o=1:aff.nout
            ∇xᵢ += aff.w.val[o,i] * aff.output.∇[o]
            aff.w.∇[o,i] = aff.input.val[i] * aff.output.∇[o]
        end
        aff.input.∇[i] = ∇xᵢ
    end

    # ∇b
    copy!(aff.b.∇, aff.output.val)
    return grad(aff)
end


# return a CatView of the param gradients
grad(aff::Affine) = aff.∇θ


# ----------------------------------------------------------------

# TODO:
#   - tests for Affine
#   - implement some basic activations: sigmoid (logistic), tanh, relu
#   - simple Chainer to connect transformations (graphs can be later: general DAGs at first, then allow cycles)

# general formula for elementwise activation: output = f(input)
# some examples of f: sigmoid, tanh, relu, etc
immutable Activation{F,T} <: Transformation
    input::Node{:input,T,1}
    output::Node{:output,T,1}
end

function transform!(f::Activation{:sigmoid})
    # TODO
end

function grad!(f::Activation{:sigmoid})
    # TODO
end

# ----------------------------------------------------------------

# export
#     logistic,
#     logit,
#     output_shape

# #
# function transform(t::Transformation, input)
#     output = Array(eltype(input), output_shape(t, input))
#     transform!(output, tfm, input)
# end
#
# "For a Transformation t, the dimensions of the output"
# function output_shape end


end # module
