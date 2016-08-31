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

# update our params
# TODO: handle learning rate better
function addgrad!(t::Transformation, dθ::AbstractVector, η::Number)
    for (i,j) in zip(eachindex(t.θ), eachindex(dθ))
        t.θ[i] += η * dθ[j]
    end
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

    function Affine(nin::Int, nout::Int)
        input = Node(:input, zeros(T, nin))
        w = Node(:param, zeros(T, nout, nin))
        b = Node(:param, zeros(T, nout))
        output = Node(:output, zeros(T, nout))
        new(nin, nout, input, w, b, output, CatView(w.val, b.val), CatView(w.∇, b.∇))
    end
end


Base.show(io::IO, t::Affine) = print(io, "Affine{$(t.nin)-->$(t.nout), input=$(t.input), w=$(t.w), b=$(t.b), output=$(t.output)}")

# compute output = wx + b
function transform!(aff::Affine)
    copy!(aff.output.val, aff.b.val)
    for o=1:aff.nout
        aff.output.val[o] += sum(aff.w.val[o,i] * aff.input.val[i] for i=1:aff.nin)
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
#   - implement some basic activations: logistic (sigmoid), tanh, relu
#   - simple Chainer to connect transformations (graphs can be later: general DAGs at first, then allow cycles)

# general formula for elementwise activation: output = f(input)
# some examples of f: logistic, tanh, relu, etc
immutable Activation{F,T} <: Transformation
    n::Int
    input::Node{:input,T,1}
    output::Node{:output,T,1}

    # construct a new Activation, and link the nodes if it's the identity
    # function Activation(fsym::Symbol, n::Int, T = Float64)
    function Activation(n::Int)
        input = Node(:input, zeros(T,n))
        output = Node(:output, zeros(T,n))
        if F == :identity
            link_nodes!(output, input)
        end
        new(n, input, output)
        # Activation{fsym,T}(input, output)
    end
end


# identity: nothing to do, since we linked the input to output
transform!(f::Activation{:identity}) = f.output.val
grad!(f::Activation{:identity}) = f.input.∇

# logistic (sigmoid): f(x) = 1 ./ (1 .+ exp.(-x))
logistic′{T<:Number}(x::T) = x * (one(T) - x)
transform!(f::Activation{:logistic}) = map!(logistic, f.output.val, f.input.val)
function grad!(f::Activation{:logistic})
    for i=1:f.n
        f.input.∇[i] = logistic′(f.output.val[i]) * f.output.∇[i]
    end
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
