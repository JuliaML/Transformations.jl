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

# ----------------------------------------------------------------

# y = wx + b
type Affine{T} <: Transformation
    nin::Int
    nout::Int
    x::Node{:input,T,1}
    w::Node{:param,T,2}
    b::Node{:param,T,1}
    y::Node{:output,T,1}
    θ::CatView{2,T}
    ∇θ::CatView{2,T}
end

function Affine{T}(::Type{T}, nin::Int, nout::Int)
    x = Node(:input, zeros(T, nin)),
    w = Node(:param, zeros(T, nout, nin)),
    b = Node(:param, zeros(T, nin)),
    y = Node(:output, zeros(T, nin))
    Affine(nin, nout, x, w, b, y, CatView(w.val, b.val), CatView(w.∇, b.∇)))
end

Base.show(io::IO, t::Affine) = print(io, "Affine{$(t.nin)-->$(t.nout), x=$(t.x), w=$(t.w), b=$(t.b), y=$(t.y)}")

# copy x and compute y
function transform!(aff::Affine, x::AbstractVector)
    copy!(aff.x.val, x)
    copy!(aff.y.val, aff.b.val)
    for i=1:aff.nout
        aff.y.val[i] += sum(aff.w.val[i,j] * x[j] for j=1:aff.nin)
    end
end

# update the partial derivatives:
#   ∇x = ∂L/∂x
#   ∇w = ∂L/∂w
#   ∇b = ∂L/∂b
# use the chain rule, assuming that we've already updated ∇y = ∂L/∂y
function grad!(aff::Affine)
    # ∇x, ∇w
    for i=1:aff.nin
        ∇xᵢ = zero(eltype(aff.x.∇))
        for o=1:aff.nout
            ∇xᵢ += aff.w.val[o,i] * aff.y.∇[o]
            aff.w.∇[o,i] = aff.x.val[i] * aff.y.∇[o]
        end
        aff.x.∇[i] = ∇xᵢ
    end

    # ∇b
    copy!(aff.b.∇, aff.y.val)
    return grad(aff)
end

# return a CatView of the param gradients
grad(aff::Affine) = aff.∇θ


# ----------------------------------------------------------------

# TODO:
#   - tests for Affine
#   - implement some basic activations: sigmoid (logistic), tanh, relu
#   - simple Chainer to connect transformations (graphs can be later: general DAGs at first, then allow cycles)


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
