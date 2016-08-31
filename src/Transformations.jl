__precompile__(true)

module Transformations

using Reexport
@reexport using LearnBase
using RecipesBase

import Base: rand
import LearnBase: transform, transform!, learn, learn!
import StatsBase: logistic, logit

function zero!{T,N}(v::AbstractArray{T,N})
    for i in eachindex(v)
        v[i] = zero(T)
    end
end

# ----------------------------------------------------------------

# part of a subgraph
immutable Node{TYPE,T}
    val::T  # value
    # δ::T  # sensitivity
    ∇::T  # gradient
end

function Node{T}(nodetype::Symbol, val::T) #, δ::T = zeros(val))
    Node{nodetype, T}(val, zeros(val))
end

Base.show{TYPE}(io::IO, node::Node{TYPE}) = print(io, "$TYPE$(size(node.val))")

# y = wx + b
immutable Affine{V,M} <: Transformation
    nin::Int
    nout::Int
    x::Node{:input, V}
    w::Node{:param, M}
    b::Node{:param, V}
    y::Node{:output, V}
end

function Affine{T}(::Type{T}, nin::Int, nout::Int)
    x = Node(:input, zeros(T, nin)),
    w = Node(:param, zeros(T, nout, nin)),
    b = Node(:param, zeros(T, nin)), #, ones(T, nin)), # sensitivity of b is constant
    y = Node(:output, zeros(T, nin)) #, ones(T, nin))
    Affine(nin,nout,x,w,b,y)
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
    return
end

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
