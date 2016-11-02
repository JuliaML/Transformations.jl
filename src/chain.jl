
# this is a convenience class to connect Transformations in a chain.
# One could represent an ANN like:
#   affine(n1,n2) --> relu(n2) --> affine(n2,n3) --> logistic(n3)
# The output of a chain could be fed into a loss model for backprop
type Chain{T,P<:Params} <: Learnable
    nin::Int
    nout::Int
    input::SumNode{T,1}
    output::OutputNode{T,1}
    ts::Vector{Transformation}
    params::P
end

Chain(ts::Transformation...) = Chain(Float64, ts...)
Chain{T}(::Type{T}, ts::Transformation...) = Chain(T, convert(Array{Transformation}, collect(ts)))

function Chain{T,TR<:Transformation}(::Type{T}, ts::AbstractVector{TR})
    link_nodes!(ts)
    Chain(
        input_length(ts[1]),
        output_length(ts[end]),
        input_node(ts[1]),
        output_node(ts[end]),
        ts,
        consolidate_params(T, ts)
    )
end

Base.getindex(chain::Chain, i) = chain.ts[i]
Base.length(chain::Chain) = length(chain.ts)
Base.size(chain::Chain) = size(chain.ts)
Base.endof(chain::Chain) = length(chain)

function Base.show{T}(io::IO, chain::Chain{T})
    println(io, "Chain{$T}(")
    for t in chain.ts
        print(io, "   ")
        show(io, t)
        println(io)
    end
    print(io, ") ")
end

# # add a transformation to the end of the pipeline
# function Base.push!(chain::Chain, t::Transformation)
#     # the new input should be linked to:
#     #   - the chain input if it's first, otherwise
#     #   - the last transformation's output
#     link_nodes!(t.input, isempty(chain.ts) ? chain.input : chain.ts[end].output)
#
#     # we always update the chain output to match the output of the last transformation
#     link_nodes!(t.output, chain.output)
#
#     push!(chain.ts, t)
#     chain
# end

# now that it's set up, just go forward to transform, or backwards to backprop

function transform!(chain::Chain)
    for (i,t) in enumerate(chain.ts)
        i > 1 && transform!(input_node(t))
        transform!(t)
    end
    output_value(chain)
end

function grad!(chain::Chain)
    for (i,t) in enumerate(reverse(chain.ts))
        i > 1 && grad!(output_node(t))
        grad!(t)
    end
    output_value(chain)
end

# transform!(chain::Chain) = (foreach(transform!, chain.ts); chain.output.val)
# grad!(chain::Chain) = foreach(grad!, reverse(chain.ts))

function reset_params!{T}(chain::Chain{T}, θ::AbstractVector, ∇::AbstractVector)
    chain.params = consolidate_params(T, chain.ts, θ=θ, ∇=∇)
end
