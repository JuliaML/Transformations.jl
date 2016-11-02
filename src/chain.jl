
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
    # transforms = vcat(t1, ts...)
    link_nodes!(ts)
    # transforms = Array(Transformation, length(ts)+1)
    # transforms[1] = t1
    # for (i,t) in enumerate(ts)
    #     # if i > 1
    #     link_nodes!(transforms[i].output, t.input)
    #     transforms[i+1] = t
    #     # end
    # end

    params = consolidate_params(T, ts)
    chain = Chain(
        input_length(ts[1]),
        output_length(ts[end]),
        input_node(ts[1]),
        output_node(ts[end]),
        ts,
        params
    )
    # link_nodes!(transforms[1].input, chain.input)
    # link_nodes!(transforms[end].output, chain.output)
    chain
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
transform!(chain::Chain) = (foreach(transform!, chain.ts); chain.output.val)
grad!(chain::Chain) = foreach(grad!, reverse(chain.ts))

function reset_params!{T}(chain::Chain{T}, θ::AbstractVector, ∇::AbstractVector)
    chain.params = consolidate_params(T, chain.ts, θ=θ, ∇=∇)
end

# ---------------------------------------------------------------------

function nnet(nin::Int, nout::Int, nh = [],
              inner_activation = :tanh,
              final_activation = :identity;
              layernorm = true,
              kw...)
    ns = vcat(nin, nh, nout)
    num_affine = length(ns) - 1
    layers = []
    for i=1:num_affine
        # push!(layers, (layernorm ? LayerNorm : Affine)(ns[i], ns[i+1]))
        if layernorm
            push!(layers, Linear(ns[i], ns[i+1]))
            push!(layers, LayerNorm(ns[i+1]; kw...))
        else
            push!(layers, Affine(ns[i], ns[i+1]))
        end
        if inner_activation != :identity && i < num_affine
            push!(layers, Activation(inner_activation, ns[i+1]))
        end
    end
    if final_activation != :identity
        push!(layers, Activation(final_activation, nout))
    end
    Chain(layers...)
end
