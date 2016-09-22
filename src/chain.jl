
# this is a convenience class to connect Transformations in a chain.
# One could represent an ANN like:
#   affine(n1,n2) --> relu(n2) --> affine(n2,n3) --> logistic(n3)
# The output of a chain could be fed into a loss model for backprop
type Chain{T,P<:Params} <: Learnable
    nin::Int
    nout::Int
    input::Node{:input,T}
    output::Node{:output,T}
    ts::Vector{Transformation}
    params::P
end

# function Chain{T}(::Type{T}, nin::Int, nout::Int)
#     input = Node(:input, zeros(T, nin))
#     output = Node(:output, zeros(T, nout))
#     Chain(nin, nout, input, output, Transformation[])
# end
Chain(ts::Transformation...) = Chain(Float64, ts...)

function consolidate_params{T}(::Type{T}, transforms::AbstractVector)
    lens = ntuple(i -> params_length(transforms[i]), length(transforms))
    nparams = sum(lens)
    θ = zeros(T, nparams)
    ∇ = zeros(T, nparams)
    sizes = map(l -> (l,), lens)
    θs = splitview(θ, sizes)[1]
    ∇s = splitview(∇, sizes)[1]
    for (i,t) in enumerate(transforms)
        if params_length(t) > 0
            # first update the values of the new array, then reset the params with this reference
            θs[i][:] = t.params.θ
            ∇s[i][:] = t.params.∇
            reset!(t.params, θs[i], ∇s[i])
            @assert t.params.θ === θs[i]
        end
    end
    Params(θ, ∇)
end

function Chain{T}(::Type{T}, t1::Transformation, ts::Transformation...)
    # chain = Chain{T}(input_length(t1), output_length(isempty(ts) ? t1 : ts[end]))
    # push!(chain, t1)
    # for t in ts
    #     push!(chain, t)
    # end
    transforms = vcat(t1, ts...)

    for (i,t) in enumerate(transforms)
        if i > 1
            link_nodes!(transforms[i-1].output, t.input)
        end
    end
    # lens = ntuple(i -> params_length(transforms[i]), length(transforms))
    # nparams = sum(lens)
    # θ = zeros(T, nparams)
    # ∇ = zeros(T, nparams)
    # sizes = map(l -> (l,), lens)
    # θs = splitview(θ, sizes)[1]
    # ∇s = splitview(∇, sizes)[1]
    # for (i,t) in enumerate(transforms)
    #     if i > 1
    #         link_nodes!(transforms[i-1].output, t.input)
    #     end
    #
    #     if params_length(t) > 0
    #         # first update the values of the new array, then reset the params with this reference
    #         θs[i][:] = t.params.θ
    #         ∇s[i][:] = t.params.∇
    #         reset!(t.params, θs[i], ∇s[i])
    #     end
    # end
    params = consolidate_params(T, transforms)

    nin = input_length(transforms[1])
    nout = output_length(transforms[end])
    # params = Params(θ, ∇)
    chain = Chain(
        nin,
        nout,
        Node(:input, zeros(T, nin)),
        Node(:output, zeros(T, nout)),
        transforms,
        params
        # Params(θ, ∇)
    )
    link_nodes!(transforms[1].input, chain.input)
    link_nodes!(transforms[end].output, chain.output)
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

# ---------------------------------------------------------------------

function nnet(nin::Int, nout::Int, nh = [],
              inner_activation = :tanh,
              final_activation = :identity)
    ns = vcat(nin, nh, nout)
    num_affine = length(ns) - 1
    layers = []
    for i=1:num_affine
        push!(layers, Affine(ns[i], ns[i+1]))
        if inner_activation != :identity && i < num_affine
            push!(layers, Activation(inner_activation, ns[i+1]))
        end
    end
    if final_activation != :identity
        push!(layers, Activation(final_activation, nout))
    end
    Chain(layers...)
end
