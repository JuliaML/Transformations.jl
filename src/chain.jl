
# this is a convenience class to connect Transformations in a chain.
# One could represent an ANN like:
#   affine(n1,n2) --> relu(n2) --> affine(n2,n3) --> logistic(n3)
# The output of a chain could be fed into a loss model for backprop
type Chain{T} <: Learnable
    nin::Int
    nout::Int
    input::Node{:input,T}
    output::Node{:output,T}
    ts::Vector{Transformation}
    params::Params
    # Θ::Vector{T}
    # ∇Θ::Vector{T}

    function Chain(nin::Int, nout::Int)
        input = Node(:input, zeros(T, nin))
        output = Node(:output, zeros(T, nout))
        new(nin, nout, input, output, Transformation[])
    end
end

function Chain{T}(::Type{T}, t1::Transformation, ts::Transformation...)
    chain = Chain{T}(input_length(t1), output_length(isempty(ts) ? t1 : ts[end]))
    push!(chain, t1)
    for t in ts
        push!(chain, t)
    end
    lens = ntuple(i -> params_length(chain.ts[i]), length(ts)+1)
    nparams = sum(lens)
    Θ = zeros(T, nparams)
    ∇Θ = zeros(T, nparams)
    sizes = map(l -> (l,), lens)
    Θs = splitview(Θ, sizes)[1]
    ∇s = splitview(∇Θ, sizes)[1]
    for (i,t) in enumerate(chain.ts)
        if params_length(t) > 0
            reset!(t.params, Θs[i], ∇s[i])
        end
    end
    chain.params = Params(Θ, ∇Θ)
    chain
end

function Base.show{T}(io::IO, chain::Chain{T})
    println(io, "Chain{$T}(")
    for t in chain.ts
        print(io, "   ")
        show(io, t)
        println(io)
    end
    print(io, ") ")
end

# add a transformation to the end of the pipeline
function Base.push!(chain::Chain, t::Transformation)
    # the new input should be linked to:
    #   - the chain input if it's first, otherwise
    #   - the last transformation's output
    link_nodes!(t.input, isempty(chain.ts) ? chain.input : chain.ts[end].output)

    # we always update the chain output to match the output of the last transformation
    link_nodes!(t.output, chain.output)

    push!(chain.ts, t)
    chain
end

# now that it's set up, just go forward to transform, or backwards to backprop
transform!(chain::Chain) = (foreach(transform!, chain.ts); chain.output.val)
grad!(chain::Chain) = foreach(grad!, reverse(chain.ts))

# TODO: should make CatViews θ and ∇θ
