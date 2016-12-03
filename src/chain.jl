
# this is a convenience class to connect Transformations in a chain.
# One could represent an ANN like:
#   affine(n1,n2) --> relu(n2) --> affine(n2,n3) --> logistic(n3)
# The output of a chain could be fed into a loss model for backprop
type Chain{T,P<:Params,PREP<:PreprocessStep} <: Learnable
    nin::Int
    nout::Int
    input::SumNode{T,1}
    output::OutputNode{T,1}
    ts::Vector{Transformation}
    params::P
    prep::PREP
    grad_calc::Symbol
    Bs::Vector{Nullable{Matrix{T}}}  # these are the fixed/random matrices Bᵢ for the DFA method
end

Chain(ts::Transformation...; kw...) = Chain(Float64, ts...; kw...)
Chain{T}(::Type{T}, ts::Transformation...; kw...) = Chain(T, convert(Array{Transformation}, collect(ts)); kw...)

function Chain{T,TR<:Transformation}(::Type{T}, ts::AbstractVector{TR};
                                     prep::PreprocessStep = NoPreprocessing(),
                                     grad_calc::Symbol = :backprop)
    link_nodes!(ts)

    # initialize DFA method
    Bs = Nullable{Matrix{T}}[]
    if grad_calc == :dfa
        for (i,t) in enumerate(ts)
            nΘ = params_length(t)
            Bi = if nΘ == 0 || i >= length(ts)-1
                # don't do anything for non-learnable transformations,
                # or the final transformation (we'll just backprop that one)
                Nullable{Matrix{T}}()
            else
                ny = output_length(t)
                # B = zeros(T, input_length(t), output_length(ts[end]))
                B = zeros(T, output_length(t), output_length(ts[end]))
                initialize_weights!(B)
                fill!(params(t), zero(T))  # reset params to 0... should we keep this?
                Nullable{Matrix{T}}(B)
            end
            push!(Bs, Bi)
        end
    end

    Chain(
        input_length(ts[1]),
        output_length(ts[end]),
        input_node(ts[1]),
        output_node(ts[end]),
        ts,
        consolidate_params(T, ts),
        prep,
        grad_calc,
        Bs
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
    if !isa(chain.prep, NoPreprocessing)
        # update the whitener and transform the input data
        copy!(input_value(chain.prep), input_value(chain))
        learn!(chain.prep)
        copy!(input_value(chain), transform!(chain.prep))
    end
    for (i,t) in enumerate(chain.ts)
        i > 1 && transform!(input_node(t))
        transform!(t)
    end
    output_value(chain)
end

function grad!(chain::Chain)
    if chain.grad_calc == :backprop
        for (i,t) in enumerate(reverse(chain.ts))
            i > 1 && grad!(output_node(t))
            grad!(t)
        end
        input_grad(chain)
    elseif chain.grad_calc == :dfa
        #= going through the list:
            - skip non-learnables
            - final transform uses normal backprop
            - other learnables use DFA: ∇yᵢ = Bᵢ∇yₘ
        =#
        ∇out = output_grad(chain)
        n = length(chain.ts)
        # @show n ∇out
        for (j,t) in enumerate(reverse(chain.ts))
            i = n - j + 1
            # @show i, t
            if isnull(chain.Bs[i])
                # non-learnables get backprop
                i < n && grad!(output_node(t))
                # @show "null" grad!(t)
                grad!(t)
            else
                if i == n
                    # final node gets backprop
                    grad!(t)
                else
                    # other learnables get DFA
                    Bᵢ = get(chain.Bs[i])
                    # @show Bᵢ
                    grad!(t, Bᵢ * ∇out)
                    # ∇y = if isa(t, Affine) || isa(t, Linear)
                    #     w = t.params.views[1]
                    #     w * (Bᵢ * ∇out)
                    # else
                    #     error("Only Affine and Linear Learnables are supported with DFA")
                    # end
                    # grad!(t, ∇y)
                end
            end
            # @show output_grad(t) grad(t) input_grad(t)
        end
        input_grad(chain)
    else
        error("Unsupported grad_calc: ", chain.grad_calc)
    end
end

# transform!(chain::Chain) = (foreach(transform!, chain.ts); chain.output.val)
# grad!(chain::Chain) = foreach(grad!, reverse(chain.ts))

function reset_params!{T}(chain::Chain{T}, θ::AbstractVector, ∇::AbstractVector)
    chain.params = consolidate_params(T, chain.ts, θ=θ, ∇=∇)
end

link_nodes!(chain::Chain, i_from::Int, i_to::Int) = link_nodes!(chain.ts[i_from], chain.ts[i_to])
