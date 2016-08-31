
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
