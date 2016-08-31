
# general formula for elementwise activation: output = f(input)
# some examples of f: logistic, tanh, relu, etc
immutable Activation{F,T} <: Transformation
    n::Int
    # α::T  # scaling parameter for some activations
    input::Node{:input,T,1}
    output::Node{:output,T,1}

    # construct a new Activation, and link the nodes if it's the identity
    function Activation(n::Int) #, α::T = zero(T))
        input = Node(:input, zeros(T,n))
        output = Node(:output, zeros(T,n))
        if F == :identity
            link_nodes!(output, input)
        end
        new(n, input, output)
    end
end

# ----------------------------------------------------------------------------


# identity: nothing to do, since we linked the input to output
transform!(f::Activation{:identity}) = f.output.val
grad!(f::Activation{:identity}) = f.input.∇

# for the following, compute the derivative f′(x), where y = f(x) is assumed precomputed
# ref: https://en.wikipedia.org/wiki/Activation_function

# logistic (sigmoid): f(x) = 1 ./ (1 .+ exp.(-x))
logistic′{T<:Number}(x::T, y::T) = y * (one(T) - y)

# tanh: f(x) = (eˣ .- e⁻ˣ) ./ (eˣ .+ e⁻ˣ)
tanh′{T<:Number}(x::T, y::T) = one(T) - y^2

softsign{T<:Number}(x::T) = x / (one(T) + abs(x))
softsign′{T<:Number}(x::T, y::T) = one(T) / (one(T) + abs(x))^2

relu{T<:Number}(x::T) = max(zero(T), x)
relu′{T<:Number}(x::T, y::T) = x >= zero(T) ? one(T) : zero(T)

softplus{T<:Number}(x::T) = log(one(T) + exp(x))
softplus′{T<:Number}(x::T, y::T) = logistic(x)

sinusoid{T<:Number}(x::T) = sin(x)
sinusoid′{T<:Number}(x::T, y::T) = cos(x)

gaussian{T<:Number}(x::T) = exp(-(x^2))
gaussian′{T<:Number}(x::T, y::T) = -2x*y



# ----------------------------------------------------------------------------

# generic implementations... ensure there's a derivative method of the correct name

const activations = [
    :logistic,
    :tanh,
    :softsign,
    :relu,
    :softplus,
    :sinusoid,
    :gaussian,
]

for f in activations
    s = string(f)
    f′ = Symbol(s*"′")

    @eval begin
        # elementwise map from input to output
        transform!(f::Activation{Symbol($s)}) = map!($f, f.output.val, f.input.val)

        # backprop gradient calc using specialized derivative
        function grad!(f::Activation{Symbol($s)})
            for i=1:f.n
                f.input.∇[i] = $f′(f.input.val[i], f.output.val[i]) * f.output.∇[i]
            end
            # no params, so nothing to return
        end

        # x-only version
        function $f′(x::Number)
            y = $f(x)
            $f′(convert(typeof(y), x), y)
        end

        value_func(f::Activation{Symbol($s)}) = $f
        deriv_func(f::Activation{Symbol($s)}) = $f′

        # export both functions
        export $f, $f′
    end
end


# ----------------------------------------------------------------------------

default_range(f::Activation) = linspace(-5,5)

# user recipe adds a default x range
@recipe f{F}(f::Activation{F}) = f, default_range(f)

# type recipe converts to a function of xi
@recipe f{A<:Activation}(::Type{A}, f::A) = Transformations.value_func(f)

# ----------------------------------------------------------------------------
