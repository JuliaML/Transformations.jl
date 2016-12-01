


# general formula for elementwise activation: output = act(input)
# some examples of act: logistic, tanh, relu, etc
immutable Activation{F,T} <: Transformation
    n::Int
    # α::T  # scaling parameter for some activations
    input::SumNode{T,1}
    output::OutputNode{T,1}

    # construct a new Activation, and link the nodes if it's the identity
    function Activation(n::Int) #, α::T = zero(T))
        input = InputNode(T,n)
        output = OutputNode(T,n)
        # if F == :identity
        #     link_nodes!(output, input)
        # end
        new(n, input, output)
    end
end
Activation(s::Symbol, n::Int) = Activation{s,Float64}(n)

Base.show{F,T}(io::IO, act::Activation{F,T}) = print(io, "$F{$(act.n)}")

input_length(act::Activation) = act.n
output_length(act::Activation) = act.n

# params_length(act::Activation) = 0
# params{F,T}(act::Activation{F,T}) = zeros(T,0)
# grad{F,T}(act::Activation{F,T}) = zeros(T,0)

# ----------------------------------------------------------------------------

const _exp_threshold = 20

# # identity: nothing to do, since we linked the input to output
# transform!(act::Activation{:identity}) = act.output.val
# grad!(act::Activation{:identity}) = act.input.∇
identity′{T<:Number}(x::T, y::T) = one(T)

# for the following, compute the derivative f′(x), where y = act(x) is assumed precomputed
# ref: https://en.wikipedia.org/wiki/Activation_function

logit(x::Real) = log(x / (one(x) - x))
logistic(x::Real) = one(x) / (one(x) + exp(-x))
logistic′{T<:Number}(x::T, y::T) = y * (one(T) - y)

# tanh: act(x) = (eˣ .- e⁻ˣ) ./ (eˣ .+ e⁻ˣ)
tanh′{T<:Number}(x::T, y::T) = one(T) - y^2

softsign{T<:Number}(x::T) = x / (one(T) + abs(x))
softsign′{T<:Number}(x::T, y::T) = (one(T) / (one(T) + abs(x)))^2

relu{T<:Number}(x::T) = max(zero(T), x)
relu′{T<:Number}(x::T, y::T) = x >= zero(T) ? one(T) : zero(T)

# softplus{T<:Number}(x::T) = log(one(T) + exp(x))
# softplus′{T<:Number}(x::T, y::T) = logistic(x)

# set to f(x)=x, f'(x)=1 when x is too big due to floating point math
softplus{T<:Number}(x::T) = x > _exp_threshold ? x : log(one(T) + exp(x))
softplus′{T<:Number}(x::T, y::T) = x > _exp_threshold ? one(T) : logistic(x)

sinusoid{T<:Number}(x::T) = sin(x)
sinusoid′{T<:Number}(x::T, y::T) = cos(x)

gaussian{T<:Number}(x::T) = exp(-(x^2))
gaussian′{T<:Number}(x::T, y::T) = -2x*y

threshold{T<:Number}(x::T) = x > zero(T) ? one(T) : zero(T)
threshold′{T<:Number}(x::T, y::T) = zero(T)

sign′{T<:Number}(x::T, y::T) = zero(T)



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
    :threshold,
    :sign,
]

for act in activations
    s = string(act)
    f′ = Symbol(s*"′")

    @eval begin
        # elementwise map from input to output
        transform!(act::Activation{Symbol($s)}) = map!($act, act.output.val, act.input.val)

        # backprop gradient calc using specialized derivative
        function grad!(act::Activation{Symbol($s)})
            ∇x = input_grad(act)
            ∇y = output_grad(act)
            x = input_value(act)
            y = output_value(act)
            for i=1:act.n
                ∇x[i] = $f′(x[i], y[i]) * ∇y[i]
            end
        end

        # x-only version
        function $f′(x::Number)
            y = $act(x)
            $f′(convert(typeof(y), x), y)
        end

        value_func(act::Activation{Symbol($s)}) = $act
        deriv_func(act::Activation{Symbol($s)}) = $f′

        # export both functions
        export $act, $f′
    end
end


# ----------------------------------------------------------------------------
# softmax

function transform!{T}(act::Activation{:softmax,T})
    val = act.input.val
    out = act.output.val
    maxinput = maximum(val)
    for i=1:act.n
        # subtract the maxinput to prevent overflow... doesn't change final result
        out[i] = exp(val[i] - maxinput)
    end
    s = one(T) / sum(out)
    for i=1:act.n
        out[i] *= s
    end
    out
end

# the calc is done in the CrossEntropyLoss... just pass that gradient back
function grad!{T}(act::Activation{:softmax,T})
    copy!(act.input.∇, act.output.∇)
end

# ----------------------------------------------------------------------------

default_range(act::Activation) = linspace(-5,5)

# user recipe adds a default x range
@recipe act{F}(act::Activation{F}) = act, default_range(act)

# type recipe converts to a function of xi
@recipe act{A<:Activation}(::Type{A}, act::A) = Transformations.value_func(act)

# ----------------------------------------------------------------------------
