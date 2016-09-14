
# # initial weightings for w
# function initial_weights{T}(::Type{T}, nin::Int, nout::Int, strat::Symbol = :default)
#     if strat == :default
#         (0.5 / sqrt(nin)) * randn(T, nout, nin)
#     else
#         error("Unknown strat in initial_weights: $strat")
#     end
# end

function initialize_weights!{T}(w::AbstractArray{T})
    nin, nout = size(w)
    scalar = sqrt(T(2.0 / (nin + nout)))
    for i in eachindex(w)
        w[i] = scalar * randn(T)
    end
end

function initialize_bias!{T}(b::AbstractArray{T})
    fill!(b, zero(T))
end

# output = wx + b
immutable Affine{T} <: Learnable
    nin::Int
    nout::Int
    input::Node{:input,T,1}
    output::Node{:output,T,1}
    params::Params

    function Affine(nin::Int, nout::Int,
                    θ::AbstractVector = zeros(T, nout*(nin+1)),
                    ∇::AbstractVector = zeros(T, nout*(nin+1)))
        input = Node(:input, zeros(T, nin))
        output = Node(:output, zeros(T, nout))
        params = Params(θ, ∇, ((nout,nin), (nout,)))
        w, b = params.views
        initialize_weights!(w)
        initialize_bias!(b)
        new(nin, nout, input, output, params)
    end
end

Affine{T}(::Type{T}, nin::Int, nout::Int, args...) = Affine{T}(nin, nout, args...)
Affine(nin::Int, nout::Int, args...) = Affine{Float64}(nin, nout, args...)

# Base.show(io::IO, aff::Affine) = print(io, "Affine{$(aff.nin)-->$(aff.nout), input=$(aff.input), w=$(aff.w), b=$(aff.b), output=$(t.output)}")
function Base.show(io::IO, t::Affine)
    print(io, "Affine{$(t.nin)-->$(t.nout), input=$(t.input), output=$(t.output)}")
end

params_length(aff::Affine) = length(aff.params.θ)
params(aff::Affine) = aff.params.θ
grad(aff::Affine) = aff.params.∇

# compute output = wx + b
function transform!(aff::Affine)
    w, b = aff.params.views
    x = aff.input.val
    y = aff.output.val
    copy!(y, b)
    for o=1:aff.nout
        y[o] += sum(w[o,i] * x[i] for i=1:aff.nin)
    end
    y
end

# update the partial derivatives:
#   ∇x = ∂L/∂x
#   ∇w = ∂L/∂w
#   ∇b = ∂L/∂b
# use the chain rule, assuming that we've already updated ∇out = ∂L/∂y
function grad!(aff::Affine)
    w, b = aff.params.views
    ∇w, ∇b = aff.params.∇_views
    x = aff.input.val
    ∇x = aff.input.∇
    ∇y = aff.output.∇

    # ∇x, ∇w
    for i=1:aff.nin
        ∇xᵢ = zero(eltype(∇x))
        for o=1:aff.nout
            ∇xᵢ += w[o,i] * ∇y[o]
            ∇w[o,i] = x[i] * ∇y[o]
        end
        ∇x[i] = ∇xᵢ
    end

    # ∇b
    copy!(∇b, ∇y)
    return
end
