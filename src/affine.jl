
# output = wx + b
immutable Affine{T,P<:Params} <: Learnable
    nin::Int
    nout::Int
    input::Node{:input,T,1}
    output::Node{:output,T,1}
    params::P
end

function Affine{T}(::Type{T}, nin::Int, nout::Int,
                    θ::AbstractVector = zeros(T, nout*(nin+1)),
                    ∇::AbstractVector = zeros(T, nout*(nin+1)))
    input = Node(:input, zeros(T, nin))
    output = Node(:output, zeros(T, nout))
    params = Params(θ, ∇, ((nout,nin), (nout,)))
    w, b = params.views
    initialize_weights!(w)
    initialize_bias!(b)
    Affine(nin, nout, input, output, params)
end
Affine(nin::Int, nout::Int, args...) = Affine(Float64, nin, nout, args...)

function Base.show(io::IO, t::Affine)
    print(io, "Affine{$(t.nin)-->$(t.nout)}")
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
