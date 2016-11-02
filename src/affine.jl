
# output = wx + b
immutable Affine{T,P<:Params} <: Learnable
    nin::Int
    nout::Int
    input::SumNode{T,1}
    output::OutputNode{T,1}
    params::P
end

function Affine{T}(::Type{T}, nin::Int, nout::Int,
                    θ::AbstractVector = zeros(T, nout*(nin+1)),
                    ∇::AbstractVector = zeros(T, nout*(nin+1)))
    input = InputNode(T, nin)
    output = OutputNode(T, nout)
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

# compute output = wx + b
function transform!(t::Affine)
    w, b = t.params.views
    x = t.input.val
    y = t.output.val
    copy!(y, b)
    for o=1:t.nout
        y[o] += sum(w[o,i] * x[i] for i=1:t.nin)
    end
    y
end

# update the partial derivatives:
#   ∇x = ∂L/∂x
#   ∇w = ∂L/∂w
#   ∇b = ∂L/∂b
# use the chain rule, assuming that we've already updated ∇out = ∂L/∂y
function grad!(t::Affine)
    w, b = t.params.views
    ∇w, ∇b = t.params.∇_views
    x = t.input.val
    ∇x = t.input.∇
    ∇y = t.output.∇

    # ∇x, ∇w
    for i=1:t.nin
        ∇xᵢ = zero(eltype(∇x))
        for o=1:t.nout
            ∇xᵢ += w[o,i] * ∇y[o]
            ∇w[o,i] = x[i] * ∇y[o]
        end
        ∇x[i] = ∇xᵢ
    end

    # ∇b
    copy!(∇b, ∇y)
    return
end
