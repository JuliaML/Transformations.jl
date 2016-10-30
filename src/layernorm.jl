
#=
Implements Layer Normalization from https://arxiv.org/abs/1607.06450:
    Ba et al (2016): Layer Normalization

The rotated input (wx) is normalized ((wx .- mean(wx)) ./ std(wx)) using a learnable matrix w,
and then re-scaled and shifted with learnable params gain (g) and bias (b).
=#

# output = wx + b
type LayerNorm{T,P<:Params} <: Learnable
    nin::Int
    nout::Int
    μ::T
    σ::T
    σ̂::T  # the clamped version
    input::Node{:input,T,1}
    output::Node{:output,T,1}
    params::P
end

function LayerNorm{T}(::Type{T}, nin::Int, nout::Int,
                    θ::AbstractVector = zeros(T, nout*(nin+2)),
                    ∇::AbstractVector = zeros(T, nout*(nin+2)))
    input = Node(:input, zeros(T, nin))
    output = Node(:output, zeros(T, nout))
    params = Params(θ, ∇, ((nout,nin), (nout,), (nout,)))
    w, g, b = params.views
    initialize_weights!(w)
    fill!(g, one(T))
    fill!(b, zero(T))
    LayerNorm(nin, nout, zero(T), one(T), one(T), input, output, params)
end
LayerNorm(nin::Int, nout::Int, args...) = LayerNorm(Float64, nin, nout, args...)

function Base.show(io::IO, t::LayerNorm)
    print(io, "LayerNorm{$(t.nin)-->$(t.nout), mu=$(t.μ), sigma=$(t.σ)}")
end

# compute output = wx + b
function transform!{T}(layer::LayerNorm{T})
    w, g, b = layer.params.views
    x = layer.input.val
    y = layer.output.val

    # first set y = wx
    @inbounds for o=1:layer.nout
        y[o] = sum(w[o,i] * x[i] for i=1:layer.nin)
    end

    # normalize the layer
    layer.μ = mean(y)
    isnan(layer.μ) && @show layer y x w g b map(extrema, (y, x, w, g, b))
    layer.σ = std(y)
    layer.σ̂ = max(layer.σ, 1e-8)

    # mult by g and add b
    @inbounds for o=1:layer.nout
        y[o] = g[o] * (y[o] - layer.μ) / layer.σ̂ + b[o]
    end
    y
end

function grad!{T}(layer::LayerNorm{T})
    w, g, b = layer.params.views
    ∇w, ∇g, ∇b = layer.params.∇_views
    x = layer.input.val
    y = layer.output.val
    ∇x = layer.input.∇
    ∇y = layer.output.∇
    D = layer.nout
    Dinv = one(T) / D

    @show layer.μ layer.σ layer.σ̂

    # ∇x, ∇w
    fill!(∇g, zero(T))
    @inbounds for i=1:layer.nin
        ∇xᵢ = zero(T)
        for o=1:layer.nout
            ∇xᵢ += if layer.σ == layer.σ̂
                (g[o] * (w[o,i] - Dinv) / layer.σ̂ -
                Dinv * y[o] * (x[i] + layer.μ * (Dinv-T(2))) / sqrt(layer.σ̂)) / layer.σ̂
            else
                g[o] * w[o,i] / layer.σ̂
            end * ∇y[o]
            ∇w[o,i] = g[o] * x[i] * ∇y[o] / layer.σ̂
            # ∇g[o] += ((x[i] * w[o,i] - layer.μ) / layer.σ̂) * ∇y[o]
            ∇g[o] += x[i] * w[o,i]
        end
        ∇x[i] = ∇xᵢ
    end

    @inbounds for o=1:layer.nout
        ∇g[o] = ∇y[o] * (∇g[o] - layer.μ) / layer.σ̂
    end

    # ∇b
    copy!(∇b, ∇y)
    return
end
