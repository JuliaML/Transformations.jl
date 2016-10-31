
#=
Implements an online version of Layer Normalization from https://arxiv.org/abs/1607.06450:
    Ba et al (2016): Layer Normalization

The input a is normalized ((a .- μ) ./ σ), and then re-scaled and shifted
with learnable params gain (g) and bias (b).
=#

#=
Input is a, output is y (same size as a):
    y = g ∘ z + b
    z = (a - μₜ) / σₜ
=#
type LayerNorm{T,P<:Params,W<:Weight} <: Learnable
    n::Int  # maps n --> n
    m::T  # this period's mean(aₜ)
    μ::T  # running average of m = mean(aₜ)
    σ::T  # running average of s = std(aₜ)
    σ̂::T  # the clamped version or σ
    αₜ::T
    wgt::W
    z::Vector{T}
    input::Node{:input,T,1}    # a
    output::Node{:output,T,1}  # y
    params::P                  # holds g and b
end

function LayerNorm{T}(::Type{T}, n::Int,
                    θ::AbstractVector = zeros(T, 2n),
                    ∇::AbstractVector = zeros(T, 2n);
                    lookback::Int = 100,
                    α::Float64 = NaN,
                    wgt::Weight =  BoundedEqualWeight(isnan(α) ? lookback : α)
                    )
    input = Node(:input, zeros(T, n))
    output = Node(:output, zeros(T, n))
    params = Params(θ, ∇, ((n,), (n,)))
    g, b = params.views
    fill!(g, one(T))
    fill!(b, zero(T))
    LayerNorm(n, zero(T), zero(T), one(T), one(T), zero(T), wgt, zeros(T,n), input, output, params)
end
LayerNorm(n::Int, args...; kw...) = LayerNorm(Float64, n, args...; kw...)

function Base.show(io::IO, t::LayerNorm)
    print(io, "LayerNorm{n=$(t.n), mu=$(t.μ), sigma=$(t.σ)}")
end

# compute output = wx + b
function transform!{T}(layer::LayerNorm{T})
    g, b = layer.params.views
    a = layer.input.val
    y = layer.output.val

    # update the αₜ weighting
    OnlineStats.updatecounter!(layer.wgt)
    layer.αₜ = weight(layer.wgt)

    # normalize the layer
    # TODO: μ/σ should be weighted avgs using BoundedEqualWeight
    layer.m = mean(a)
    layer.μ = layer.αₜ * layer.m + (one(T) - layer.αₜ) * layer.μ
    if isnan(layer.μ)
        warn("layer.μ is NaN:")
        @show layer a g b map(extrema, (a, g, b))
    end
    layer.σ = layer.αₜ * std(a) + (one(T) - layer.αₜ) * layer.μ
    layer.σ̂ = if layer.σ == 0
        @show layer.μ, layer.σ, layer.σ̂
        1e-10
    else
        layer.σ
    end

    # mult by g and add b
    @inbounds for o=1:layer.n
        layer.z[o] = (a[o] - layer.μ) / layer.σ̂
        y[o] = g[o] * layer.z[o] + b[o]
    end
    y
end

function grad!{T}(layer::LayerNorm{T})
    g, b = layer.params.views
    ∇g, ∇b = layer.params.∇_views
    a = layer.input.val
    y = layer.output.val
    ∇a = layer.input.∇
    ∇y = layer.output.∇

    D = layer.n
    Dinv = one(T) / D
    Dm1 = D - one(T)
    # @show layer.μ, layer.σ, layer.σ̂

    # TODO: ∇a, ∇g
    fill!(∇a, zero(T))
    for o=1:layer.n
        # ∇g
        ∇g[o] = ∇y[o] * layer.z[o]

        # ∇a
        base_dzₒdaₚ = Dinv - layer.αₜ * layer.z[o] * (layer.μ - layer.m) / layer.σ̂
        for p=1:layer.n
            dzₒdaₚ = base_dzₒdaₚ + layer.z[o] * layer.z[p] / Dm1
            dzₒdaₚ = (o==p ? one(T) : zero(T)) - layer.αₜ * dzₒdaₚ
            dzₒdaₚ /= layer.σ̂
            ∇a[p] += ∇y[o] * g[o] * dzₒdaₚ
        end
    end

    # ∇b
    copy!(∇b, ∇y)
    return
end
