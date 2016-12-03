

function nnet_layers(nin::Int, nout::Int;
              nh = Int[],
              inner_activation = :tanh,
              final_activation = :identity,
              layernorm = false,
              inputnorm = false,
              kw...)
    ns = vcat(nin, nh, nout)
    num_affine = length(ns) - 1
    layers = Transformation[]

    for i=1:num_affine
        # push!(layers, (layernorm ? LayerNorm : Affine)(ns[i], ns[i+1]))
        if inputnorm
            push!(layers, InputNorm(ns[i]; kw...))
        end

        if layernorm && !(i==num_affine && final_activation == :identity)
            push!(layers, Linear(ns[i], ns[i+1]))
            push!(layers, LayerNorm(ns[i+1]; kw...))
        else
            push!(layers, Affine(ns[i], ns[i+1]))
        end

        if inner_activation != :identity && i < num_affine
            push!(layers, Activation(inner_activation, ns[i+1]))
        end
    end
    if final_activation != :identity
        push!(layers, Activation(final_activation, nout))
    end
    layers
end

function nnet(nin::Int, nout::Int, nh::AbstractVector{Int},
              iact = :tanh, fact = :identity;
              prep = NoPreprocessing(),
              grad_calc = :backprop,
              kw...)
    Chain(Float64, nnet_layers(
        nin, nout;
        nh=nh,
        inner_activation=iact,
        final_activation=fact,
        kw...
    ), prep=prep, grad_calc=grad_calc)
end

# ---------------------------------------------------------------------

"""
A ResNet layer: y = F(x) + x

Kaiming He et al (2015): "Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385v1.pdf

Notes:
    - x and y have the same dimension
    - F can be any transformation mapping n-->n
"""
type ResidualLayer{T,F<:Transformation,P} <: Learnable
    n::Int
    f::F
    input::SumNode{T,1}
    output::OutputNode{T,1}
    params::P
end
ResidualLayer(n::Int; kw...) = ResidualLayer(Float64, n; kw...)
ResidualLayer(n::Int, f::Transformation) = ResidualLayer(Float64, n, f)
ResidualLayer{T}(::Type{T}, n::Int; nh = Int[], kw...) = ResidualLayer(T, n, nnet(n, n, nh; kw...))

params(t::ResidualLayer) = params(t.f)

function ResidualLayer{T}(::Type{T}, n::Int, f::Transformation)
    @assert input_length(f) == n
    @assert output_length(f) == n
    ResidualLayer(
        n,
        f,
        InputNode(T,n),
        OutputNode(T,n),
        f.params
    )
end

function transform!(t::ResidualLayer)
    x = input_value(t)
    y = output_value(t)

    yᶠ = transform!(t.f, x)
    for i=1:t.n
        y[i] = yᶠ[i] + x[i]
    end
    y
end

function grad!{T}(t::ResidualLayer{T})
    ∇y = output_grad(t)
    ∇x = input_grad(t)
    ∇xᶠ = input_grad(t.f)

    grad!(t.f, ∇y)
    for i=1:t.n
        ∇x[i] = ∇xᶠ[i] + ∇y[i]
    end
    ∇x
end

function reset_params!{T}(t::ResidualLayer{T}, θ::AbstractVector, ∇::AbstractVector)
    t.params = reset_params!(t.f, θ, ∇)
end

# ---------------------------------------------------------------------

"Construct nblocks layers of stacked ResidualLayer blocks, with a final affine mapping to nout"
function resnet(nin::Int, nout::Int, nblocks::Int; kw...)
    layers = Transformation[ResidualLayer(nin; kw...) for i=1:nblocks]
    push!(layers, Affine(nin, nout))
    Chain(Float64, layers)
end
