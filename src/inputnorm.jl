
#=
Similar to Batch Normalization, except online and without the rescaling/skew
    y = (a .- μ) ./ σ

    TODO: This is currently broken because OnlineStats.Variances no longer
    exists. 
=#
type InputNorm{T,W<:Weight} <: Transformation
    n::Int  # maps n --> n
    vars::Variances{W}
    input::SumNode{T,1}    # a
    output::OutputNode{T,1}  # y
end

function InputNorm{T}(::Type{T}, n::Int,
                    lookback::Int = 100,
                    α::Float64 = NaN,
                    wgt::Weight =  BoundedEqualWeight(isnan(α) ? lookback : α)
                    )
    InputNorm(n,
        Variances(n, wgt),
        InputNode(T, n),
        OutputNode(T, n)
    )
end
InputNorm(n::Int, args...; kw...) = InputNorm(Float64, n, args...; kw...)

function Base.show(io::IO, t::InputNorm)
    print(io, "InputNorm{n=$(t.n)}")
end

function transform!{T}(layer::InputNorm{T})
    a = layer.input.val
    y = layer.output.val

    OnlineStats.fit!(layer.vars, a)
    μ = mean(layer.vars)
    σ = std(layer.vars)
    for i=1:layer.n
        y[i] = (a[i] - μ[i]) / max(σ[i], T(1e-10))
    end
    y
end

function grad!{T}(layer::InputNorm{T})
    ∇a = layer.input.∇
    ∇y = layer.output.∇

    σ = std(layer.vars)
    for i=1:layer.n
        ∇a[i] = ∇y[i] / max(σ[i], T(1e-10))
    end
    ∇a
end
