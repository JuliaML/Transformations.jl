
# output = wx
immutable Linear{T,P<:Params} <: Learnable
    nin::Int
    nout::Int
    input::SumNode{T,1}
    output::OutputNode{T,1}
    params::P
end

function Linear{T}(::Type{T}, nin::Int, nout::Int,
                    θ::AbstractVector = zeros(T, nout*nin),
                    ∇::AbstractVector = zeros(T, nout*nin))
    input = InputNode(T, nin)
    output = OutputNode(T, nout)
    params = Params(θ, ∇, ((nout,nin),))
    w = params.views[1]
    initialize_weights!(w)
    Linear(nin, nout, input, output, params)
end
Linear(nin::Int, nout::Int, args...) = Linear(Float64, nin, nout, args...)

function Base.show(io::IO, t::Linear)
    print(io, "Linear{$(t.nin)-->$(t.nout)}")
end

function transform!(t::Linear)
    w = t.params.views[1]
    x = t.input.val
    y = t.output.val
    for o=1:t.nout
        y[o] = sum(w[o,i] * x[i] for i=1:t.nin)
    end
    y
end

function grad!{T}(t::Linear{T})
    w = t.params.views[1]
    ∇w = t.params.∇_views[1]
    x = t.input.val
    ∇x = t.input.∇
    ∇y = t.output.∇

    # ∇x, ∇w
    for i=1:t.nin
        ∇xᵢ = zero(T)
        for o=1:t.nout
            ∇xᵢ += w[o,i] * ∇y[o]
            ∇w[o,i] = x[i] * ∇y[o]
        end
        ∇x[i] = ∇xᵢ
    end
    return
end
