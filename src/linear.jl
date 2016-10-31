
# output = wx
immutable Linear{T,P<:Params} <: Learnable
    nin::Int
    nout::Int
    input::Node{:input,T,1}
    output::Node{:output,T,1}
    params::P
end

function Linear{T}(::Type{T}, nin::Int, nout::Int,
                    θ::AbstractVector = zeros(T, nout*nin),
                    ∇::AbstractVector = zeros(T, nout*nin))
    input = Node(:input, zeros(T, nin))
    output = Node(:output, zeros(T, nout))
    params = Params(θ, ∇, ((nout,nin),))
    w = params.views[1]
    initialize_weights!(w)
    Linear(nin, nout, input, output, params)
end
Linear(nin::Int, nout::Int, args...) = Linear(Float64, nin, nout, args...)

function Base.show(io::IO, t::Linear)
    print(io, "Linear{$(t.nin)-->$(t.nout)}")
end

function transform!(aff::Linear)
    w = aff.params.views[1]
    x = aff.input.val
    y = aff.output.val
    for o=1:aff.nout
        y[o] = sum(w[o,i] * x[i] for i=1:aff.nin)
    end
    y
end

function grad!{T}(aff::Linear{T})
    w = aff.params.views[1]
    ∇w = aff.params.∇_views[1]
    x = aff.input.val
    ∇x = aff.input.∇
    ∇y = aff.output.∇

    # ∇x, ∇w
    for i=1:aff.nin
        ∇xᵢ = zero(T)
        for o=1:aff.nout
            ∇xᵢ += w[o,i] * ∇y[o]
            ∇w[o,i] = x[i] * ∇y[o]
        end
        ∇x[i] = ∇xᵢ
    end
    return
end
