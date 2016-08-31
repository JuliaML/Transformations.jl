
# output = wx + b
immutable Affine{T} <: Transformation
    nin::Int
    nout::Int
    input::Node{:input,T,1}
    w::Node{:param,T,2}
    b::Node{:param,T,1}
    output::Node{:output,T,1}
    θ::CatView{2,T}
    ∇θ::CatView{2,T}

    function Affine(nin::Int, nout::Int)
        input = Node(:input, zeros(T, nin))
        w = Node(:param, zeros(T, nout, nin))
        b = Node(:param, zeros(T, nout))
        output = Node(:output, zeros(T, nout))
        new(nin, nout, input, w, b, output, CatView(w.val, b.val), CatView(w.∇, b.∇))
    end
end


Base.show(io::IO, t::Affine) = print(io, "Affine{$(t.nin)-->$(t.nout), input=$(t.input), w=$(t.w), b=$(t.b), output=$(t.output)}")

# compute output = wx + b
function transform!(aff::Affine)
    copy!(aff.output.val, aff.b.val)
    for o=1:aff.nout
        aff.output.val[o] += sum(aff.w.val[o,i] * aff.input.val[i] for i=1:aff.nin)
    end
    aff.output.val
end

# update the partial derivatives:
#   ∇x = ∂L/∂x
#   ∇w = ∂L/∂w
#   ∇b = ∂L/∂b
# use the chain rule, assuming that we've already updated ∇out = ∂L/∂y
function grad!(aff::Affine)
    # ∇x, ∇w
    for i=1:aff.nin
        ∇xᵢ = zero(eltype(aff.input.∇))
        for o=1:aff.nout
            ∇xᵢ += aff.w.val[o,i] * aff.output.∇[o]
            aff.w.∇[o,i] = aff.input.val[i] * aff.output.∇[o]
        end
        aff.input.∇[i] = ∇xᵢ
    end

    # ∇b
    copy!(aff.b.∇, aff.output.∇)
    return grad(aff)
end
