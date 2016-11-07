

# output = vcat(input1, input2, ...)
immutable Concat{T} <: Transformation
    n::Int
    input::CatNode{T,1}
    output::OutputNode{T,1}
end
Concat(n::Int) = Concat(Float64, n)

# sort of tricky... we do the concatenating in the input node, so instead
# of copying val/∇ through, just make them point to the same arrays
function Concat{T}(::Type{T}, n::Int)
    input = InputNode(:cat, T, n)
    output = OutputNode(T, n)
    output.val = input.val
    output.∇ = input.∇
    Concat(n, input, output)
end

transform!(t::Concat) = t.output.val
grad!(t::Concat) = t.input.∇

# ---------------------------------------------------------

# output = input1 .* input2 .* ...
immutable Gate{T} <: Transformation
    n::Int
    input::ProdNode{T,1}
    output::OutputNode{T,1}
end
Gate(n::Int) = Gate(Float64, n)

# sort of tricky... we do the gating in the input node, so instead
# of copying val/∇ through, just make them point to the same arrays
function Gate{T}(::Type{T}, n::Int)
    input = InputNode(:*, T, n)
    output = OutputNode(T, n)
    output.val = input.val
    output.∇ = input.∇
    Gate(n, input, output)
end

transform!(t::Gate) = t.output.val
grad!(t::Gate) = t.input.∇
