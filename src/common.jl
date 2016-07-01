function transform{T<:Transformation, A<:AbstractArray}(tfm::T, x::A)
    y = copy(x)
    transform!(tfm, y)
end

function invert{T<:Transformation, A<:AbstractArray}(tfm::T, x::A)
    y = copy(x)
    invert!(tfm, y)
end
