##--- Learnable Transforms ---##

type CenteringTransformation{T<:Real}
    shift::ShiftTransform{T}
end
CenteringTransformation() = CenteringTransformation(ShiftTransform())
function CenteringTransformation(x::AbstractArray)
    tfm = CenteringTransformation()
    learn!(tfm, x)
end
learn!(tfm::CenteringTransformation, x) = (tfm.shift = mean(x); tfm)
transform!(tfm::CenteringTransformation, x) = (transform!(tfm.shift, x); x)
invert!(tfm::CenteringTransformation, x) = (invert!(tfm.shift, x); x)

type Standardize{T<:Real}
    shift::ShiftTransformation{T}
    scale::ScaleTransformation{T}
end
function learn!{T}(tfm::Standardize{T}, x)
    tfm.shift = mean(x)
    tfm.scale = one(T)/std(x)
end
function transform!(tfm::Standardize, x)
    transform!(tfm.shift, x)
    transform!(tfm.scale, x)
end
function invert!(tfm::Standardize, x)
    invert!(tfm.scale, x)
    invert!(tfm.shift, x)
end
