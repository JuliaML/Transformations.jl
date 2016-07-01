##--- Identity ---##
type IdentityTransformation <: Transformation end
transform!(::IdentityTransformation, x) = x
invert!(::IdentityTransformation, x) = x
transform(::IdentityTransformation, x) = copy(x)
invert(::IdentityTransformation, x) = copy(x)
isinvertible(::IdentityTransformation) = true
Base.inv(::IdentityTransformation) = IdentityTransformation()

##--- Invertible Transforms ---##
for (t1,f1,t2,f2) in ( (:ExpTransformation, :exp, :LogTransformation, :log),
                       (:LogisticTransformation, :logistic, :LogitTransformation, :logit) )
    @eval begin
        type $(t1) <: Transformation end
        type $(t2) <: Transformation end
        transform!(::$(t1), x) = map!($(f1),x)
        transform!(::$(t2), x) = map!($(f2),x)
        transform(::$(t1), x) = $(f1)(x)
        transform(::$(t2), x) = $(f2)(x)
        invert!(::$(t1), x) = map!($(f2),x)
        invert!(::$(t2), x) = map!($(f1),x)
        invert(::$(t1), x) = $(f2)(x)
        invert(::$(t2), x) = $(f1)(x)
        Base.inv(::$(t1)) = $(t2)()
        Base.inv(::$(t2)) = $(t1)()
        isinvertible(::$(t1)) = true
        isinvertible(::$(t2)) = true
    end
end

## --- Scaling --- ##
immutable ScaleTransformation{T<:Number} <: Transformation
    val::T
end
ScaleTransformation() = ScaleTransformation(1.0)
isinvertible(::ScaleTransformation) = true
function invert!{T}(tfm::ScaleTransformation{T}, x::AbstractArray{T})
    scale!(one(T)/tfm.val, x)
end
function transform!{T}(tfm::ScaleTransformation, x::AbstractArray{T})
    scale!(tfm.val, x)
end
Base.inv{T<:Number}(tfm::ScaleTransformation{T}) = ScaleTransformation(one(T)/tfm.val)

## --- Shifting --- ##
immutable ShiftTransformation{T<:Number} <: Transformation
    val::T
end
ShiftTransformation() = ShiftTransformation(0.0)
isinvertible(::ShiftTransformation) = true
function transform!{T}(tfm::ShiftTransformation{T}, x::AbstractArray{T})
    for i in eachindex(x); x[i] += tfm.val; end
    x
end
function invert!{T}(tfm::ShiftTransformation{T}, x::AbstractArray{T})
    for i in eachindex(x); x[i] -= tfm.val; end
    x
end
Base.inv(tfm::ShiftTransformation) = ShiftTransformation(-tfm.val)
