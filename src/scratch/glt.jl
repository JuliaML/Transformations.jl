"""
Transformation associated with a Generalized Linear Model (GLM). Maps an input
`x` to a random variable with expected value equal to `g(x)` where g(⋅) is a
inverse link function.
"""
immutable GeneralizedLinearTransformation{ST<:StochasticTransformation}
    noise::ST
    invlink::Transformation
end

# constructors
function GeneralizedLinearTransformation{D<:Distribution}(d::D)
    tfm = StochasticTransformation(d)
    invlink = cannonical_inv_link(d)
    GeneralizedLinearTransformation(tfm,invlink)
end
function GeneralizedLinearTransformation{D<:Distribution, T<:Transformation}(d::D, invlink::T)
    tfm = StochasticTransformation(d)
    GeneralizedLinearTransformation(tfm,invlink)
end

# functions
function transform(tfm::GeneralizedLinearTransformation, x)
    transform(tfm.noise, transform(tfm.invlink, x))
end
function rand(tfm::GeneralizedLinearTransformation, x)
    rand(tfm.noise, transform(tfm.invlink, x))
end

# cannonical inverse link functions
cannonical_inv_link(::Normal) = IdentityTransformation()
cannonical_inv_link(::Bernoulli) = LogisticTransformation()
cannonical_inv_link(::Poisson) = ExpTransformation()
cannonical_inv_link(::NegativeBinomial) = ExpTransformation()
