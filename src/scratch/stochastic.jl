# # TODO -- find a better home for Domains (or delete them...)
# abstract Domain

# immutable RealDomain <: Domain end
# Base.in(x,::Type{RealDomain}) = typeof(x) <: Real

# immutable NonNegRealDomain <: Domain end
# Base.in(x,::Type{NonNegRealDomain}) = (typeof(x) <: Real && x >= 0)

# domain(::GaussianTransformation) = Real
# domain(::PoissonTransformation) = NonNegRealDomain

## GAUSSIAN ##
"""
    tfm = GaussianTransformation(σ)

Adds zero-mean gaussian noise to an input variable x with standard deviation σ.

X -> ξ,  ξ ∼ N(X,σ)
"""
immutable GaussianTransformation{T<:Real} <: StochasticTransformation
    σ::T
end

# constructors
"""
Adds zero-mean gaussian noise to an input variable x with standard deviation σ.

X -> ξ,  ξ ∼ N(X,σ)
"""
GaussianTransformation(d::Normal) = GaussianTransformation(d.σ)
GaussianTransformation() = GaussianTransformation(1.0)
Transformation(d::Normal) = GaussianTransformation(d.σ)
StochasticTransformation(d::Normal) = GaussianTransformation(d.σ)

# functions
Base.rand(tfm::GaussianTransformation, x) = x + randn()*tfm.σ
transform(tfm::GaussianTransformation, x) = Normal(x,tfm.σ)


## POISSON ##
"""
Transform a non-negative variable, x, into poisson random variable with mean x.

x -> ξ,  ξ ∼ Poisson(x)
"""
immutable PoissonTransformation <: StochasticTransformation end

# constructors
Transformation(d::Poisson) = PoissonTransformation()
StochasticTransformation(d::Poisson) = PoissonTransformation()

# functions
Base.rand(::PoissonTransformation, x) = rand(Poisson(x))
transform(::PoissonTransformation, x) = Poisson(x)


## BERNOULLI ##
"""
Transform an input 0 ≤ x ≤ 1 into a Bernoulli random variable with mean x.

x -> ξ,  ξ ∼ Bernoulli(x)
"""
immutable BernoulliTransformation <: StochasticTransformation end

# constructors
Transformation(d::Bernoulli) = BernoulliTransformation()
StochasticTransformation(d::Bernoulli) = BernoulliTransformation()

# functions
Base.rand(::BernoulliTransformation, x) = rand(Bernoulli(x))
transform(::BernoulliTransformation, x) = Bernoulli(x)


## NEGATIVE BINOMIAL ##
"""
Transform a non-negative variable, x, into a negative binomial random variable
with mean x and (fixed) variance σ²

x -> ξ,  ξ ∼ NegativeBinomial,  E{ξ} = x
"""
immutable NegativeBinomialTransformation{T<:Real} <: StochasticTransformation 
    σ²::T
end

# constructors
Transformation(d::NegativeBinomial) = NegativeBinomialTransformation(var(d))
StochasticTransformation(d::NegativeBinomial) = NegativeBinomialTransformation(var(d))
NegativeBinomialTransformation(d::NegativeBinomial) = NegativeBinomialTransformation(var(d))
NegativeBinomialTransformation() = NegativeBinomialTransformation(1.0)

# functions
function specify_moments(::Type{NegativeBinomial},μ,σ²)
    p = μ / σ²
    r = μ*p / (1.0-p)
    return NegativeBinomial(r,p)
end
Base.rand(tfm::NegativeBinomialTransformation, x) = rand(specify_moments(NegBinomial,x,tfm.σ²))
transform(tfm::NegativeBinomialTransformation, x) = specify_moments(NegBinomial,x,tfm.σ²)
