using Transformations
using Base.Test
using Distributions
import OnlineStats: Means

# ## Basic ##
# xv = abs(randn(200,100))+0.1 # array input
# @test isapprox(xv, transform(IdentityTransformation(),xv))
# @test isapprox(log(xv), transform(LogTransformation(),xv))
# @test isapprox(exp(xv), transform(ExpTransformation(),xv))

# xs = rand()+0.1 # scalar input
# @test isapprox(xs, transform(IdentityTransformation(),xs))
# @test isapprox(log(xs), transform(LogTransformation(),xs))
# @test isapprox(exp(xs), transform(ExpTransformation(),xs))

# ## Shifting and Scaling ##
# xv = randn(100)
# s = 2.3
# tfm = ScaleTransformation(s)
# @test isapprox(xv*s, transform(ScaleTransformation(s), xv))
# @test isapprox(xv+s, transform(ShiftTransformation(s), xv))

# tfm = ShiftTransformation(s)
# y = transform(tfm, xv)
# @test isapprox(xv, transform(inv(tfm), y))
# @test isapprox(xv, invert(tfm, y))

# #@test isapprox(xv-mean(xv), transform(CenteringTransformation(xv),xv))

# ## Generalized Linear Transform ##
# μ,σ = 1.0,1.0
# tfm = GeneralizedLinearTransformation(Normal(μ,σ))
# @test Normal(μ,σ) == transform(tfm, μ)

# x = -1.0
# λ = exp(x)
# tfm = GeneralizedLinearTransformation(Poisson())
# @test Poisson(λ) == transform(tfm, x)

Transformations.input_size(m::Means) = size(m.value)
Transformations.output_size(m::Means) = size(m.value)
Transformations.transform!(y, m::Means, x) = (y[:] = x - m.value)
Transformations.is_learnable(m::Means) = true
Transformations.learn!(m, x) = OnlineStats.fit!(m, x)

m = Means(5); #fit!(m, 1:5)
t = transformation(m)
@show t
@show learn!(t, 1:5)
@show t
@show y = transform(t, 10ones(5))

@test y == 10ones(5) - (1:5)


