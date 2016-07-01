using Transformations
using Base.Test
using Distributions

## Basic ##
xv = abs(randn(200,100))+0.1 # array input
@test isapprox(xv, transform(IdentityTransformation(),xv))
@test isapprox(log(xv), transform(LogTransformation(),xv))
@test isapprox(exp(xv), transform(ExpTransformation(),xv))

xs = rand()+0.1 # scalar input
@test isapprox(xs, transform(IdentityTransformation(),xs))
@test isapprox(log(xs), transform(LogTransformation(),xs))
@test isapprox(exp(xs), transform(ExpTransformation(),xs))

## Shifting and Scaling ##
xv = randn(100)
s = 2.3
tfm = ScaleTransformation(s)
@test isapprox(xv*s, transform(ScaleTransformation(s), xv))
@test isapprox(xv+s, transform(ShiftTransformation(s), xv))
#@test isapprox(xv-mean(xv), transform(CenteringTransformation(xv),xv))

## Generalized Linear Transform ##
μ,σ = 1.0,1.0
tfm = GeneralizedLinearTransformation(Normal(μ,σ))
@test Normal(μ,σ) == transform(tfm, μ)

x = -1.0
λ = exp(x)
tfm = GeneralizedLinearTransformation(Poisson())
@test Poisson(λ) == transform(tfm, x)
