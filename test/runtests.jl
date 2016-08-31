using Transformations
using Base.Test
# using Distributions
# import OnlineStats: Means

using Losses

let nin=2, nout=3, input=rand(nin), target=rand(nout)
    a = Affine{Float64}(nin, nout)
    loss = L2DistLoss()
    @show a loss

    for i=1:2
        println()

        output = transform!(a, input)
        @show i output

        @test output == a.w.val * input + a.b.val
        @test size(a.input.val) == (nin,)
        @test size(a.w.val) == (nout,nin)
        @test size(a.b.val) == (nout,)
        @test size(a.output.val) == (nout,)
        @test size(output) == (nout,)

        l = value(loss, target, output)
        dl = deriv(loss, target, output)
        @show l dl

        ∇ = grad!(a, dl)
        ∇w = ∇[1:(nin*nout)]
        ∇b = ∇[(nin*nout+1):end]
        @show ∇ a.w.∇ a.b.∇

        @test a.output.∇ == dl
        @test isa(∇, Transformations.CatView)
        @test size(∇) == (length(a.w.∇) + length(a.b.∇),)
        @test size(∇w) == (length(a.w.∇),)
        @test size(∇b) == (length(a.b.∇),)
        @test ∇w ≈ vec(repmat(input', nout, 1) .* repmat(dl, 1, nin))
        @test ∇b == dl
        @test a.input.∇ ≈ a.w.val' * dl

        θ = copy(a.θ)
        addgrad!(a, ∇, 1e-2)
        @show a.θ

        @test a.θ == θ + 1e-2∇
    end
end

# using Plots
# unicodeplots(size=(400,100))

let n=2, input=rand(n)
    for s in Transformations.activations
        f = Activation{s,Float64}(n)
        output = transform!(f, input)
        @test output == map(@eval($s), input)

        grad!(f, ones(2))
        @test f.input.∇ == map(@eval($(Symbol(s,"′"))), input)

        println()
        # plot(f, show=true)
        @show s f input output f.output.∇ f.input.∇
    end
end

let n1=4, n2=3, n3=2, input=rand(4)
    println()

    T = Float64
    chain = Chain(T,
        Affine{T}(n1, n2),
        Activation{:relu,T}(n2),
        Affine{T}(n2, n3),
        Activation{:logistic,T}(n3)
    )
    @show chain

    @test length(chain.ts) == 4
    @test chain.ts[1].input.val === chain.input.val
    @test chain.ts[end].output.val === chain.output.val

    # first compute the chain of transformations manually,
    manual_output = input
    for t in chain.ts
        manual_output = transform!(t, manual_output)
        @show t manual_output
    end

    # now do it using the chain, and make sure it matches
    output = transform!(chain, input)
    @show output
    @test manual_output == output
end

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


# # "link" this object into the transformations world
# Transformations.input_size(m::Means) = size(m.value)
# Transformations.output_size(m::Means) = size(m.value)
# Transformations.is_learnable(m::Means) = true
# Transformations.transform!(y, m::Means, x) = (y[:] = x - m.value)
# Transformations.learn!(m, x) = OnlineStats.fit!(m, x)
#
# # instantiate an OnlineStats.Means, which computes an online mean of a vector
# m = Means(5)
#
# # wrap the object in a Transformation, which stores input/output dimensions in its type,
# # and allows for the common functionality to apply to the Transformation object.
# t = transformation(m)
#
# # update/learn the parameters for this transformation
# #   (at this point, we don't care what the transformation is!)
# learn!(t, 1:5)
#
# # center the data by applying the transform. this dispatches generically.
# # we only need to define a single `transform!` method
# y = transform(t, 10ones(5))
#
# # make sure everything worked
# @test y == 10ones(5) - (1:5)
