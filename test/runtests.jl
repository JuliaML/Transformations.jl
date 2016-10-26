using Transformations
using Base.Test

import LossFunctions: L2DistLoss
using Transformations.TestTransforms
using Distributions

@testset "Whiten" begin
    nin, nout = 4, 4
    μ = rand(nin)
    Λ = UpperTriangular(rand(nin,nin))
    Σ = Λ'*Λ
    mv = MultivariateNormal(μ, Σ)
    x = rand(mv, 1000)

    w = Whiten(Float64, 1, nin, nout)
    for i=1:10000
        input_value(w)[:] = x[:,rand(1:1000)]
        learn!(w)
        @show w,i
    end
    # y = ?
end

@testset "Distributions" begin
    n = 4
    μ = rand(n)
    σ = rand(n)
    t = MvNormalTransformation(μ, σ)
    @test typeof(t.dist.Σ) <: Distributions.PDiagMat
    @test t.n == n
    @test t.nμ == n
    @test t.nU == n
    @test input_length(t) == 2n
    @test output_length(t) == n
    newx = rand(2n)
    transform!(t, newx)
    grad!(t)
    @test t.dist.Σ.diag ≈ newx[n+1:end] .^ 2
    @test t.dist.Σ.inv_diag ≈ 1 ./ (newx[n+1:end] .^ 2)

    U = rand(n,n)
    t = MvNormalTransformation(μ, U*U')
    @test typeof(t.dist.μ) <: Vector{Float64}
    @test typeof(t.dist.Σ) <: Distributions.PDMat
    @test t.n == n
    @test t.nμ == n
    @test t.nU == div(n*(n+1), 2)
    @test input_length(t) == n + div(n*(n+1), 2)
    @test output_length(t) == n
    newx = rand(input_length(t))
    transform!(t, newx)
    grad!(t)
    @test t.dist.μ == newx[1:n]
    cf = t.dist.Σ.chol[:UL]
    @test cf[1,2] == newx[n+2]
    @test cf[end] == newx[end]
end

@testset "Affine" begin
    let nin=2, nout=3, input=rand(nin), target=rand(nout)
        a = Affine(nin, nout)
        loss = L2DistLoss()
        w, b = a.params.views
        x, y = input_value(a), output_value(a)
        ∇w, ∇b = a.params.∇_views
        ∇x, ∇y = input_grad(a), output_grad(a)
        nparams = nout*(nin+1)

        for i=1:2
            output = transform!(a, input)

            @test y == w * x + b
            @test size(x) == (nin,)
            @test size(w) == (nout,nin)
            @test size(b) == (nout,)
            @test size(y) == (nout,)
            @test size(∇x) == (nin,)
            @test size(∇w) == (nout,nin)
            @test size(∇b) == (nout,)
            @test size(∇y) == (nout,)
            @test size(output) == (nout,)
            @test size(params(a)) == (nparams,)
            @test size(grad(a)) == (nparams,)

            l = value(loss, target, output)
            dl = deriv(loss, target, output)
            grad!(a, dl)

            @test grad(a.output) == dl
            @test ∇w ≈ repmat(input', nout, 1) .* repmat(dl, 1, nin)
            @test ∇b == dl
            @test ∇x ≈ w' * dl
        end
    end
end

@testset "LayerNorm" begin
    let nin=2, nout=3, input=rand(nin), target=rand(nout)
        a = LayerNorm(nin, nout)
        loss = L2DistLoss()
        w, g, b = a.params.views
        x, y = input_value(a), output_value(a)
        ∇w, ∇g, ∇b = a.params.∇_views
        ∇x, ∇y = input_grad(a), output_grad(a)
        nparams = nout*(nin+2)

        for i=1:2
            output = transform!(a, input)

            wx = w * x
            @test y ≈ g .* (wx .- mean(wx)) ./ std(wx) .+ b
            @test size(x) == (nin,)
            @test size(w) == (nout,nin)
            @test size(g) == (nout,)
            @test size(b) == (nout,)
            @test size(y) == (nout,)
            @test size(∇x) == (nin,)
            @test size(∇w) == (nout,nin)
            @test size(∇g) == (nout,)
            @test size(∇b) == (nout,)
            @test size(∇y) == (nout,)
            @test size(output) == (nout,)
            @test size(params(a)) == (nparams,)
            @test size(grad(a)) == (nparams,)

            l = value(loss, target, output)
            dl = deriv(loss, target, output)
            grad!(a, dl)

            @test grad(a.output) == dl
            @test ∇w ≈ repmat(input', nout, 1) .* repmat(dl .* g, 1, nin)
            @test ∇g ≈ dl .* wx
            @test ∇b == dl
            @test ∇x ≈ w' * (dl .* g)
        end
    end
end

# using Plots
# unicodeplots(size=(400,100))

@testset "Activations" begin
    let n=2, input=rand(n)
        for s in Transformations.activations
            f = Activation{s,Float64}(n)
            output = transform!(f, input)
            @test output == map(@eval($s), input)

            grad!(f, ones(2))
            @test f.input.∇ == map(@eval($(Symbol(s,"′"))), input)

            # println()
            # plot(f, show=true)
            # @show s f input output f.output.∇ f.input.∇
        end
    end
end

@testset "Chain" begin
    let n1=4, n2=3, n3=2, input=rand(4)
        # println()

        T = Float64
        chain = Chain(T,
            Affine(T, n1, n2),
            Activation{:relu,T}(n2),
            Affine(T, n2, n3),
            Activation{:logistic,T}(n3)
        )
        # @show chain

        @test length(chain.ts) == 4
        @test chain.ts[1].input.val === chain.input.val
        @test chain.ts[end].output.val === chain.output.val

        # first compute the chain of transformations manually,
        manual_output = input
        for t in chain.ts
            manual_output = transform!(t, manual_output)
            # @show t manual_output
        end

        # now do it using the chain, and make sure it matches
        output = transform!(chain, input)
        # @show output
        @test manual_output == output
    end
end

@testset "ConvFilter" begin
    nin = (3,3)
    nfilter = (2,2)
    stride = (1,1)
    nrf, ncf = nfilter
    x = reshape(linspace(1,9,9), nin...)
    f = ConvFilter(nin, nfilter, stride)

    @test f.sizein == nin
    @test f.sizefilter == nfilter
    @test f.sizeout == (2,2)

    w, b = f.params.views
    b[1] = 0.5
    @test size(w) == nfilter
    @test size(b) == (1,)

    y = transform!(f, x)
    nr,nc = f.sizeout
    for r=1:nr, c=1:nc
        @test y[r,c] ≈ sum(w .* view(x, r:r+nrf-1, c:c+ncf-1)) + b[1]
    end

    ∇y = rand(f.sizeout...)
    grad!(f, ∇y)

    ∇x = input_grad(f)
    ∇w, ∇b = f.params.∇_views
    for i=1:nrf, j=1:ncf
        @test ∇w[i,j] ≈ sum(view(x, i:nr+i-1, j:nc+j-1) .* ∇y)
    end
    @test ∇b[1] == sum(∇y)
    for r=1:nin[1], c=1:nin[2]
        @test ∇x[r,c] ≈ sum(view(w, nrf:-1:1, ncf:-1:1) .*
            Transformations.TileView{Float64,2,typeof(∇y)}(∇y, (r-nrf+1,c-ncf+1), (nrf,ncf)))
    end

    # TODO: once stride is implemented, test that
end

@testset "Differentiable" begin
    f(θ) = sum(100θ.^2)
    df(θ) = map(θi -> 200θi, θ)
    t = tfunc(f, 2, df)
    θ = params(t)
    θ[:] = [1.,2.]
    @test transform!(t) == f(θ)
    grad!(t)
    @test grad(t) == df(θ)

    t = tfunc(rosenbrock, 4, rosenbrock_gradient)
    @test typeof(t) <: OnceDifferentiable{Float64}
    θ = params(t)
    @test size(θ) == (4,)
    θ[:] = linspace(0,3,4)
    @test transform!(t) == rosenbrock(θ)
    grad!(t)
    @test grad(t) == rosenbrock_gradient(θ)
end
