using Transformations
using Base.Test
# using Distributions
# import OnlineStats: Means

using Losses

@testset "Affine" begin
    let nin=2, nout=3, input=rand(nin), target=rand(nout)
        a = Affine(nin, nout)
        loss = L2DistLoss()
        w, b = a.params.views
        x, y = input_value(a), output_value(a)
        ∇w, ∇b = a.params.∇_views
        ∇x, ∇y = input_grad(a), output_grad(a)
        nparams = nout*(nin+1)
        # @show a loss

        for i=1:2
            # println()

            output = transform!(a, input)
            # @show i output

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
            # @show l dl

            grad!(a, dl)
            # ∇w = ∇[1:(nin*nout)]
            # ∇b = ∇[(nin*nout+1):end]
            # @show ∇ a.w.∇ a.b.∇

            @test grad(a.output) == dl
            # @test isa(∇, Transformations.CatView)
            # @test size(∇) == (length(a.w.∇) + length(a.b.∇),)
            @test ∇w ≈ repmat(input', nout, 1) .* repmat(dl, 1, nin)
            @test ∇b == dl
            @test ∇x ≈ w' * dl

            # θ = copy(a.θ)
            # addgrad!(a, ∇, 1e-2)
            # @show a.θ
            #
            # @test a.θ == θ + 1e-2∇
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
            Affine{T}(n1, n2),
            Activation{:relu,T}(n2),
            Affine{T}(n2, n3),
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
