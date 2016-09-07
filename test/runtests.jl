using Transformations
using Base.Test
# using Distributions
# import OnlineStats: Means

using Losses

@testset "Affine" begin
    let nin=2, nout=3, input=rand(nin), target=rand(nout)
        a = Affine{Float64}(nin, nout)
        loss = L2DistLoss()
        # @show a loss

        for i=1:2
            # println()

            output = transform!(a, input)
            # @show i output

            @test output == a.w.val * input + a.b.val
            @test size(a.input.val) == (nin,)
            @test size(a.w.val) == (nout,nin)
            @test size(a.b.val) == (nout,)
            @test size(a.output.val) == (nout,)
            @test size(output) == (nout,)

            l = value(loss, target, output)
            dl = deriv(loss, target, output)
            # @show l dl

            ∇ = grad!(a, dl)
            ∇w = ∇[1:(nin*nout)]
            ∇b = ∇[(nin*nout+1):end]
            # @show ∇ a.w.∇ a.b.∇

            @test a.output.∇ == dl
            @test isa(∇, Transformations.CatView)
            @test size(∇) == (length(a.w.∇) + length(a.b.∇),)
            @test size(∇w) == (length(a.w.∇),)
            @test size(∇b) == (length(a.b.∇),)
            @test ∇w ≈ vec(repmat(input', nout, 1) .* repmat(dl, 1, nin))
            @test ∇b == dl
            @test a.input.∇ ≈ a.w.val' * dl

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
