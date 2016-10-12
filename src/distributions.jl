
import Distributions: MvNormal, ZeroVector, PDiagMat, invcov

# NOTE: see http://qwone.com/~jason/writing/multivariateNormal.pdf
#   for derivations of gradients wrt μ and U:
#       ∇log P(z|ϕ)


"""
A transformation which is a MultivariateNormal generated using either of the equations:

dist = N(μ, σ²)  where σ is a vector of standard deviations
dist = N(μ, Σ)   where U is an upper-triangular matrix with covariance Σ = U'U

The output is sampled from this distribution: z ~ dist

`transform!` will first update the underlying distribution with input vector ϕ = (μ, U)
and then call `rand!(dist, output)`
"""
type MvNormalTransformation{T,DIST<:MvNormal} <: Transformation
    dist::DIST
    n::Int
    nμ::Int
    nU::Int
    z̄::Vector{T}         # scratch space to avoid allocation in grad!
    input::Node{:input,T,1}    # the sufficient stats: ϕ = vec(μ, U)
    output::Node{:output,T,1}  # the random sample
end

function MvNormalTransformation{T}(::Type{T}, args...)
    dist = MvNormal(args...)
    n = length(dist.μ)

    # do we need inputs for μ?
    nμ = isa(dist.μ, ZeroVector) ? 0 : n

    # do we have a diagonal or upper-triangular input for Σ?
    nU = typeof(dist.Σ) <: PDiagMat ? n : div(n*(n+1), 2)

    MvNormalTransformation{T,typeof(dist)}(
        dist,
        n,
        nμ,
        nU,
        zeros(T, n),
        Node(:input, zeros(T, nμ+nU)),
        Node(:output, zeros(T, n))
    )
end
MvNormalTransformation(args...) = MvNormalTransformation(Float64, args...)

# update the distribution params, then randomly sample into the output node
function transform!(mv::MvNormalTransformation)
    # update the distribution using ϕ
    transform_μ!(mv, mv.dist.μ)
    transform_Σ!(mv, mv.dist.Σ)

    # output: sample from the distribution
    rand!(mv.dist, output_value(mv))
end

# # update the gradient w.r.t. inputs ϕ = vec(μ, U)
# function grad!(mv::MvNormalTransformation)
#     Σ⁻¹ = grad_Σ!(mv, mv.dist.Σ)
#     grad_μ!(mv, mv.dist.μ, Σ⁻¹)
# end

# -----------------------------------------------------------------------
# these methods update the underlying storage of the distribution
# using the input vector

# don't do anything for zeros
function transform_μ!(mv::MvNormalTransformation, μ::ZeroVector)
    return
end

# just copy
function transform_μ!(mv::MvNormalTransformation, μ)
    copy!(μ, view(input_value(mv), 1:mv.n))
    return
end

# diagonal Σ = diagmat(σ²)
function transform_Σ!(mv::MvNormalTransformation, Σ::PDiagMat)
    input = input_value(mv)
    for i=1:mv.n
        xi = input[mv.nμ + i]
        Σ.diag[i] = xi
        Σ.inv_diag[i] = inv(xi)
    end
end

# Σ = U'U   (where U is assumed upper-triangular)
# To accomplish, we update the upper triangle of the cholesky decomp of dist.Σ
function transform_Σ!(mv::MvNormalTransformation, Σ)
    input = input_value(mv)
    cf = Σ.chol.factors
    i = 1
    for r=1:mv.n, c=r:mv.n
        cf[r,c] = input[mv.nμ + i]
        i += 1
    end
end

# -----------------------------------------------------------------------
# these methods compute the gradients w.r.t. input ϕ using the
# underlying storage of the distribution

    # ϕ = mv.input.val
    # ∇ϕ = mv.input.∇
    # z = mv.output.val
    # ∇z = mv.output.∇

# # don't do anything for zeros
# function grad_μ!(mv::MvNormalTransformation, μ::ZeroVector, Σ⁻¹)
#     return
# end
#
# # just copy
# function grad_μ!{T}(mv::MvNormalTransformation{T}, μ, Σ⁻¹)
#     ∇ϕ = input_grad(mv)
#     z = output_value(mv)
#     ∇z = output_grad(mv)
#     for i=1:mv.n
#         ∇ϕᵢ = zero(T)
#         for j=1:mv.n
#             ∇ϕᵢ -= T(2) * (z[j] - μ[j]) * Σ⁻¹[j,i] * ∇z[i]
#         end
#         ∇ϕ[i] = ∇ϕᵢ
#     end
# end

# diagonal Σ = diagmat(σ²)
# function grad_Σ!(mv::MvNormalTransformation, Σ::PDiagMat)

# function compute_Σ⁻¹(Σ::PDiagMat)
#     Diagonal(Σ.inv_diag)
# end
#
# function compute_Σ⁻¹(Σ)
#     U = Σ.chol.factors
#     @show typeof(U)
#     U⁻¹ = inv(U)
#     Diagonal(Σ.inv_diag)
# end


# function gradlogprob!(mv::MvNormalTransformation, μ, Σ)
#     ϕ = input_value(mv)
#     ∇ϕ = input_grad(mv)
#     z = output_value(mv)
#     ∇z = output_grad(mv)
#
#     # update: z̄ = (z - μ)
#     z̄ = mv.z̄
#     copy!(z̄, z)
#     for i=1:mv.nμ
#         z̄[i] -= ϕ[i]
#     end
#
# end


# NOTE: this computes the "grad log prob": ∇log P(z | ϕ)
#   and thus it will not backprop gradients.
#   It is meant to be used only for policy gradient methods for now!
function grad!{T}(mv::MvNormalTransformation{T})
    ϕ = input_value(mv)
    ∇ϕ = input_grad(mv)
    z = output_value(mv)
    ∇z = output_grad(mv)
    z̄ = mv.z̄
    nμ = mv.nμ

    # update: z̄ = (z - μ)
    copy!(z̄, z)
    for i=1:nμ
        z̄[i] -= ϕ[i]
    end
    scalar = T(2) - T(2) * sqrt(norm(z̄))

    if typeof(mv.dist.Σ) <: PDiagMat
        # do update for diagonal
        # note: diag(U) .* diag(Σ⁻¹) == 1 ./ diag(U)
        for i=1:mv.n
            # Σ⁻¹ᵢᵢ = Σ⁻¹[i,i]
            if i <= nμ
                ∇ϕ[i] = -T(2) *  mv.dist.Σ.inv_diag[i] * z̄[i]
            end
            ∇ϕ[nμ+i] = scalar / ϕ[nμ+i]
        end
    else
        # do update for upper-triangular
        Σ⁻¹ = invcov(mv.dist)
        U = UpperTriangular(mv.dist.Σ.chol.factors)
        # U⁻¹ = inv(U)
        # Σ⁻¹ =
        @show U Σ⁻¹ typeof(Σ⁻¹)

        # compute gradient of μ: ∇μ = -2 Σ⁻¹ (z-μ)
        if nμ > 0
            A_mul_B!(view(∇ϕ, 1:nμ), Σ⁻¹, z̄)
        end

        # compute gradient of U: ∇U = (2-2√‖z-μ‖) U Σ⁻¹
        i = 1
        for r=1:mv.n, c=r:mv.n
            ∇ϕᵢ = zero(T)
            for j=c:mv.n
                ∇ϕᵢ += U[r,j] * Σ⁻¹[j,c]
            end
            ∇ϕ[nμ+i] = scalar * ∇ϕᵢ
            i += 1
        end
    end

    @show mv.n nμ ϕ ∇ϕ z ∇z z̄ scalar


    # error()

    # U = UpperTriangular(mv.dist.Σ)
    #
    # for i=1:mv.n
    #     xi = ϕ[nμ + i]
    #     Σ.diag[i] = xi
    #     Σ.inv_diag[i] = inv(xi)
    # end
    #
    # # return Σ⁻¹
    # Diagonal(Σ.inv_diag)
end

# # Σ = U'U   (where U is assumed upper-triangular)
# # To accomplish, we update the upper triangle of the cholesky decomp of dist.Σ
# function grad_Σ!(mv::MvNormalTransformation, Σ)
#     ϕ = input_value(mv)
#     U = Σ.chol.factors
#     i = 1
#     for r=1:mv.n, c=r:mv.n
#         U[r,c] = ϕ[mv.nμ + i]
#         i += 1
#     end
# end
