
import Distributions: MvNormal, ZeroVector, PDiagMat

# NOTE: see http://qwone.com/~jason/writing/multivariateNormal.pdf
#   for derivations of gradients wrt μ and Λ:
#       ∇log P(z|ϕ)


"""
A transformation which is a MultivariateNormal generated using either of the equations:

dist = N(μ, σ²)   when σ is a vector of standard deviations
dist = N(μ, ΛΛ')  when Λ is an upper-triangular matrix with covariance Σ = ΛΛ'

`transform!` will first update the underlying distribution with input vector ϕ = (μ, Λ)
and then call `rand!(dist, output)`
"""
type MvNormalTransformation{T,DIST<:MvNormal} <: Transformation
    dist::DIST
    n::Int
    nμ::Int
    nΛ::Int
    input::Node{:input,T,1}    # the sufficient stats: ϕ = vec(μ, Λ)
    output::Node{:output,T,1}  # the random sample
end

function MvNormalTransformation{T}(::Type{T}, args...)
    dist = MvNormal(args...)
    n = length(dist.μ)

    # do we need inputs for μ?
    nμ = isa(dist.μ, ZeroVector) ? 0 : n

    # do we have a diagonal or upper-triangular input for Σ?
    nΛ = typeof(dist.Σ) <: PDiagMat ? n : div(n*(n+1), 2)

    MvNormalTransformation{T,typeof(dist)}(
        dist,
        n,
        nμ,
        nΛ,
        Node(:input, zeros(T, nμ+nΛ)),
        Node(:output, zeros(T, n))
    )
end
MvNormalTransformation(args...) = MvNormalTransformation(Float64, args...)

# update the distribution params, then randomly sample into the output node
function transform!(mv::MvNormalTransformation)
    input = input_value(mv)
    n = output_length(mv)
    transform_μ!(mv, mv.dist.μ)
    transform_Σ!(mv, mv.dist.Σ)
    rand!(mv.dist, output_value(mv))
end

# TODO: grad!

# -----------------------------------------------------------------------

# copy, or not
function transform_μ!(mv::MvNormalTransformation, μ)
    copy!(μ, view(input_value(mv), 1:mv.n))
    return
end
transform_μ!(mv::MvNormalTransformation, μ::ZeroVector) = return

# diagonal Σ = diagmat(σ²)
function transform_Σ!(mv::MvNormalTransformation, Σ::PDiagMat)
    input = input_value(mv)
    for i=1:mv.n
        xi = input[mv.nμ + i]
        Σ.diag[i] = xi
        Σ.inv_diag[i] = inv(xi)
    end
end

# Σ = ΛΛ'   (where Λ is assumed upper-triangular)
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
