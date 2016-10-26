# Modified from my (tbreloff) implementation of OnlinePCA in the OnlineStats package.
# That implementation had this note:
#       Roughly based on Weng et al (2003): "Candid covariance-free incremental principal component analysis"
#       used https://github.com/kevinhughes27/pyIPCA/blob/master/pyIPCA/ccipca.py as a reference

abstract PreprocessStep <: Learnable
immutable NoPreprocessing <: PreprocessStep end

# # solving for a dimension-reduced Y = XV', where X (n x nin) is the original data, and Y (n x nout) is projected
# # V (nout x nin) is a matrix where the columns are the first nout eigenvectors of X'X/n (the covariance of X)
# # e (nout x 1) is a vector of eigenvectors of the covariance matrix

# # We compute e and V by incrementally updating e and U, where U is a (nout x nin) matrix with the properties:
# #		Uᵢ = iᵗʰ column of U
# #		Vᵢ = iᵗʰ column of V
# #			 = iᵗʰ eigenvector of X-covariance
# #			 = Uᵢ / ‖Uᵢ‖
# #		eᵢ = iᵗʰ eigenvalue of X-covariance
# #		   = ‖Uᵢ‖
# # note: V is the normalized version of U


type Whiten{T,N,W<:Weight} <: PreprocessStep
    nin::Int        # number of input vars
    nout::Int       # number of principal components
    wgt::W
    U::Matrix{T}    # (nout x nin)
    V::Matrix{T}    # (nout x nin)
    e::Vector{T}    # (nout x 1)
    x̄::Vector{T}    # (nin x 1) mean(x)
    input::Node{:input,T,1}
    output::Node{:output,T,1}
 end

function Whiten{T}(::Type{T},
                    N::Int,
                    nin::Int,
                    nout::Int = nin;
                    lookback::Int = 100,
                    α::Float64 = NaN,
                    wgt::Weight =  BoundedEqualWeight(isnan(α) ? lookback : α)
                   )
    Whiten{T,N,typeof(wgt)}(
        nin,
        nout,
        wgt,
        zeros(T,nout,nin),
        zeros(T,nout,nin),
        zeros(T,nout),
        zeros(T,nin),
        Node(:input, zeros(T, nin)),
        Node(:output, zeros(T, nout))
    )
end

function Base.empty!{T}(o::Whiten{T})
    for a in (o.U,o.V,o.e,o.x̄)
        fill!(a, zero(T))
    end
    o.wgt.nobs = 0
    o
end

function smooth!{T,N}(a::AbstractArray{T,N}, b::AbstractArray{T,N}, λ::T)
    @assert length(a) == length(b)
    oml = one(λ) - λ
    @inbounds for i=1:length(a)
        a[i] = λ * b[i] + oml * a[i]
    end
end

function center!{T<:AbstractArray}(x::T, x̄::T)
    for i=1:length(x)
        x[i] -= x̄[i]
    end
end

# returns a vector z = Vx
# TODO: remove temporaries
function transform!{T}(o::Whiten{T})
    x = input_value(o)
    y = output_value(o)
    y[:] = o.V * (x - o.x̄)
    y
end

function learn!(o::Whiten)
    x = copy(input_value(o))
    center!(x, o.x̄)

    # get the smoothing param
    OnlineStats.updatecounter!(o.wgt)
	λ = weight(o.wgt)

    # update the mean
    smooth!(o.x̄, x, λ)

    #=
    TODO:
        - remove temporaries
        - handle ZCA whitening: for U'U = Σ⁻¹
            y_pca = D⁻¹/² * U'
            y_zca = U * y_pca
    =#

	@inbounds for i in 1:min(o.nout, o.wgt.nobs)
        # TODO this could be more robust?
		if o.e[i] == 0.
			# initialize ith principal component
			Uᵢ = x
			eᵢ = norm(Uᵢ)
			Vᵢ = Uᵢ / eᵢ
		else
			# update the ith eigvec/eigval
			Uᵢ = view(o.U, i, :)
			smooth!(Uᵢ, x * (dot(x, Uᵢ) / o.e[i]), λ)
			eᵢ = norm(Uᵢ)
			Vᵢ = Uᵢ / eᵢ

			# subtract projection on ith PC
			x -= dot(x, Vᵢ) * Vᵢ
		end

		# store these updates
		o.U[i,:] = Uᵢ
		o.V[i,:] = Vᵢ
		o.e[i] = eᵢ
	end
end

grad!(o::Whiten) = error("Can't backprop through Whiten")
