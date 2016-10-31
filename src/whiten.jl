# Modified from my (tbreloff) implementation of OnlinePCA in the OnlineStats package.
# That implementation had this note:
#       Roughly based on Weng et al (2003): "Candid covariance-free incremental principal component analysis"
#       used https://github.com/kevinhughes27/pyIPCA/blob/master/pyIPCA/ccipca.py as a reference

abstract PreprocessStep <: Learnable
immutable NoPreprocessing <: PreprocessStep end

#=
Solving for a dimension-reduced Y = VX, where X (nin x nobs) is the original data,
and Y (nout x nobs) is projected. V (nout x nin) is a matrix where the columns are the first
nout eigenvectors of Σ = XX'/n (the covariance of X).
e (nout x 1) is a vector of eigenvectors of the covariance matrix.

We compute e and V by incrementally updating e and U, where U is a (nout x nin) matrix with the properties:
		Uᵢ = iᵗʰ column of U
		Vᵢ = iᵗʰ column of V
			 = iᵗʰ eigenvector of X-covariance
			 = Uᵢ / ‖Uᵢ‖
		eᵢ = iᵗʰ eigenvalue of X-covariance
		   = ‖Uᵢ‖
note: V is the normalized version of U
=#


type Whiten{T,METHOD,W<:Weight} <: PreprocessStep
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
 Whiten(nin::Int, nout::Int; kw...) = Whiten(Float64, nin, nout; kw...)

function Whiten{T}(::Type{T},
                    nin::Int,
                    nout::Int = nin;
                    method::Symbol = :zca,  # choose from pca, whitened_pca, zca
                    lookback::Int = 100,
                    α::Float64 = NaN,
                    wgt::Weight =  BoundedEqualWeight(isnan(α) ? lookback : α)
                   )
    @assert method in (:pca, :whitened_pca, :zca)
    if method == :zca && nin != nout
        error("ZCA whitening requires nin==nout.  Got: nin=$nin, nout=$nout")
    end
    Whiten{T,method,typeof(wgt)}(
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

function center!{T<:AbstractArray}(x::T, x̄::T)
    for i=1:length(x)
        x[i] -= x̄[i]
    end
end

function learn!{T}(o::Whiten{T}, x::AbstractVector = input_value(o))
    # get the smoothing param
    OnlineStats.updatecounter!(o.wgt)
	λ = weight(o.wgt)

    # update the mean
    smooth!(o.x̄, x, λ)

    # copy and center x
    x = x - o.x̄
    # center!(x, o.x̄)

    #=
    TODO:
        - remove temporaries
        - handle ZCA whitening: for V'V = Σ⁻¹, D = diag(e)
            y_pca = D⁻¹/² * V'
            y_zca = V * y_pca
    =#

    n = min(o.nout, o.wgt.nobs)
	@inbounds for i in 1:n
        # TODO this could be more robust?
		if o.e[i] == zero(T)
			# initialize ith principal component
			Uᵢ = x
		else
			# update the ith eigvec/eigval
			Uᵢ = view(o.U, i, :)
			smooth!(Uᵢ, x * (dot(x, Uᵢ) / o.e[i]), λ)
        end
		eᵢ = norm(Uᵢ)
        # if eᵢ ≈ zero(T)
        #     warn("Got small e[$i] ($(eᵢ)) while computing iteration $(o.wgt.nobs) for:\n  $o")
        # end
		Vᵢ = Uᵢ / max(eᵢ, T(1e-3))

		# store these updates
		o.U[i,:] = Uᵢ
		o.V[i,:] = Vᵢ
		o.e[i] = eᵢ

        if i < n
    		# subtract projection on ith PC for the next step
    		x -= dot(x, Vᵢ) * Vᵢ
        end
	end
end

grad!(o::Whiten) = error("Can't backprop through Whiten")

# -------------------------------------------------------------------
# transform defs

# TODO: cut down on temporaries

function normal_pca{T}(o::Whiten{T})
    x = input_value(o) - o.x̄
    y = output_value(o)
    y[:] = o.V * x
    y
end

function whitened_pca{T}(o::Whiten{T})
    y = normal_pca(o)
    y[:] = y ./ sqrt.(o.e .+ T(1e-1))
    y
end

function zca{T}(o::Whiten{T})
    y = whitened_pca(o)
    y[:] = o.V' * y
    y
end

# "normal" pca... should match MultivariateStats.fit(PCA, x, maxdimout=k)
#   y = V (x - mean(x))
transform!{T}(o::Whiten{T,:pca}) = normal_pca(o)

# whitened pca... multiply by D⁻¹/², where D = diag(e) is a diagonal matrix
# with the eigenvalues e on the diagonal
#   yᵢ = D⁻¹/² V (x - mean(x))
transform!{T}(o::Whiten{T,:whitened_pca}) = whitened_pca(o)


# whitened zca... multiply by (V' * D⁻¹/²), where D = diag(e) is a diagonal matrix
# with the eigenvalues e on the diagonal.
# Note: ZCA is similar to whitened PCA, except that it rotates the whitened projection
#       back into "input space".  If you want whitening while maintaining the directions
#       of the inputs, this is the one to use.
#   yᵢ = D⁻¹/² V (x - mean(x))
transform!{T}(o::Whiten{T,:zca}) = zca(o)
