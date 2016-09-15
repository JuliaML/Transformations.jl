
"A convolutional filter. One would normally fit many such filters as part of a convolutional layer."
immutable ConvFilter{T,N,P<:Params} <: Learnable
    sizein::NTuple{N,Int}
    sizeout::NTuple{N,Int}
    sizefilter::NTuple{N,Int}
    stride::NTuple{N,Int}
    input::Node{:input,T,N}
    output::Node{:output,T,N}
    params::P
end
ConvFilter(args...) = ConvFilter(Float64, args...)

function ConvFilter{T,N}(::Type{T}, sizein::NTuple{N,Int}, sizefilter::NTuple{N,Int}, stride::NTuple{N,Int} = (1,1))
    input = Node(:input, zeros(T, sizein...))
    nr_x,nc_x = sizein
    nr_f,nc_f = sizefilter
    sizeout = (nr_x-nr_f+1, nc_x-nc_f+1)
    output = Node(:output, zeros(T, sizeout...))
    nparams = prod(sizefilter) + 1
    params = Params(zeros(T, nparams), zeros(T, nparams), (sizefilter, (1,)))
    w, b = params.views
    initialize_weights!(w)
    initialize_bias!(b)
    ConvFilter(sizein, sizeout, sizefilter, stride, input, output, params)
end

# non-allocating sum(x .* y)
function summult{T}(x::AbstractMatrix{T}, y::AbstractMatrix{T})
    tot = zero(T)
    nr, nc = size(x)
    @assert size(x) == size(y)
    for c=1:nc, r=1:nr
        tot += x[r,c]*y[r,c]
    end
    tot
end

"allows reading out of bounds (returns `zero(T)`)"
immutable TileView{T,N,A<:AbstractArray} <: AbstractArray{T,N}
    a::A
    pos::NTuple{N,Int}
    sz::NTuple{N,Int}
end
Base.size(tile::TileView) = tile.sz
function Base.getindex{T}(tile::TileView{T,2}, i::Int, j::Int)
    ai = tile.pos[1] + i - 1
    aj = tile.pos[2] + j - 1
    if ai <= 0 || aj <= 0 || ai > size(tile.a,1) || aj > size(tile.a,2)
        zero(T)
    else
        tile.a[ai,aj]
    end
end



# specialized for matrix... TODO: make this a generated function
function transform!{T}(filter::ConvFilter{T,2})
    w, b = filter.params.views
    x = filter.input.val
    y = filter.output.val

    # start with the bias
    fill!(y, b[1])

    # now apply the filter, passing over the input
    nr_y, nc_y = filter.sizeout
    nr_f, nc_f = filter.sizefilter
    for r=1:nr_y, c=1:nc_y
        x_tile = view(x, r:r+nr_f-1, c:c+nc_f-1)
        # x_tile = TileView{T,2,typeof(x)}(x, (r,c), filter.sizefilter)
        y[r, c] = summult(w, x_tile) + b[1]
    end
    y
end

function grad!{T}(filter::ConvFilter{T,2})
    w, b = filter.params.views
    ∇w, ∇b = filter.params.∇_views
    x = filter.input.val
    ∇x = filter.input.∇
    ∇y = filter.output.∇

    # reset the gradients to 0
    fill!(∇x, zero(T))
    fill!(filter.params.∇, zero(T))

    nr_y, nc_y = filter.sizeout
    nr_f, nc_f = filter.sizefilter
    for ry=1:nr_y, cy=1:nc_y, i=1:nr_f, j=1:nc_f
        rx = ry+i-1
        cx = cy+j-1
        ∇x[rx,cx] += w[i,j] * ∇y[ry,cy]
        ∇w[i,j] += x[rx,cx] * ∇y[ry,cy]
    end

    ∇b[1] = sum(∇y)
    return
end

# ----------------------------------------------------------------------------

"Wraps several filters for a convolution layer"
immutable ConvLayer{T,N,P<:Params} <: Learnable
    sizein::NTuple{N,Int}
    sizeout::NTuple{N,Int}
    input::Node{:input,T,N}
    output::Node{:output,T,N}
    filters::Vector{ConvFilter{T,N,P}}
    params::P
end

function ConvLayer{T,N}(::Type{T}, sizein::NTuple{N,Int}, sizefilter::NTuple{N,Int}, numfilters::Int)
    filters = [ConvFilter(T, sizein, sizefilter) for i=1:numfilters]
    sizeout = filters[1].sizeout
    input = Node(:input, zeros(T,sizein))
    output = Node(:output, zeros(T,(sizeout..., numfilters)))
    params = consolidate_params(T, filters)

    # TODO: need a concatenating layer of some sort... hmmm
    # or maybe we need a "link" function that connects a view to part of the array?
    for filter in filters
        link_nodes!(input, filter.input)
        # link_nodes!(output, filter.output)
    end

    ConvLayer(sizein, sizeout, input, output, filters, params)
end

function transform!(conv::ConvLayer)
    foreach(transform!, conv.filters)
    value(output)
end

function grad!(conv::ConvLayer)
    foreach(grad!, conv.filters)
end
