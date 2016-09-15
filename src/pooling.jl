
immutable MaxPooling{T,N} <: Transformation
    input::Node{:input,T,N}
    output::Node{:output,T,N}
    stride::NTuple{N,Int}
end
MaxPooling(args...) = MaxPooling(Float64, args...)

function MaxPooling{T,N}(::Type{T}, sizein::NTuple{N,Int}, stride::NTuple{N,Int} = ntuple(i->2,N))
    sizeout = map(s->div(s,2), sizein)
    @show sizein, sizeout
    input = Node(:input,zeros(T,sizein))
    output = Node(:output,zeros(T,sizeout))
    MaxPooling(input, output, stride)
end

function transform!{T}(pool::MaxPooling{T,2})
    nr,nc = pool.sizeout
    rstride,cstride = pool.stride
    for r=1:nr, c=1:nc
        # one before the start
        rstart = (r-1)*rstride
        cstart = (c-1)*cstride

        tile = view(pool.input.val, rstart+1:rstart+rstride, cstart+1:cstart+cstride)
        pool.output.val[r,c] = max(tile)
    end
    value(output)
end

function grad!{T}(pool::MaxPooling{T,2})
    # TODO: pass the full gradient to the input with the maximum value in the tile
    #   note: we should really store this index during the forward pass... maybe have a matrix of indices??
end
