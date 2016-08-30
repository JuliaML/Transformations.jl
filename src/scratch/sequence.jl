immutable TransformSequence{N,T}
    pipeline::NTuple{N,T}
end
function transform!(transform::TransformSequence, x)
    for tfm in transform.pipeline
        transform!(tfm,x)
    end
end

function invert!(transform::TransformSequence, x)
    for tfm in reverse(transform.pipeline)
        !invertible(tfm) && error("transform not invertible")
    end
    for tfm in reverse(transform.pipeline)
        invert!(tfm,x)
    end
end
