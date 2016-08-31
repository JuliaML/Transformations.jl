
# part of a subgraph, representing input, output, or learnable parameters
type Node{TYPE,T,N}
    val::Array{T,N}  # value
    ∇::Array{T,N}  # gradient
end

function Node{T,N}(nodetype::Symbol, val::Array{T,N})
    Node{nodetype,T,N}(val, zeros(val))
end

Base.show{TYPE}(io::IO, node::Node{TYPE}) = print(io, "$TYPE$(size(node.val))")

# two nodes can be "linked together", which means that they are the "same node"
# from the perspective of the computational graph, even though one is an output
# of a transformation(s) and the other is the input to a transformation(s).
# this reduces memory requirements and unnecessary copying
function link_nodes!{T,N}(outnode::Node{:output,T,N}, innode::Node{:input,T,N})
    innode.val = outnode.val
    innode.∇ = outnode.∇
end
