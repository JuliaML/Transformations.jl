
# part of a subgraph, representing input, output, or learnable parameters
type Node{TYPE,T,N}
    val::Array{T,N}  # value
    ∇::Array{T,N}  # gradient
end

function Node{T,N}(nodetype::Symbol, val::Array{T,N})
    Node{nodetype,T,N}(val, zeros(val))
end

function Node(nodetype::Symbol, n::Int...)
    Node{nodetype,Float64,length(n)}(zeros(n...), zeros(n...))
end

function Node{T}(::Type{T}, nodetype::Symbol, n::Int...)
    Node{nodetype,T,length(n)}(zeros(T,n...), zeros(T,n...))
end

Base.show{TYPE}(io::IO, node::Node{TYPE}) = print(io, "$TYPE$(size(node.val))")

Base.length(node::Node) = length(value(node))
value(node::Node) = node.val
grad(node::Node) = node.∇

# two nodes can be "linked together", which means that they are the "same node"
# from the perspective of the computational graph, even though one is an output
# of a transformation(s) and the other is the input to a transformation(s).
# this reduces memory requirements and unnecessary copying
function link_nodes!{TYPE1,TYPE2,T,N}(outnode::Node{TYPE1,T,N}, innode::Node{TYPE2,T,N})
    innode.val = outnode.val
    innode.∇ = outnode.∇
end
