
abstract Node{T,N}

Base.size(node::Node) = size(node.val)
Base.length(node::Node) = length(node.val)
value(node::Node) = node.val
grad(node::Node) = node.∇

# ----------------------------------------------------------

"The output from a Transformation.  Can project to zero to many InputNodes"
type OutputNode{T,N} <: Node{T,N}
    tonodes::Vector{Node}
    val::Array{T,N}
    ∇::Array{T,N}
end
OutputNode(dims::Int...) = OutputNode(Float64, dims...)
function OutputNode{T}(::Type{T}, dims::Int...)
    OutputNode{T,length(dims)}(Node[], zeros(T, dims...), zeros(T, dims...))
end

# backprop gradients from linked nodes
function grad!{T}(fromnode::OutputNode{T})
    ∇ = grad(fromnode)
    fill!(∇, zero(T))
    for tonode in fromnode.tonodes
        # ∇ .+= tonode.∇
        backward!(fromnode, tonode)
    end
    ∇
end

# add out∇ to the backprop gradient
function grad!{T,N}(fromnode::OutputNode{T,N}, out∇::AbstractArray{T,N})
    ∇ = grad!(fromnode)
    ∇ .+= out∇
    ∇
end

# ----------------------------------------------------------

"""
The input to a transformation.  Can aggregate from zero to many OutputNodes.

For example a SumNode has op `:+`, and so the values of the linked nodes are summed as input.
"""
immutable InputNode{OP,T,N} <: Node{T,N}
    fromnodes::Vector{Node}
    val::Array{T,N}
    ∇::Array{T,N}
    offset::Dict{OutputNode{T,N},Int}  # used for starting index offset of concatenation
end
InputNode(dims::Int...) = InputNode(:+, Float64, dims...)
InputNode{T}(::Type{T}, dims::Int...) = InputNode(:+, T, dims...)
InputNode(op::Symbol, dims::Int...) = InputNode(op, Float64, dims...)

# the real constructor
function InputNode{T}(op::Symbol, ::Type{T}, dims::Int...)
    InputNode{op,T,length(dims)}(Node[], zeros(T, dims...), zeros(T, dims...), Dict{OutputNode{T,length(dims)},Int}())
end

# forward pass, aggregate output values from OutputNodes
function transform!(tonode::InputNode)
    reset_val!(tonode)
    for fromnode in tonode.fromnodes
        forward!(fromnode, tonode)
    end
    tonode.val
end

# aggregate inval along with the normal forward pass
function transform!{OP,T,N}(tonode::InputNode{OP,T,N}, inval::AbstractArray{T,N})
    transform!(tonode)
    forward!(inval, tonode)
    tonode.val
end

function forward!(fromnode::OutputNode, tonode::InputNode)
    forward!(fromnode.val, tonode)
end

function link_nodes!(fromnode::OutputNode, tonode::InputNode)
    push!(fromnode.tonodes, tonode)
    push!(tonode.fromnodes, fromnode)
    nodes_linked(fromnode, tonode)
    return
end

# a callback when nodes are linked
nodes_linked(fromnode::OutputNode, tonode::InputNode) = return

# ----------------------------------------------------------

# elementwise sum
typealias SumNode{T,N} InputNode{:+,T,N}
reset_val!{T}(tonode::SumNode{T}) = fill!(tonode.val, zero(T))
forward!{T,N}(val::AbstractArray{T,N}, tonode::SumNode{T,N}) = (tonode.val .+= val)
backward!{T}(fromnode::OutputNode{T}, tonode::SumNode{T}) = (fromnode.∇ .+= tonode.∇)

# ----------------------------------------------------------

# elementwise product
typealias ProdNode{T,N} InputNode{:*,T,N}
reset_val!{T}(tonode::ProdNode{T}) = fill!(tonode.val, one(T))
forward!{T,N}(val::AbstractArray{T,N}, tonode::ProdNode{T,N}) = (tonode.val .*= val)
function backward!{T}(fromnode::OutputNode{T}, tonode::ProdNode{T})
    fromnode.∇ .+= tonode.∇ .* tonode.val ./ fromnode.val
end

# ----------------------------------------------------------

# concatenation
typealias CatNode{T,N} InputNode{:cat,T,N}
function nodes_linked{T,N}(fromnode::OutputNode{T,N}, tonode::CatNode{T,N})
    tonode.offset[fromnode] = isempty(tonode.offset) ? 0 : sum(node->length(node.val), keys(tonode.offset))
end
reset_val!(tonode::CatNode) = return
forward!{T,N}(val::AbstractArray{T,N}, tonode::CatNode{T,N}) = (tonode.val .+= val)
function forward!{T,N}(fromnode::OutputNode{T,N}, tonode::CatNode{T,N})
    offset = tonode.offset[fromnode]
    tonode.val[offset+1:offset+length(fromnode.val)] = fromnode.val
end
function backward!{T,N}(fromnode::OutputNode{T,N}, tonode::CatNode{T,N})
    offset = tonode.offset[fromnode]
    fromnode.∇ .+= view(tonode.∇, offset+1:offset+length(fromnode.∇))
end

# ----------------------------------------------------------

# # part of a subgraph, representing input, output, or learnable parameters
# type Node{TYPE,T,N}
#     val::Array{T,N}  # value
#     ∇::Array{T,N}  # gradient
# end
#
# function Node{T,N}(nodetype::Symbol, val::Array{T,N})
#     Node{nodetype,T,N}(val, zeros(val))
# end
#
# function Node(nodetype::Symbol, n::Int...)
#     Node{nodetype,Float64,length(n)}(zeros(n...), zeros(n...))
# end
#
# function Node{T}(::Type{T}, nodetype::Symbol, n::Int...)
#     Node{nodetype,T,length(n)}(zeros(T,n...), zeros(T,n...))
# end
#
# Base.show{TYPE}(io::IO, node::Node{TYPE}) = print(io, "$TYPE$(size(node.val))")
#
# Base.length(node::Node) = length(value(node))
# value(node::Node) = node.val
# grad(node::Node) = node.∇
#
# # two nodes can be "linked together", which means that they are the "same node"
# # from the perspective of the computational graph, even though one is an output
# # of a transformation(s) and the other is the input to a transformation(s).
# # this reduces memory requirements and unnecessary copying
# function link_nodes!{TYPE1,TYPE2,T,N}(outnode::Node{TYPE1,T,N}, innode::Node{TYPE2,T,N})
#     innode.val = outnode.val
#     innode.∇ = outnode.∇
# end


function link_nodes!(ts::Transformation...)
    for i=2:length(ts)
        link_nodes!(output_node(ts[i-1]), input_node(ts[i]))
    end
end

function link_nodes!{T<:Transformation}(ts::AbstractVector{T})
    for i=2:length(ts)
        link_nodes!(output_node(ts[i-1]), input_node(ts[i]))
    end
end
