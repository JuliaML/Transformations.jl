
export
    Operator,
    OpGraph,
    InputNode,
    OutputNode,
    Learnable,
    AbstractTransformation,
    op,
    @op

"""
The idea here is that an Operator wraps an arbitrary function, and there are static query methods
which define how to apply the function to input(s) and how to update the output.

In addition, we should be able to define (as part of the type, and thus at compile time) inversion
functions, deriviatives, etc.

some core traits:
- is_mappable: one input, function applied elementwise, input size matches output size
    examples: sin, exp, log, sigmoid, tanh
- is_operator: two (or more?) inputs, output dimension/size is function of inputs
    examples: * (matrix multiply)
- is_element_operator: two (or more?) inputs, applied elementwise, input sizes match output size
    examples: +, -, .*, ./, max
- is_aggregator: one input, output dimension is less than input dimension
    examples: sum, product
"""

# wrap a function call (static transformation) and input/output dimensions in a type signature
immutable Operator{F,I,O} <: AbstractTransformation{I,O}
    f::Function
end

# convenience to instantiate an Operator
op(f::Function, I::Int, O::Int) = Operator{Symbol(f), I, O}(f)
op(f::Symbol, I::Int, O::Int) = Operator{f, I, O}(eval(f))

function Base.show{F,I,O}(io::IO, o::Operator{F,I,O})
    print(io, "Op{$F, $I, $O")
    for f in (:is_mappable, :is_operator, :is_element_operator)
        print(io, ", $f=$(eval(f)(o))")
    end
    print(io, "}")
end

const _mappables = [:sin, :exp, :log, :sigmoid, :tanh]
const _operators = [:*, :/, :\]
const _element_ops = [:+, :-, :.*, :./, :max, :min]
const _aggregators = [:sum, :product, :minimum, :maximum]

function is_op(s::Symbol)
    for v in (_mappables, _operators, _element_ops, _aggregators)
        if s in v
            return true
        end
    end
    false
end
# const _all_ops = vcat(_mappables, _operators, _element_ops, _aggregators)

@generated function is_mappable{F,I,O}(o::Operator{F,I,O})
    if F in _mappables
        I==O || error("Mappables input and output dimensions should match")
        :(true)
    else
        :(false)
    end
end
@generated function is_operator{F,I,O}(o::Operator{F,I,O})
    F in _operators ? :(true) : :(false)
end
@generated function is_element_operator{F,I,O}(o::Operator{F,I,O})
    F in _element_ops ? :(true) : :(false)
end
@generated function is_aggregator{F,I,O}(o::Operator{F,I,O})
    F in _aggregators ? :(true) : :(false)
end

# -------------------------------------------------------

# input and output nodes allow for easy connectivity
immutable InputNode{I} <: AbstractTransformation{I,I} end
immutable OutputNode{O} <: AbstractTransformation{O,O} end

# a Learnable has no inputs, but produces an output (the parameters)
immutable Learnable{O} <: AbstractTransformation{0,O}
    name::Symbol
end

# an OpGraph is a lightweight representation of a directed graph of AbstractTransformations
type OpGraph{I,O} <: AbstractTransformation{I,O}
    nodes::Vector{AbstractTransformation}
    edges::Vector{NTuple{2,Int}}
    names::Vector{String}
end


# -------------------------------------------------------


"""
We want to build an op graph, where each node has input(s) and output(s)

```
# single input, single output
# w and b are assumed 'learnable parameters'
@op affine(x) = x * w + b

# single input, single output
# no learnable parameters
@op sigmoid(x) = 1 / (1 + exp(-x))

# this shouldn't need a special definiton... use the built-in method
# no learnable parameters
@op tanh(x) = tanh(x)

# 2 inputs, 2 outputs
w is a learnable parameter, and can be fused to avoid recomputing
@op f(x, y) = (w * (x-y), w * (x+y))
```

The point of this design is that we should be able to define simple functions
composed of previously defined components, and the `@op` macro will build
a connected graph of transformations.  Some notes:

- If all components and operations inside an op graph are differentiable, then so is the graph
    - macro should build `deriv` method (and others in a similar way)
- If a variable appears in the body, but not in the function signature, we assume it
    is a `LearnableParameter`, and include it (with gradient state objects) in the internals
    of the op graph (or its sub-components).
"""

const _input_index = 1
const _output_index = 2

# NOTE: in the add_item_to_graph methods, we want to:
#   - increase the numnodes
#   - call add_edge! for each connection

# remember that numnodes is also the index of the last-added node

function add_item_to_graph!(block::Expr, input_idx::Int, variables, numnodes::Base.RefValue{Int}, item::Expr, opidx)
    @show "EXPR",item input_idx variables numnodes[]
    dump(item, 20)
    # TODO: handle the expression
    # item is expected to be an "operation"
    func = shift!(item.args)
    if item.head == :call && is_op(func)
        # node_indices = Int[]

        add_node!(block, numnodes, func, :operation)
        for x in item.args
            add_item_to_graph!(block, input_idx, variables, numnodes, x, numnodes[])

            # # the node just added is an input to the op
            # if !(x in variables)
            #     push!(node_indices, numnodes[])
            # end
        end

        # now add the op
        # add_node!(block, numnodes, func, :operation)
        # for idx in node_indices
        #     add_edge!(block, idx, numnodes[])
        # end

    else
        if item.head == :line
            # if it's a line number block, pass through to wrapped arg
            # add_item_to_graph!(block, input_idx, variables, numnodes, item.args[1])
            info("line")
        else
            error("OpGraph parse error: $item")
        end
    end
end

function add_item_to_graph!(block::Expr, input_idx::Int, variables, numnodes::Base.RefValue{Int}, item::Symbol, opidx)
    @show "SYM",item input_idx variables numnodes[]
    if item in variables
        # TODO: this is a Variable... connect input node to this
        # add_node!(block, numnodes, item, :variable)
        add_edge!(block, _input_index, opidx)
    else
        # TODO: this is a Learnable
        add_node!(block, numnodes, item, :learnable)
        add_edge!(block, numnodes[], opidx)
    end
    return
end

function add_node!(block::Expr, numnodes, node, nodetype::Symbol)
    nodesym = QuoteNode(node)
    nodeexpr = if nodetype == :learnable
        :(Learnable{1}($nodesym))
    # elseif nodetype == :variable

    elseif nodetype == :operation
        :(op($nodesym, 1, 1))
    else
        error("how do I add this node? $node $nodetype")
    end
    # push!(block.args, esc(:(push!(g.nodes, ($(QuoteNode(node)), $(QuoteNode(nodetype)))))))
    push!(block.args, esc(:(push!(g.nodes, $nodeexpr))))
    push!(block.args, esc(:(push!(g.names, string($nodesym)))))
    numnodes[] += 1
end

function add_edge!(block::Expr, i::Int, j::Int)
    push!(block.args, esc(:(push!(g.edges, ($i, $j)))))
end

function _op_macro(funcexpr::Expr, inout::NTuple{2,Int} = (1,1))
    dump(funcexpr, 20)
    @show inout typeof(inout)

    func_signature, func_body = funcexpr.args

    if !(funcexpr.head in (:(=), :function))
        error("Must wrap a valid function call!")
    end
    if !(isa(func_signature, Expr) && func_signature.head == :call)
        error("Expected `func_signature = ...` with func_signature as a call Expr... got: $func_signature")
    end

    func_name = func_signature.args[1]
    variables = func_signature.args[2:end]
    @show func_name variables

    # TODO: we want to build an OpGraph as a function of Variable, Learnable, and Op components
    # To simplify, we'll assume that:
    #   - Expr --> Op
    #   - Symbol --> haskey(_ops, k) ? Variable : Learnable

    # Variables are not learnable, and they are NOT part of the graph.  They represent the inputs
    # to the OpGraph.

    I,O = inout

    block = Expr(:block)
    push!(block.args, esc(:(
        g = OpGraph{$I, $O}(
            AbstractTransformation[InputNode{$I}(), OutputNode{$O}()],
            NTuple{2,Int}[],
            String["Input", "Output"]
        )
    )))
    @show block

    # graph_nodes = [InputNode(), OutputNode()]
    # graph_edges = []
    numnodes = Ref(2) # input and output
    for item in func_body.args
        add_item_to_graph!(block, _input_index, variables, numnodes, item, 2)
        
        # connect this node to the output node, since it is returned from the function
        add_edge!(block, numnodes[], _output_index)
    end

    push!(block.args, esc(:(g)))
    @show block
    block
end

macro op(args...)
    _op_macro(args...)
end











#####
# below this point is old code:

# Base.call(o::Operator, input) = map(o.f, input)
# transform(o::Operator, input) = map(o.f, input)

# # operator(f::Function, I, O) = Operator{Symbol(f), I, O}(f)

# # same dimensions
# function transform!{T,F,I,O}(output::AbstractArray{T,O}, o::Operator{F,I,O}, input::AbstractArray{T,I})
#     map!(o.f, output, input)
# end
# function transform!{T,F,I,O}(output::AbstractArray{T,O}, o::Operator{F,I,O}, input::AbstractArray{T,I})
#     map!(o.f, output, input)
# end

# # @generated function transform!{T,F,I,O}(output::AbstractArray{T,O}, o::Operator{F,I,O}, input::AbstractArray{T,I})
# #     @show T, F, O, I
    
# #     :(for (i,j) in zip(eachindex(output), eachindex(input))
# #         output[i] = (o.f)(input[j])
# #     end)
# # end