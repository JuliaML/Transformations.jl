
export
    Operator,
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

const _all_ops = vcat(_mappables, _operators, _element_ops, _aggregators)

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
immutable InputNode{I} <: AbstractTransformation end
immutable OutputNode{O} <: AbstractTransformation end

# a Learnable has no inputs, but produces an output (the parameters)
immutable Learnable{O} <: AbstractTransformation end

# an OpGraph is a lightweight representation of a directed graph of AbstractTransformations
type OpGraph{I,O} <: AbstractTransformation{I,O}
    nodes::Vector{AbstractTransformation}
    edges::Vector{NTuple{2,Int}}
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

function add_item_to_graph!(graph_nodes, graph_edges, item)
    @show item typeof(item)
    if isa(item, Expr)
        # TODO: 
    end
end

function add_edge!(block::Expr, i::Int, j::Int)
    push!(block, :(push!(g.edges, ($i, $j))))
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
    func_args = func_signature.args[2:end]
    @show func_name func_args

    # TODO: we want to build an OpGraph as a function of Variable, Learnable, and Op components
    # To simplify, we'll assume that:
    #   - Expr --> Op
    #   - Symbol --> haskey(_ops, k) ? Variable : Learnable

    # Variables are not learnable, and they are NOT part of the graph.  They represent the inputs
    # to the OpGraph.

    block = quote
        g = OpGraph{$(inout[1]), $(inout[2])}([InputNode{I}(), OutputNode{O}()], [])
    end

    graph_nodes = [InputNode(), OutputNode()]
    graph_edges = []
    for item in func_body
        add_item_to_graph!(block, graph_nodes, graph_edges, item)
        
        # connect this node to the output node, since it is returned from the function
        add_edge!(block, length(graph_nodes), _output_index)
    end

    push!(block, :(g))
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