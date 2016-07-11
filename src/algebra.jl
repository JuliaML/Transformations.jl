
export
    Operator,
    op,
    @ops

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


immutable InputNode <: AbstractTransformation end
immutable OutputNode <: AbstractTransformation end

# opgraphs are a lightweight representation of a directed graph of AbstractTransformations

type OpGraph{I,O} <: AbstractTransformation{I,O}
    nodes::Vector{AbstractTransformation}
    source::Vector{Int}
    destiny::Vector{Int}
    # nodemap::Dict{Symbol,AbstractTransformation}
    # edges::Dict{AbstractTransformation, AbstractTransformation}
    # input_nodes::Vector{AbstractTransformation}
    # output_nodes::Vector{AbstractTransformation}
end

# function subgraph(ops::OpGraph, expr::Expr)
#     expr.head == :call || error("Parse error in ops.  Expected `call`: $expr")
#     fsym = expr.args[1]
#     fsym in _all_ops || error("Parse error in ops. Function is not in _all_ops: $fsym")
    
# end

# function subgraph(ops::OpGraph, s::Symbol)

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

macro ops(expr, I::Int, O::Int)
    dump(expr, 20)
    ops = OpGraph{I,O}()
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