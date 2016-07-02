
export
    Operator,
    op

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


@generated function is_mappable{F,I,O}(o::Operator{F,I,O})
    F in (:sin, :exp, :log, :sigmoid, :tanh) ? :(true) : :(false)
end
@generated function is_operator{F,I,O}(o::Operator{F,I,O})
    F in (:*, :/, :\) ? :(true) : :(false)
end
@generated function is_element_operator{F,I,O}(o::Operator{F,I,O})
    F in (:+, :-, :.*, :./, :max, :min) ? :(true) : :(false)
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