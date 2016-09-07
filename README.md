# Transformations

[![Build Status](https://travis-ci.org/JuliaML/Transformations.jl.svg?branch=master)](https://travis-ci.org/JuliaML/Transformations.jl)

Static transforms, activation functions, and more.

---

A Transformation is an abstraction which represents a (possibly differentiable, possibly parameterized) mapping from input(s) to output(s).  In a classic computational graph framework, nodes of the graph are primitives: "variables", "constants", or "operators". They are connected together by edges which define a tree-like definition of computation.  Complex operations and automatic differentiation can be applied at the primitive-level, and the full connectivity of a graph must be considered during a "compilation" stage.

Transformations takes an alternative view in which each Transformation is a sub-graph from input node(s) to output node(s).  There may be parameter nodes and operations embedded inside, but from the outside it can be treated as a black box function: `output = f(input, θ)`.  The output of one Transformation can be "linked" to the input of another, which binds the underlying array storage and connects them in the computation pipeline.

The end goal is one of specialization and consolidation.  Instead of expanding out a massive graph into primitives, we can maintain modular building blocks of our choosing and make it simple (and fast) to dynamically add and remove transformations in a larger graph, without recompiling.

TODO: images to help explain the concepts
