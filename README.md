# DEPRECATED

This package is deprecated.

# Transformations

[![Build Status](https://travis-ci.org/JuliaML/Transformations.jl.svg?branch=master)](https://travis-ci.org/JuliaML/Transformations.jl)

Static transforms, activation functions, learnable transformations, neural nets, and more.  

---

A Transformation is an abstraction which represents a (possibly differentiable, possibly parameterized) mapping from input(s) to output(s).  In a classic computational graph framework, nodes of the graph are primitives: "variables", "constants", or "operators". They are connected together by edges which define a tree-like definition of computation.  Complex operations and automatic differentiation can be applied at the primitive-level, and the full connectivity of a graph must be considered during a "compilation" stage.

Transformations takes an alternative view in which each Transformation is a sub-graph from input node(s) to output node(s).  There may be parameter nodes and operations embedded inside, but from the outside it can be treated as a black box function: `output = f(input, θ)`.  The output of one Transformation can be "linked" to the input of another, which binds the underlying array storage and connects them in the computation pipeline.

![](https://cloud.githubusercontent.com/assets/933338/20273883/edfaa236-aa60-11e6-9c6c-9e8c8945201b.png)

The end goal is one of specialization and consolidation.  Instead of expanding out a massive graph into primitives, we can maintain modular building blocks of our choosing and make it simple (and fast) to dynamically add and remove transformations in a larger graph, without recompiling.

For more on the design, see [my blog post](http://www.breloff.com/transformations/).

Implemented:

- Linear (y = wx)
- Affine (y = wx + b)
- Activations:
    - logistic (sigmoid)
    - tanh
    - softsign
    - ReLU
    - softplus
    - sinusoid
    - gaussian
- Multivariate Normal
- Online/Incremental Layer Normalization
- N-Differentiable Functions
- Convolution/Pooling (WIP)
- Online/Incremental Whitening:
    - PCA
    - Whitened PCA
    - ZCA
- Feed-Forward ANNs
- Aggregators:
    - Sum
    - Gate (Product)
    - Concat

### Primary author: Tom Breloff (@tbreloff)
