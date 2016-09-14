
abstract LearnableParams

immutable NoParams <: LearnableParams end

"""
We need a wrapper for params that can hold:
- a vector view of the full parameter vector
- one or more reshaped views of that vector view

so for t::Transformation:

θ(t) --> SubArray
t.w, t.b --> view(θ)

Problem: every time we change the "master parameter vector"
we need to update the views (but... do we?? gotta check if
    just appending to the master params will invalidate the views)

Solution?  Store anonymous functions which return views of θ(t),
    and update w/b by calling those functions.


By default we can do `view(zeros(T, params_length(t)), :)`, then swap
out with a master param vector on update.
"""
type Params{T <: AbstractVector, VIEWS <: Tuple, S <: Tuple} <: LearnableParams
    θ::T
    ∇::T
    views::VIEWS
    ∇_views::VIEWS
    sizes::S
end

function Params(θ::AbstractVector, ∇::AbstractVector, sizes = ())
    views = splitview(θ, sizes)[1]
    ∇_views = splitview(∇, sizes)[1]
    Params(θ, ∇, views, ∇_views, sizes)
end

function reset!(p::Params, θ::AbstractVector, ∇::AbstractVector)
    p.θ = θ
    p.∇ = ∇
    p.views = splitview(p.θ, p.sizes)[1]
    p.∇_views = splitview(p.∇, p.sizes)[1]
    return
end

Base.getindex(p::Params, i::Int) = p.views[i]

value(p::Params) = p.θ
grad(p::Params) = p.∇
