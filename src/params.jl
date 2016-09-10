
abstract LearnableParams

immutable NoParams <: LearnableParams end

"""
We need a wrapper for params that can hold:
- a vector view of the full parameter vector
- one or more reshaped views of that vector view

so for t::Transformation:

Θ(t) --> SubArray
t.w, t.b --> view(Θ)

Problem: every time we change the "master parameter vector"
we need to update the views (but... do we?? gotta check if
    just appending to the master params will invalidate the views)

Solution?  Store anonymous functions which return views of Θ(t),
    and update w/b by calling those functions.


By default we can do `view(zeros(T, params_length(t)), :)`, then swap
out with a master param vector on update.
"""
type Params{T <: SubArray, VIEWS <: Tuple, S <: Tuple} <: LearnableParams
    Θ::T
    ∇::T
    views::VIEWS
    ∇_views::VIEWS
    sizes::S
end

function Params(Θ::AbstractVector, ∇::AbstractVector, sizes = ())
    views = splitview(Θ, sizes)
    ∇_views = splitview(∇, sizes)
    Params(Θ, ∇, views, ∇_views, sizes)
end

function reset!(p::Params, Θ::AbstractVector, ∇::AbstractVector)
    p.Θ = Θ
    p.∇ = ∇
    p.views = splitview(p.Θ, p.sizes)
    p.∇_views = splitview(p.∇, p.sizes)
    return
end

Base.getindex(p::Params, i::Int) = p.views[i]

value(p::Params) = p.Θ
grad(p::Params) = p.∇
