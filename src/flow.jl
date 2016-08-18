
@reexport using Flow

export
	edges

function edges(v::DVertex, source = Int[], destiny = Int[], names = Symbol[], cache = Flow.ODict())
	haskey(cache,v) && return source,destiny,names
	cache[v] = length(cache) + 1
	push!(names, string(v.value))
	for v′ in inputs(v)
		edges(v′, source, destiny, names, cache)
		push!(source, cache[v′])
		push!(destiny, cache[v])
	end
	source, destiny, names
end
edges(g::Flow.SyntaxGraph, args...) = edges(g.output, args...)

using PlotRecipes
@recipe function f(g::Flow.SyntaxGraph)
	source, destiny, names = edges(g.output)
    arrow --> arrow()
    markersize --> 50
    markeralpha --> 0.2
    linealpha --> 0.4
    linewidth --> 2
    names --> names
    func --> :tree
    root --> :left
    PlotRecipes.GraphPlot((source, destiny))
end
