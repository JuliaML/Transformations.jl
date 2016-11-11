
# ---------------------------------------------------------------------------
# This is experimental, and currently unused
# ---------------------------------------------------------------------------

using DataFlow

export
	edges

function nodename(v)
    val = v.value
    isconst = isa(val, DataFlow.Constant)
    val = isconst ? val.value : val
    val, isconst
end


function edges(v::DVertex, source = Int[], destiny = Int[], names = Symbol[], cache = DataFlow.ODict())
	# haskey(cache,v) && return source,destiny,names
    name, isconst = nodename(v)
    haskey(cache, isconst ? name : v) && return source,destiny,names
    push!(names, name)
    didx = length(cache)+1
    cache[isconst ? name : v] = didx
    # @show name cache
	# cache[v] = length(cache) + 1
    # val = v.value
	# push!(names, string(isa(val, DataFlow.Constant) ? val.value : val))
	for v′ in DataFlow.inputs(v)
		edges(v′, source, destiny, names, cache)
		# push!(source, cache[v′])
		# push!(destiny, cache[v])
        sname, sisconst = nodename(v′)
		push!(source, cache[sisconst ? sname : v′])
		push!(destiny, didx)
	end
	source, destiny, names
end
edges(g::DataFlow.SyntaxGraph, args...) = edges(g.output, args...)

using PlotRecipes
@recipe function f(g::Union{DataFlow.SyntaxGraph,DVertex})
	source, destiny, names = edges(g)
    # arrow --> arrow()
    # markersize --> 50
    # markeralpha --> 0.2
    # linealpha --> 0.4
    # linewidth --> 2
    names --> map(string, names)
    method --> :tree
    root --> :left
    PlotRecipes.GraphPlot((source, destiny))
end
