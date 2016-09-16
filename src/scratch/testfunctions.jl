# using Plots; pyplot(); x = linspace(-2,2,20); y = linspace(-1,3,20); contourf(x, y, rosenbrock, zlim=(0,1000))
#
# julia> xy = vec([(xi,yi) for xi=x,yi=y]);
#
# julia> uv = map(xyi -> (g=rosenbrock_gradient(collect(xyi))/2000; (g[1],g[2])), xy);
#
# julia> quiver!(Plots.unzip(xy)..., quiver=uv, c=:cyan, alpha=0.4)
#
# julia> uv = map(xyi -> (g=rosenbrock_gradient(collect(xyi))/-2000; (g[1],g[2])), xy);

using Plots; begin
pyplot()
x = linspace(-2,2,20)
y = linspace(-1,3,20)
xy = vec([(xi,yi) for xi=x,yi=y])
uv = map(xyi -> (g=rosenbrock_gradient(collect(xyi))/-2000; (g[1],g[2])), xy)
contourf(x, y, rosenbrock, xlim=extrema(x), ylim=extrema(y))
quiver!(Plots.unzip(xy)..., quiver=uv, c=:cyan, alpha=0.4)
end
