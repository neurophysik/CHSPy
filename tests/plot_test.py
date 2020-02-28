from chspy import CubicHermiteSpline
from matplotlib.pyplot import subplots

spline = CubicHermiteSpline(2)

spline.add((0,[1,3],[0,1]))
spline.add((1,[3,2],[0,4]))
spline.add((4,[0,1],[0,2]))

fig,axes = subplots()
spline.plot(axes,resolution=100,components=[1,0])
fig.savefig("test.pdf")

