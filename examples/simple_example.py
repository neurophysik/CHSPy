from chspy import CubicHermiteSpline
from matplotlib.pyplot import subplots

spline = CubicHermiteSpline(n=3)

#            time   state    slope
spline.add((   0 , [1,3,0], [0,1,0] ))
spline.add((   1 , [3,2,0], [0,4,0] ))
spline.add((   4 , [0,1,3], [0,4,0] ))

fig,axes = subplots(figsize=(7,2))
spline.plot(axes)
axes.set_xlabel("time")
axes.set_ylabel("state")
fig.legend(loc="right",labelspacing=1)
fig.subplots_adjust(right=0.7)
