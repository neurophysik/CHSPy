#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from bisect import insort, bisect_left, bisect_right
from warnings import warn

def rel_dist(x,y):
	x = np.asarray(x)
	y = np.asarray(y)
	return np.linalg.norm(x-y)/np.linalg.norm(np.mean((x,y)))

class Anchor(tuple):
	"""
	Class for a single anchor. Behaves mostly like a tuple, except that the respective components can also be accessed via the attributes `time`, `state`, and `diff`, and some copying and checks are performed upon creation.
	Also, it implements the less-than operator (<) for comparison by time, which allows to use Python’s sort routines.
	"""
	def __new__( cls, time, state, diff ):
		state = np.atleast_1d(np.array(state,dtype=float,copy=True))
		diff  = np.atleast_1d(np.array(diff ,dtype=float,copy=True))
		if len(state.shape) != 1:
			raise ValueError("State must be a number or one-dimensional iterable.")
		if state.shape != diff.shape:
			raise ValueError("State and diff do not match in shape.")
		return super().__new__(cls,(time,state,diff))
	
	def __init__(self, *args):
		self.time  = self[0]
		self.state = self[1]
		self.diff  = self[2]
	
	# This is for sorting, which is guaranteed (docs.python.org/3/howto/sorting.html) to use __lt__, and bisect_left:
	def __lt__(self,other):
		if isinstance(other,Anchor):
			return self.time < other.time
		else:
			return self.time < float(other)
	
	def __gt__(self,other):
		if isinstance(other,Anchor):
			return self.time > other.time
		else:
			return self.time > float(other)

def interpolate(t,i,anchors):
	"""
	Returns the `i`-th value of a cubic Hermite interpolant of the `anchors` at time `t`.
	"""
	return interpolate_vec(t,anchors)[i]

def interpolate_vec(t,anchors):
	"""
	Returns all values of a cubic Hermite interpolant of the `anchors` at time `t`.
	"""
	q = (anchors[1].time-anchors[0].time)
	x = (t-anchors[0].time) / q
	a = anchors[0].state
	b = anchors[0].diff * q
	c = anchors[1].state
	d = anchors[1].diff * q
	
	return (1-x) * ( (1-x) * (b*x + (a-c)*(2*x+1)) - d*x**2) + c

def interpolate_diff(t,i,anchors):
	"""
	Returns the `i`-th component of the derivative of a cubic Hermite interpolant of the `anchors` at time `t`.
	"""
	return interpolate_diff_vec(t,anchors)[i]

def interpolate_diff_vec(t,anchors):
	"""
	Returns the derivative of a cubic Hermite interpolant of the `anchors` at time `t`.
	"""
	q = (anchors[1].time-anchors[0].time)
	x = (t-anchors[0].time) / q
	a = anchors[0].state
	b = anchors[0].diff * q
	c = anchors[1].state
	d = anchors[1].diff * q
	
	return ( (1-x)*(b-x*3*(2*(a-c)+b+d)) + d*x ) /q

sumsq = lambda x: np.sum(x**2)

# The matrix induced by the scalar product of the cubic Hermite interpolants of two anchors, if their distance is normalised to 1.
sp_matrix = np.array([
			[156,  22,  54, -13],
			[ 22,   4,  13,  -3],
			[ 54,  13, 156, -22],
			[-13,  -3, -22,   4],
		])/420

# The matrix induced by the scalar product of the cubic Hermite interpolants of two anchors, if their distance is normalised to 1, but the initial portion z of the interval is not considered for the scalar product.
def partial_sp_matrix(z):
	h_1 = - 120*z**7 - 350*z**6 - 252*z**5
	h_2 = -  60*z**7 - 140*z**6 -  84*z**5
	h_3 = - 120*z**7 - 420*z**6 - 378*z**5
	h_4 = -  70*z**6 - 168*z**5 - 105*z**4
	h_6 =            - 105*z**4 - 140*z**3
	h_7 =            - 210*z**4 - 420*z**3
	h_5 = 2*h_2 + 3*h_4
	h_8 = - h_5 + h_7 - h_6 - 210*z**2
	
	return np.array([
			[  2*h_3   , h_1    , h_7-2*h_3        , h_5              ],
			[    h_1   , h_2    , h_6-h_1          , h_2+h_4          ],
			[ h_7-2*h_3, h_6-h_1, 2*h_3-2*h_7-420*z, h_8              ],
			[   h_5    , h_2+h_4, h_8              , -h_1+h_2+h_5+h_6 ]
		])/420

def norm_sq_interval(anchors, indices):
	"""
	Returns the squared norm of the interpolant of `anchors` for the `indices`.
	"""
	q = (anchors[1].time-anchors[0].time)
	vector = np.vstack([
			anchors[0].state[indices]   , # a
			anchors[0].diff[indices] * q, # b
			anchors[1].state[indices]   , # c
			anchors[1].diff[indices] * q, # d
		])
	
	return np.einsum(
			vector   , [0,2],
			sp_matrix, [0,1],
			vector   , [1,2],
		)*q

def norm_sq_partial(anchors, indices, start):
	"""
	Returns the sqared norm of the interpolant of `anchors` for the `indices`, but only taking into account the time after `start`.
	"""
	q = (anchors[1].time-anchors[0].time)
	z = (start-anchors[1].time) / q
	vector = np.vstack([
			anchors[0].state[indices]   , # a
			anchors[0].diff[indices] * q, # b
			anchors[1].state[indices]   , # c
			anchors[1].diff[indices] * q, # d
		])
	
	return np.einsum(
			vector              , [0,2],
			partial_sp_matrix(z), [0,1],
			vector              , [1,2],
		)*q

def scalar_product_interval(anchors, indices_1, indices_2):
	"""
	Returns the (integral) scalar product of the interpolants of `anchors` for `indices_1` (one side of the product) and `indices_2` (other side).
	"""
	q = (anchors[1].time-anchors[0].time)
	
	vector_1 = np.vstack([
		anchors[0].state[indices_1],    # a_1
		anchors[0].diff[indices_1] * q, # b_1
		anchors[1].state[indices_1],    # c_1
		anchors[1].diff[indices_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0].state[indices_2],    # a_2
		anchors[0].diff[indices_2] * q, # b_2
		anchors[1].state[indices_2],    # c_2
		anchors[1].diff[indices_2] * q, # d_2
	])
	
	return np.einsum(
			vector_1, [0,2],
			sp_matrix, [0,1],
			vector_2, [1,2]
		)*q

def scalar_product_partial(anchors, indices_1, indices_2, start):
	"""
	Returns the scalar product of the interpolants of `anchors` for `indices_1` (one side of the product) and `indices_2` (other side), but only taking into account the time after `start`.
	"""
	q = (anchors[1].time-anchors[0].time)
	z = (start-anchors[1].time) / q
	
	vector_1 = np.vstack([
		anchors[0].state[indices_1],    # a_1
		anchors[0].diff[indices_1] * q, # b_1
		anchors[1].state[indices_1],    # c_1
		anchors[1].diff[indices_1] * q, # d_1
	])
	
	vector_2 = np.vstack([
		anchors[0].state[indices_2],    # a_2
		anchors[0].diff[indices_2] * q, # b_2
		anchors[1].state[indices_2],    # c_2
		anchors[1].diff[indices_2] * q, # d_2
	])
	
	return np.einsum(
			vector_1, [0,2],
			partial_sp_matrix(z), [0,1],
			vector_2, [1,2]
		)*q

class Extrema(object):
	"""
	Class for containing the extrema and their positions in `n` dimensions. These can be accessed via the attributes `minima`, `maxima`, `arg_min`, and `arg_max`.
	"""
	def __init__(self,n):
		self.arg_min = np.full(n,np.nan)
		self.arg_max = np.full(n,np.nan)
		self.minima = np.full(n, np.inf)
		self.maxima = np.full(n,-np.inf)
	
	def update(self,times,values,condition=True):
		"""
		Updates the extrema if `values` are more extreme.
		
		Parameters
		----------
		condition : boolean or array of booleans
			Only the components where this is `True` are updated.
		"""
		update_min = np.logical_and(values<self.minima,condition)
		self.arg_min = np.where(update_min,times ,self.arg_min)
		self.minima  = np.where(update_min,values,self.minima )
		
		update_max = np.logical_and(values>self.maxima,condition)
		self.arg_max = np.where(update_max,times ,self.arg_max)
		self.maxima  = np.where(update_max,values,self.maxima )

def extrema_from_anchors(anchors,beginning=None,end=None,target=None):
	"""
	Finds minima and maxima of the Hermite interpolant for the anchors.
	
	Parameters
	----------
	beginning : float or `None`
		Beginning of the time interval for which extrema are returned. If `None`, the time of the first anchor is used.
	end : float or `None`
		End of the time interval for which extrema are returned. If `None`, the time of the last anchor is used.
	target : Extrema or `None`
		If an Extrema instance, this one is updated with the newly found extrema and also returned (which means that newly found extrema will be ignored when the extrema in `target` are more extreme).
	
	Returns
	-------
	extrema: Extrema object
		An `Extrema` instance containing the extrema and their positions.
	"""
	
	q = (anchors[1].time-anchors[0].time)
	retransform = lambda x: q*x+anchors[0].time
	a = anchors[0].state
	b = anchors[0].diff * q
	c = anchors[1].state
	d = anchors[1].diff * q
	evaluate = lambda x: (1-x)*((1-x)*(b*x+(a-c)*(2*x+1))-d*x**2)+c
	
	left_x  = 0 if beginning is None else (beginning-anchors[0].time)/q
	right_x = 1 if end       is None else (end      -anchors[0].time)/q
	beginning = anchors[0].time if beginning is None else beginning
	end       = anchors[1].time if end       is None else end
	
	extrema = Extrema(len(anchors[0].state)) if target is None else target
	
	extrema.update(beginning,evaluate(left_x ))
	extrema.update(end      ,evaluate(right_x))
	
	radicant = b**2 + b*d + d**2 + 3*(a-c)*(3*(a-c) + 2*(b+d))
	A = 1/(2*a + b - 2*c + d)
	B = a + 2*b/3 - c + d/3
	for sign in (-1,1):
		with np.errstate(invalid='ignore'):
			x = (B+sign*np.sqrt(radicant)/3)*A
			extrema.update(
					retransform(x),
					evaluate(x),
					np.logical_and.reduce(( radicant>=0, left_x<=x, x<=right_x ))
				)
	
	return extrema

def solve_from_anchors(anchors,i,value,beginning=None,end=None):
	"""
	Finds the times at which a component of the Hermite interpolant for the anchors assumes a given value and the derivatives at those points (allowing to distinguish upwards and downwards threshold crossings).
	
	Parameters
	----------
	i : integer
		The index of the component.
	value : float
		The value that shall be assumed
	beginning : float or `None`
		Beginning of the time interval for which positions are returned. If `None`, the time of the first anchor is used.
	end : float or `None`
		End of the time interval for which positions are returned. If `None`, the time of the last anchor is used.
	
	Returns
	-------
	positions : list of pairs of floats
		Each pair consists of a time where `value` is assumed and the derivative (of `component`) at that time.
	"""
	
	q = (anchors[1].time-anchors[0].time)
	retransform = lambda x: q*x+anchors[0].time
	a = anchors[0].state[i]
	b = anchors[0].diff[i] * q
	c = anchors[1].state[i]
	d = anchors[1].diff[i] * q
	
	left_x  = 0 if beginning is None else (beginning-anchors[0].time)/q
	right_x = 1 if end       is None else (end      -anchors[0].time)/q
	
	candidates = np.roots([
			2*a + b - 2*c + d,
			-3*a - 2*b + 3*c - d,
			b,
			a - value,
		])
	
	solutions = sorted(
			retransform(candidate.real)
			for candidate in candidates
			if np.isreal(candidate) and left_x<=candidate<=right_x
		)
	
	return [ (t,interpolate_diff(t,i,anchors)) for t in solutions ]


class CubicHermiteSpline(list):
	"""
	Class for a cubic Hermite Spline of one variable (time) with `n` values. This behaves like a list with additional functionalities and checks. Note that the times of the anchors must always be in ascending order.
	
	Parameters
	----------
	n : integer
		Dimensionality of the values. If `None`, the following argument must be an instance of CubicHermiteSpline.
	anchors : iterable of triplets
		Contains all the anchors with which the spline is initiated.
		If `n` is `None` and this is an instance of CubicHermiteSpline, all properties are copied from it.
	"""
	def __init__(self,n=None,anchors=()):
		self._times = None
		if n is None:
			assert isinstance(anchors,CubicHermiteSpline)
			CubicHermiteSpline.__init__( self, anchors.n, anchors)
		else:
			self.n = n
			super().__init__( [self.prepare_anchor(anchor) for anchor in anchors] )
			self.sort()
	
	def prepare_anchor(self,x):
		x = x if isinstance(x,Anchor) else Anchor(*x)
		if x.state.shape != (self.n,):
			raise ValueError("State has wrong shape.")
		return x
	
	def append(self,anchor):
		self._times = None
		anchor = self.prepare_anchor(anchor)
		if self and anchor.time <= self[-1].time:
			raise ValueError("Anchor must follow last one in time. Consider using `add` instead.")
		super().append(anchor)
	
	def extend(self,anchors):
		self._times = None
		for anchor in anchors:
			self.append(anchor)
	
	def copy(self):
		# Using type so this works with inheritance.
		return type(self)(anchors=self)
	
	def __setitem__(self,key,item):
		anchor = self.prepare_anchor(item)
		if (
					(key!= 0 and key!=-len(self)   and self[key-1].time>=anchor.time)
				or  (key!=-1 and key!= len(self)-1 and self[key+1].time<=anchor.time)
			):
			raise ValueError("Anchor’s time does not fit.")
		self._times = None
		super().__setitem__(key,anchor)
	
	def insert(self,key,item):
		anchor = self.prepare_anchor(item)
		if (
					(key!= 0 and key!=-len(self) and self[key-1].time>=anchor.time)
				or  (            key!= len(self) and self[key  ].time<=anchor.time)
			):
			raise ValueError("Anchor’s time does not fit. Consider using `add` instead")
		self._times = None
		super().insert(key,anchor)
	
	def sort(self):
		self.check_for_duplicate_times()
		super().sort()
		self._times = None
	
	def check_for_duplicate_times(self):
		if len({anchor.time for anchor in self}) != len(self):
			raise ValueError("You cannot have two anchors with the same time.")
	
	def add(self,anchor):
		"""
		Inserts `anchor` at the appropriate time.
		"""
		insort(self,self.prepare_anchor(anchor))
		self._times = None
	
	def pop(self,index=-1):
		self._times = None
		return super().pop(index)
	
	def remove(self,value):
		self._times = None
		super().remove(value)
	
	def clear_from(self,n):
		"""
		Removes all anchors with an index of `n` or higher.
		"""
		while len(self)>n:
			self.pop()
	
	def clear(self):
		super().__init__()
		self._times = None
	
	def reverse(self):
		raise AssertionError("Anchors must be ordered by time. Therefore this does not make sense.")
	
	@property
	def t(self):
		"""
		The time of the last anchor. This may be overwritten in subclasses such that `self.t` and the time of the last anchor are not identical anymore.
		"""
		return self[-1].time
	
	@property
	def times(self):
		"""
		The times of all anchors.
		"""
		if self._times is None:
			self._times = [anchor.time for anchor in self]
		return self._times
	
	def last_index_before(self,time):
		"""
		Returns the index of the last anchor before `time`.
		Returns 0 if `time` is before the first anchor.
		"""
		return bisect_left(self,float(time),lo=1)-1
	
	def first_index_after(self,time):
		"""
		Returns the index of the first anchor after `time`.
		If `time` is after the last anchors, the latter’s index is returned.
		"""
		return bisect_right(self,float(time),hi=len(self)-1)
	
	def constant(self,state,time=0):
		"""
		makes the spline constant, removing possibly previously existing anchors.
		
		Parameters
		----------
		state : iterable of floats
		time : float
			The time of the last point.
		"""
		
		if self:
			warn("The spline already contains points. This will remove them. Be sure that you really want this.")
			self.clear()
		
		self.append(( time-1., state, np.zeros_like(state) ))
		self.append(( time   , state, np.zeros_like(state) ))
	
	def from_function(self,function,times_of_interest=None,max_anchors=100,tol=5):
		"""
		Like `from_func` except for not being a class method and overwriting previously existing anchors. In most cases, you want to use `from_func` instead.
		"""
		
		assert tol>=0, "tol must be non-negative."
		assert max_anchors>0, "Maximum number of anchors must be positive."
		assert len(times_of_interest)>=2, "I need at least two time points of interest."
		
		if self:
			warn("The spline already contains points. This will remove them. Be sure that you really want this. If not, consider using `from_func`.")
			self.clear()
		
		# A happy anchor is sufficiently interpolated by its neighbours, temporally close to them, or at the border of the interval.
		def unhappy_anchor(*args):
			result = Anchor(*args)
			result.happy = False
			return result

		if callable(function):
			array_function = lambda time: np.asarray(function(time))
			def get_anchor(time):
				value = array_function(time)
				eps = time*10**-tol or 10**-tol
				derivative = (array_function(time+eps)-value)/eps
				return unhappy_anchor(time,value,derivative)
		else:
			import sympy
			function = [ sympy.sympify(comp) for comp in function ]
			
			symbols = set.union(*(comp.free_symbols for comp in function))
			if len(symbols)>2:
				raise ValueError("Expressions must contain at most one free symbol")
			
			def get_anchor(time):
				substitutions = {symbol:time for symbol in symbols}
				evaluate = lambda expr: expr.subs(substitutions).evalf(tol)
				return unhappy_anchor(
						time,
						np.fromiter((evaluate(comp       ) for comp in function),dtype = float),
						np.fromiter((evaluate(comp.diff()) for comp in function),dtype = float),
					)
		
		for time in sorted(times_of_interest):
			self.append(get_anchor(time))
		self[0].happy = self[-1].happy = True
		
		# Insert at least one anchor, if there are only two:
		if len(self)==2<max_anchors:
			time = np.mean((self[0].time,self[1].time))
			self.insert(1,get_anchor(time))
		
		while not all(anchor.happy for anchor in self) and len(self)<=max_anchors:
			for i in range(len(self)-2,-1,-1):
				# Update happiness
				if not self[i].happy:
					guess = interpolate_vec( self[i].time, (self[i-1], self[i+1]) )
					self[i].happy = (
							rel_dist(guess,self[i].state) < 10**-tol or
							rel_dist(self[i+1].time,self[i-1].time) < 10**-tol
						)
				
				# Add new anchors, if unhappy
				if not (self[i].happy and self[i+1].happy):
					time = np.mean((self[i].time,self[i+1].time))
					self.insert(i+1,get_anchor(time))
				
				if len(self)>max_anchors:
					break
	
	@classmethod
	def from_func(cls,function,times_of_interest=None,max_anchors=100,tol=5):
		"""
		makes the spline interpolate a given function at heuristically determined points. More precisely, starting with `times_of_interest`, anchors are added until either:
		
		* anchors are closer than the tolerance
		* the value of an anchor is approximated by the interpolant of its neighbours within the tolerance
		* the maximum number of anchors is reached.

		This removes possibly previously existing anchors.
		
		Parameters
		----------
		function : callable or iterable of SymPy/SymEngine expressions
			The function to be interpolated.
			If callable, this is interpreted like a regular function mapping a time point to a state vector (as an iterable).
			If an iterable of expressions, each expression represents the respective component of the function.
		
		times_of_interest : iterable of numbers
			Initial set of time points considered for the interpolation. All created anhcors will between the minimal and maximal timepoint.
		
		max_anchors : positive integer
			The maximum number of anchors that this routine will create (including those for the `times_of_interest`).
		
		tol : integer
			This is a parameter for the heuristics, more precisely the number of digits considered for tolerance in several places.
		"""
		
		if callable(function):
			test_time = times_of_interest[0] if times_of_interest else 0
			n = len(function(test_time))
		else:
			n = len(function)
		
		spline = cls(n=n)
		spline.from_function(function,times_of_interest,max_anchors,tol)
		return spline
	
	@classmethod
	def from_data(cls,times,states):
		"""
		Creates a new cubic Hermite spline based on a provided dataset. The derivative of a given anchor is estimated from a quadratic interpolation of that anchor and the neighbouring ones. (For the first and last anchor, it’s only a linear interpolation.)
		
		This is only a best general guess how to interpolate the data. Often you can apply your knowledge of the data to do better.
		
		Parameters
		----------
		times : array-like
			The times of the data points.
		states : array-like
			The values of the data. The first dimension has to have the same length as `times`.
		"""
		assert len(times)==len(states)
		states = np.asarray(states)
		assert states.ndim==2
		spline = cls(n=states.shape[1])
		
		diffs = np.empty_like(states)
		diffs[ 0] = (states[ 1]-states[ 0])/(times[ 1]-times[ 0])
		diffs[-1] = (states[-1]-states[-2])/(times[-1]-times[-2])
		
		y_1 = states[ :-2]
		y_2 = states[1:-1]
		y_3 = states[2:  ]
		t_1 = times [ :-2]
		t_2 = times [1:-1]
		t_3 = times [2:  ]
		diffs[1:-1] = (
				  y_1 * ((t_2-t_3)/(t_2-t_1)/(t_3-t_1))[:,None]
				+ y_2 / (t_2-t_3)[:,None]
				+ y_2 / (t_2-t_1)[:,None]
				+ y_3 * ((t_2-t_1)/(t_3-t_1)/(t_3-t_2))[:,None]
			)
		
		for anchor in zip(times,states,diffs):
			spline.add(anchor)
		return spline
	
	def get_anchors(self, time):
		"""
		Find the two anchors neighbouring `time`.
		If `time` is outside the ranges of times covered by the anchors, return the two nearest anchors.
		"""
		s = bisect_left(self,float(time),lo=1,hi=len(self)-1)-1
		return ( self[s], self[s+1] )
	
	def get_state(self,times):
		"""
		Get the state of the spline at `times`.
		If any time point lies outside of the anchors, the state will be extrapolated.
		"""
		if np.ndim(times)==0:
			return interpolate_vec(times,self.get_anchors(times))
		elif np.ndim(times)==1:
			return np.vstack([
					interpolate_vec(time,self.get_anchors(time))
					for time in times
				])
		else:
			raise ValueError("times is not zero- or one-dimensional sequence of numbers")
	
	def get_recent_state(self,t):
		"""
		Interpolate the state at time `t` from the last two anchors.
		This usually only makes sense if `t` lies between the last two anchors.
		"""
		anchors = self[-2], self[-1]
		output = interpolate_vec(t,anchors)
		assert type(output) == np.ndarray
		return output
	
	def get_current_state(self):
		return self[-1].state
	
	def forget(self, delay):
		"""
		Remove all anchors that are “out of reach” of the delay with respect to `self.t`.
		"""
		threshold = self.t - delay
		while self[1].time<threshold:
			self.pop(0)
	
	def extrema(self,beginning=None,end=None):
		"""
		Returns the positions and values of the minima and maxima of the spline (for each component) within the specified time interval.
		
		Parameters
		----------
		beginning : float or `None`
			Beginning of the time interval for which extrema are returned. If `None`, the time of the first anchor is used.
		end : float or `None`
			End of the time interval for which extrema are returned. If `None`, the time of the last anchor is used.
		
		Returns
		-------
		extrema: Extrema object
			An `Extrema` instance containing the extrema and their positions.
		"""
		
		beginning = self[ 0].time if beginning is None else beginning
		end       = self[-1].time if end       is None else end
		
		if not self[0].time <= beginning < end <= self[-1].time:
			raise ValueError("Beginning and end must in the time interval spanned by the anchors.")
		
		extrema = Extrema(self.n)
		
		for i in range(self.last_index_before(beginning),len(self)-1):
			if self[i].time>end:
				break
			
			extrema_from_anchors(
					( self[i], self[i+1] ),
					beginning = max( beginning, self[i  ].time ),
					end       = min( end      , self[i+1].time ),
					target = extrema,
				)
		
		return extrema

	def solve(self,i,value,beginning=None,end=None):
		"""
		Finds the times at which a component of the spline assumes a given value and the derivatives at those points (allowing to distinguish upwards and downwards threshold crossings). This will not work well if the spline is constantly at the given value for some interval.
		
		Parameters
		----------
		i : integer
			The index of the component.
		value : float
			The value that shall be assumed
		beginning : float or `None`
			Beginning of the time interval for which solutions are returned. If `None`, the time of the first anchor is used.
		end : float or `None`
			End of the time interval for which solutions are returned. If `None`, the time of the last anchor is used.
		
		Returns
		-------
		positions : list of pairs of floats
			Each pair consists of a time where `value` is assumed and the derivative (of `component`) at that time.
		"""
		
		beginning = self[ 0].time if beginning is None else beginning
		end       = self[-1].time if end       is None else end
		
		if not self[0].time <= beginning < end <= self[-1].time:
			raise ValueError("Beginning and end must in the time interval spanned by the anchors.")
		
		extrema = Extrema(self.n)
		
		sols = []
		
		for j in range(self.last_index_before(beginning),len(self)-1):
			if self[j].time>end:
				break
			
			new_sols = solve_from_anchors(
					anchors = ( self[j], self[j+1] ),
					i = i,
					value = value,
					beginning = max( beginning, self[j  ].time ),
					end       = min( end      , self[j+1].time ),
				)
			
			if sols and new_sols and sols[-1][0]==new_sols[0][0]:
				del new_sols[0]
			sols.extend(new_sols)
		
		return sols
	
	def norm(self, delay, indices):
		"""
		Computes the norm of the spline for the given indices taking into account the time between `self.t` − `delay` and `self.t`.
		"""
		threshold = self.t - delay
		i = self.last_index_before(threshold)
		
		# partial norm of first relevant interval
		anchors = (self[i],self[i+1])
		norm_sq = norm_sq_partial(anchors, indices, threshold)
		
		# full norms of all others
		for i in range(i+1, len(self)-1):
			anchors = (self[i],self[i+1])
			norm_sq += norm_sq_interval(anchors, indices)
		
		return np.sqrt(norm_sq)
	
	def scalar_product(self, delay, indices_1, indices_2):
		"""
		Computes the scalar product of the spline between `indices_1` (one side of the product) and `indices_2` (other side) taking into account the time between `self.t` − `delay` and `self.t`.
		"""
		threshold = self.t - delay
		i = self.last_index_before(threshold)
		
		# partial scalar product of first relevant interval
		anchors = (self[i],self[i+1])
		sp = scalar_product_partial(anchors, indices_1, indices_2, threshold)
		
		# full scalar product of all others
		for i in range(i+1, len(self)-1):
			anchors = (self[i],self[i+1])
			sp += scalar_product_interval(anchors, indices_1, indices_2)
		
		return sp
	
	def scale(self, indices, factor):
		"""
		Scales the spline for `indices` by `factor`.
		"""
		for anchor in self:
			anchor.state[indices] *= factor
			anchor.diff [indices] *= factor
	
	def subtract(self, indices_1, indices_2, factor):
		"""
		Substract the spline for `indices_2` multiplied by `factor` from the spline for `indices_1`.
		"""
		for anchor in self:
			anchor.state[indices_1] -= factor*anchor.state[indices_2]
			anchor.diff [indices_1] -= factor*anchor.diff [indices_2]
	
	def interpolate_anchor(self,time):
		"""
		Interpolates an anchor at `time`.
		"""
		i = bisect_left(self,float(time))
		if i==len(self) or self[i].time!=time:
			s = max(1,min(i,len(self)-1))
			anchors = (self[s-1],self[s])
			value =     interpolate_vec(time,anchors)
			diff = interpolate_diff_vec(time,anchors)
			self.insert( i, (time,value,diff) )
	
	def truncate(self,time):
		"""
		Interpolates an anchor at `time` and removes all later anchors.
		"""
		assert self[0].time<=time<=self[-1].time, "Truncation time must be within current range of anchors."
		i = self.last_index_before(time)
		
		value =     interpolate_vec(time,(self[i],self[i+1]))
		diff = interpolate_diff_vec(time,(self[i],self[i+1]))
		self[i+1] = Anchor(time,value,diff)
		self.interpolate_anchor(time)
		self.clear_from(i+2)
		assert len(self)>=1
	
	def plus(self,other):
		"""
		Sum with another spline in place. If the other spline has an anchor that does not have the time of an existing anchor, a new anchor will be added at this time.
		"""
		other = other.copy()
		match_anchors(self,other)
		assert self.times == other.times
		
		for i,anchor in enumerate(other):
			self[i].state += other[i].state
			self[i].diff  += other[i].diff
	
	def plot(self,axes,components="all",resolution=20,*args,**kwargs):
		"""
		Plots the interpolant onto the provided Matplotlib axes object. If `components` is `None`, all components are plotted at once. Otherwise only the selected component is plotted.
		By default this calls `plot` with `markevery=resolution` (marking the anchors) and `marker="o"`, but you can override those arguments.
		It will also label each component with `f"Component {i}"`.
		All further arguments are forwarded to Matplotlib’s `plot`.
		
		Parameters
		----------
		components : int, iterable of ints, or "all"
		
			Which components should be plotted. If `"all"`, all components will be plotted.
		
		resolution : int
		
			How often the Hermite polynomial should be evaluated for plotting between each anchor. The higher this number, the more accurate the plot.
		"""
		
		assert resolution>=1, "Resolution must at least be 1."
		
		if components=="all":
			components = range(self.n)
		components = np.atleast_1d(components)
		
		plot_times = []
		times = self.times
		for i in range(len(times)-1):
			added_points = np.linspace( times[i], times[i+1], resolution, endpoint=False )
			plot_times.extend(added_points)
		plot_times.append(times[-1])
		
		kwargs.setdefault("marker","o")
		kwargs.setdefault("markevery",resolution)
		values = self.get_state(plot_times)
		return [
				axes.plot( plot_times, values[:,c], label=f"Component {c}", *args, **kwargs )
				for c in (components)
			]

def match_anchors(*splines):
	"""
	Ensure that splines have anchors at the same times, interpolating intermediate anchors if necessary. All of this happens in place.
	"""
	timess = [ set(spline.times) for spline in splines ]
	all_times = set.union(*timess)
	for times,spline in zip(timess,splines):
		for time in all_times-times:
			spline.interpolate_anchor(time)
	
	# from itertools import combinations
	# for spline_1,spline_2 in combinations(splines,2):
	# 	assert spline_1.times == spline_2.times == sorted(all_times)

def join(*splines):
	"""
	Glues the splines together along the value dimension, i.e., returns a new spline that features the input splines as disjoint subsets of its components.
	"""
	splines = [spline.copy() for spline in splines]
	match_anchors(*splines)
	positions = np.cumsum([0,*(spline.n for spline in splines)])
	n = positions[-1]
	
	joined_spline = CubicHermiteSpline(n=n)
	
	for i,time in enumerate(splines[0].times):
		state = np.empty(n)
		diff = np.empty(n)
		for j,spline in enumerate(splines):
			state[positions[j]:positions[j+1]] = spline[i].state
			diff [positions[j]:positions[j+1]] = spline[i].diff
		joined_spline.add((time,state,diff))
	
	return joined_spline

