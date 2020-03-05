CHSPy (Cubic Hermite Splines for Python)
========================================

This module provides Python tools for `cubic Hermite splines <https://en.wikipedia.org/wiki/Cubic_Hermite_spline>`_ with one argument (time) and multiple values (:math:`ℝ→ℝ^n`).
It was branched of from `JiTCDDE <http://github.com/neurophysik/jitcdde>`_, which uses it for representing the past of a delay differential equation.
CHSPy is not optimised for efficiency, however it should be fairly effective for high-dimensionally valued splines.

Each spline (`CubicHermiteSpline`) is stored as a series of *anchors* (using the `Anchor` class) each of which contains:

* a time point (`time`),
* an :math:`n`-dimensional state (`state`),
* an :math:`n`-dimensional temporal derivative (`diff`).

Between such anchors, the spline is uniquely described by a polynomial of third degree. With other words, the spline is a piecewise cubic Hermite interpolant of its anchors.

Example
-------

The following example implements a simple three-dimensional spline with three anchors (at :math:`t=0`, :math:`t=1`, and :math:`t=4`) and plots it.

.. plot:: ../examples/simple_example.py
   :include-source:

The markers depict the anchors.
Note how the slope at the anchors is zero for Components 0 and 2 (which is how we defined the spline), while it isn’t for Component 1.

Command Reference
-----------------

.. automodule:: _chspy
	:members:
