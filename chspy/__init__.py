from ._chspy import Anchor, CubicHermiteSpline, Extrema, extrema_from_anchors, interpolate, interpolate_diff, interpolate_diff_vec, interpolate_vec, join, match_anchors, norm_sq_interval, norm_sq_partial, scalar_product_interval, scalar_product_partial  # noqa: F401


try:
	from .version import version as __version__  # noqa: F401
except ImportError:
	from warnings import warn
	warn("Failed to find (autogenerated) version.py. Do not worry about this unless you really need to know the version.", stacklevel=1)
