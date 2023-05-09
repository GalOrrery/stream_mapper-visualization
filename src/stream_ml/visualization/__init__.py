"""Core library for stream membership likelihood, with ML."""

from astropy.visualization import quantity_support

from stream_ml.visualization._defaults import YLABEL_DEFAULTS
from stream_ml.visualization._diagnostic import astrometric_model_panels
from stream_ml.visualization._distributions import (
    coord_panels,
    plot_coordinate_histograms_in_phi1_slices,
)
from stream_ml.visualization._likelihood import component_likelihood
from stream_ml.visualization._parameter import parameter, weight

__all__ = [
    # Constants
    "YLABEL_DEFAULTS",
    # Pre
    "coord_panels",
    "plot_coordinate_histograms_in_phi1_slices",
    # Diagnostic
    "astrometric_model_panels",
    # Likelihood
    "component_likelihood",
    # Parameter
    "weight",
    "parameter",
]


quantity_support()
