"""Core library for stream membership likelihood, with ML."""

from astropy.visualization import quantity_support

from stream_mapper.visualization._defaults import LABEL_DEFAULTS
from stream_mapper.visualization._diagnostic import astrometric_model_panels
from stream_mapper.visualization._distribution import coord_panels
from stream_mapper.visualization._likelihood import component_likelihood
from stream_mapper.visualization._parameter import parameter, weight
from stream_mapper.visualization._slices import plot_coordinates_in_slices

__all__ = (
    # Constants
    "LABEL_DEFAULTS",
    # Pre
    "coord_panels",
    "plot_coordinates_in_slices",
    # Diagnostic
    "astrometric_model_panels",
    # Likelihood
    "component_likelihood",
    # Parameter
    "weight",
    "parameter",
)


quantity_support()
