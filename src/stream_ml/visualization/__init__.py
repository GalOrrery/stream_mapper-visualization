"""Core library for stream membership likelihood, with ML."""

from astropy.visualization import quantity_support

from stream_ml.visualization.diagnostic import astrometric_model_panels
from stream_ml.visualization.distributions import (
    coord_panels,
    plot_coordinate_histograms_in_phi1_slices,
)
from stream_ml.visualization.likelihood import (
    component_likelihood_dataspace,
    component_likelihood_modelspace,
)
from stream_ml.visualization.parameter import parameter, weight

__all__ = [
    # Pre
    "coord_panels",
    "plot_coordinate_histograms_in_phi1_slices",
    # Diagnostic
    "astrometric_model_panels",
    # Likelihood
    "component_likelihood_dataspace",
    "component_likelihood_modelspace",
    # Parameter
    "weight",
    "parameter",
]


quantity_support()
