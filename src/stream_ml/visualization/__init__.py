"""Core library for stream membership likelihood, with ML."""

from astropy.visualization import quantity_support

from stream_ml.visualization.diagnostic import diagnostic_plot
from stream_ml.visualization.likelihood import (
    component_likelihood_dataspace,
    component_likelihood_modelspace,
)
from stream_ml.visualization.parameter import parameter, weight
from stream_ml.visualization.pre import plot_coordinate_histograms_in_phi1_slices

__all__ = [
    "plot_coordinate_histograms_in_phi1_slices",
    "diagnostic_plot",
    # Likelihood
    "component_likelihood_dataspace",
    "component_likelihood_modelspace",
    # Parameter
    "weight",
    "parameter",
]


quantity_support()
