"""Core library for stream membership likelihood, with ML."""


__all__: tuple[str, ...] = ()

from types import MappingProxyType

_LABEL_DEFAULTS: dict[str, str] = {
    # ICRS - RA
    "ra": r"$\alpha$",
    # ICRS - Dec
    "dec": r"$\delta$",
    # Distance
    "distance": r"$r$",
    "parallax": r"$\varpi$",
    "plx": r"$\varpi$",
    # ICRS - Proper Motion
    "pmra": r"$\mu_{\alpha}$",
    "pmra_cosdec": r"$\mu_{\alpha}^*$",
    "pmra_cosdec_unrefl": r"$\mu_{\alpha}^*$",
    "pmra_cosdec_refl": r"$\mu_{\alpha}^*$",
    "pmdec": r"$\mu_{\delta}$",
    "pmdec_unrefl": r"$\mu_{\delta}^*$",
    "pmdec_refl": r"$\mu_{\delta}^*$",
    "radial_velocity": r"$v_r$",
    # Custom
    "phi1": r"$\phi_1$",
    "phi2": r"$\phi_2$",
    "pm_phi1": r"$\mu_{\phi_1}$",
    "pm_phi1_cosphi2": r"$\mu_{\phi_1}^*$",
    "pm_phi1_cosphi2_unrefl": r"$\mu_{\phi_1}^*$",
    "pm_phi1_cosphi2_refl": r"$\mu_{\phi_1}^*$",
    "pm_phi2": r"$\mu_{\phi_2}$",
    "pm_phi2_unrefl": r"$\mu_{\phi_2}$",
    "pm_phi2_refl": r"$\mu_{\phi_2}$",
}
LABEL_DEFAULTS = MappingProxyType(_LABEL_DEFAULTS)
