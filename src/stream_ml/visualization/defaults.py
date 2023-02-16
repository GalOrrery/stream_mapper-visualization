"""Core library for stream membership likelihood, with ML."""


__all__: list[str] = []


COL_NAME_DEFAULTS = (
    "phi1",
    "phi2",
    "parallax",
    "pm_phi1_cosphi2_unrefl",
    "pm_phi2_unrefl",
)

YLABEL_DEFAULTS = {
    "phi1": "$\\phi_1$",
    "phi2": r"$\phi_2$",
    "parallax": r"$\varpi$",
    "pm_phi1_cosphi2_unrefl": r"$\mu_{phi_1}^*$",
    "pm_phi2_unrefl": r"$\mu_{phi_2}$",
}

COORD_TO_YLABEL = {
    "phi1": "$\\phi_1$",
    "phi2": r"$\phi_2$",
    "plx": r"$\varpi$",
    "pm_phi1": r"$\mu_{phi_1}^*$",
    "pm_phi2": r"$\mu_{phi_2}$",
}


COORD_TO_TABLE = {
    "phi1": "phi1",
    "phi2": "phi2",
    "plx": "parallax",
    "pm_phi1": "pm_phi1_cosphi2_unrefl",
    "pm_phi2": "pm_phi2_unrefl",
}
