import enum


class Perturbations(enum.Enum):
    """
    List of acceptable perturbations to be included by the user for propagation purposes.
    """
    J2 = "J2"
    J3 = "J3"
    Drag = "Drag"
    Moon = "Moon"
    Sun = "Sun"


class Frames(enum.Enum):
    """
    List of acceptable frames of reference for physical locations
    """
    ECI = "ECI"
    ECEF = "ECEF"
