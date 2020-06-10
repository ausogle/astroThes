import enum


class Perturbations(enum.Enum):
    """
    List of acceptable perturbations to be included by the user for propagation purposes.
    """
    J2 = "J2"
    J3 = "J3"
    Drag = "Drag"
    SRP = "SRP"
    Moon = "Moon"
    Sun = "Sun"


class Frames(enum.Enum):
    """
    List of acceptable frames of reference for physical locations.
    """
    ECI = "ECI"
    ECEF = "ECEF"
    LLA = "LLA"


class Angles(enum.Enum):
    """
    List of acceptable angle pairs.
    """
    Local = "Local"
    Celestial = "Celestial"
