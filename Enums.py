import enum


class Perturbations(enum.Enum):
    J2 = "J2"
    J3 = "J3"
    Drag = "Drag"
    Moon = "Moon"
    Sun = "Sun"
    Jupiter = "Jupiter"
    Venus = "Venus"


class Frames(enum.Enum):
    ECI = "ECI"
    ECEF = "ECEF"
