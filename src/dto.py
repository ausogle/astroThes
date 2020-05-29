from typing import List
from astropy.time import Time

class ObsParams:
    """
    Object intended to provide all relevant information to the predictor function.
    """
    def __init__(self, position, frame, epoch_obs: Time):
        self.position = position   # If LLA frame, this is a list. If ECI/ECEF this is a numpy array. See cleaning.py
        self.frame = frame
        self.epoch = epoch_obs


class PropParams:
    """
    Object intended to provide all relevant information to the propagate function
    """
    def __init__(self, dt: float, epoch_i: Time):
        self.dt = dt
        self.epoch = epoch_i
        self.perturbations = {}

    def add_perturbation(self, name, perturbation):
        self.perturbations[name] = perturbation


class J2:
    """
    Object intended to include all values required by poliastro/core/perturbation.py J2_perturbation()
    """
    def __init__(self, J2: float, R: float):
        self.J2 = J2
        self.R = R


class J3:
    """
    Object intended to include all values required by poliastro/core/perturbation.py J3_perturbation()
    """
    def __init__(self, J3: float, R: float):
        self.J3 = J3
        self.R = R


class Drag:
    """
    Object intended to include all values required by poliastro/core/perturbation.py atmospheric_drag()
    """
    def __init__(self, R: float, C_D: float, A: float, m: float, H0: float, rho0: float):
        self.R = R
        self.C_D = C_D
        self.A = A
        self.m = m
        self.H0 = H0
        self.rho0 = rho0


class ThirdBody:
    """
    Object intended to include all values required by poliastro/core/perturbation.py Third_Body(). Can be used for
    Lunar and Solar gravity.
    """
    def __init__(self, k_third: float, third_body):
        self.k_third = k_third
        self.third_body = third_body        # Build from ephemeris


class SRP:
    """
    Object intended to include all values require by poliastro/core/perturbation.py radiation_pressure().
    A_over_m is the new implementation. May lead to issues.
    """
    def __init__(self, R: float, C_R: float, A: float, m: float, Wdivc_s: float, star):
        self.R = R
        self.C_R = C_R
        self.A = A
        self.m = m
        self.Wdivc_s = Wdivc_s
        self.star = star                    # Build from ephemeris
