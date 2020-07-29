from typing import List
import numpy as np
from astropy.time import Time
from src.enums import Angles, Frames


class FilterOutput:
    """
    Object intended to store all relevant information from the filter.
    """
    def __init__(self, x_in=np.zeros(6), epoch=Time("2000-01-01T00:00:00.000", format='isot', scale='utc'),
                 x_out=np.zeros(6), delta_x=np.zeros(6), p=np.zeros((6, 6))):
        self.x_in = x_in
        self.epoch = epoch
        self.x_out = x_out
        self.delta_x = delta_x
        self.p = p

    def tostring(self) -> str:
        """
        Converts Filter Output into a string capable of being printed into a file
        """
        output_string = "x_in:\t" + np.array2string(self.x_in, precision=16)
        self.epoch.format = 'jd'
        output_string += "\n\nepoch:\t" + str(self.epoch.value)
        output_string += "\n\nx_out:\t" + np.array2string(self.x_out, precision=16)
        output_string += "\n\ndeta_x:\t" + np.array2string(self.delta_x, precision=16)
        output_string += "\n\np:\t" + np.array2string(self.p, precision=16)
        return output_string


class Observation:
    """
    Object intended to provide all relevant information to the predictor function.
    """
    def __init__(self, position, frame: Frames, epoch_obs: Time, obs_values: np.ndarray, obs_type: Angles, obs_sigmas=np.ones(2)):
        self.position = position   # If LLA frame, this is a list. If ECI/ECEF this is a numpy array. See cleaning.py
        self.frame = frame
        self.epoch = epoch_obs
        self.obs_values = obs_values
        self.obs_sigmas = obs_sigmas
        self.obs_type = obs_type

    def tostring(self):
        if self.frame == Frames.LLA:
            return "[" + self.epoch.fits + ", [" +\
                   str(self.position[0].value) + ", " + str(self.position[1].value) + ", " + str(self.position[2].value) + \
                   "], " + str(self.obs_values[0]) + "\u00B1" + str(self.obs_sigmas[0]) + ", " + \
                   str(self.obs_values[1]) + "\u00B1" + str(self.obs_sigmas[1]) + "]"
        else:
            return "[" + self.epoch.fits + ", [" +\
                   str(self.position[0]) + ", " + str(self.position[1]) + ", " + str(self.position[2]) + \
                   "], " + str(self.obs_values[0]) + "\u00B1" + str(self.obs_sigmas[0]) + ", " + \
                   str(self.obs_values[1]) + "\u00B1" + str(self.obs_sigmas[1]) + "]"



class PropParams:
    """
    Object intended to provide all relevant information to the propagate function
    """
    def __init__(self, epoch_i: Time):
        self.epoch = epoch_i
        self.perturbations = {}

    def add_perturbation(self, name, perturbation):
        self.perturbations[name] = perturbation

    def tostring(self):
        output = ""
        for key, value in self.perturbations.items():
            if output != "":
                output += ", "
            output += key.value + ""
        return output


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
