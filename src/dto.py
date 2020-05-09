class ObsParams:
    """
    Object intended to provide all relevant information to the predictor function.
    """
    def __init__(self, obs_loc, obs_frame, epoch_i):
        self.obs_loc = obs_loc
        self.obs_frame = obs_frame
        self.epoch_i = epoch_i


class PropParams:
    """
    Object intended to provide all relevant information to the propagate function
    """
    def __init__(self, dt, epoch_f):
        self.dt = dt
        self.epoch = epoch_f
        self.perturbations = {}

    def add_perturbation(self, name, perturbation):
        self.perturbations[name] = perturbation


class J2:
    """
    Object intended to include all values required by poliastro/core/perturbation.py J2_perturbation()
    """
    def __init__(self, J2, R):
        self.J2 = J2
        self.R = R


class J3:
    """
    Object intended to include all values required by poliastro/core/perturbation.py J3_perturbation()
    """
    def __init__(self, J3, R):
        self.J3 = J3
        self.R = R


class Drag:
    """
    Object intended to include all values required by poliastro/core/perturbation.py atmospheric_drag()
    """
    def __init__(self, R, C_D, A, m, H0, rho0):
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
    def __init__(self, k_third, third_body):
        self.k_third = k_third
        self.third_body = third_body


class SRP:
    """
    Object intended to include all values require by poliastro/core/perturbation.py radiation_pressure().
    A_over_m is the new implementation. May lead to issues.
    """
    def __init__(self, R, C_R, A, m, Wdivc_s, star):
        self.R = R
        self.C_R = C_R
        self.A = A
        self.m = m
        self.Wdivc_s = Wdivc_s
        self.star = star
