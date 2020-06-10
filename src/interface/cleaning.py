from src.dto import Observation
from src.enums import Frames
from src.frames import lla_to_ecef, ecef_to_eci
import astropy.units as u


def convert_obs_from_lla_to_ecef(observation: Observation) -> Observation:
    """
    Converts Observer location from LLA to ECEF frame before calculations to limit total computational cost.

    :param observation: Observational params relevant to prediction function
    """
    assert observation.frame == Frames.LLA
    observation.frame = Frames.ECEF
    observation.position = lla_to_ecef(observation.position)
    return observation


def convert_obs_from_ecef_to_eci(observation: Observation) -> Observation:
    """
    Converts Observer location from ECEF to ECI frame before calculations to limit total computational cost.

    :param observation: Observational params relevant to prediction function
    """
    assert observation.frame == Frames.ECEF
    observation.frame = Frames.ECI
    observation.position = ecef_to_eci(observation.position, observation.epoch)
    return observation


def convert_obs_from_lla_to_eci(obs_params: Observation) -> Observation:
    """
    Converts Observer location from LLA to ECI frame before calculations to limit total computational cost.

    :param obs_params: Observational params relevant to prediction function
    """
    assert obs_params.frame == Frames.LLA
    obs_params.frame = Frames.ECI
    obs_params.position = ecef_to_eci(lla_to_ecef(obs_params.position), obs_params.epoch)
    return obs_params


def verify_locational_units(obs_params: Observation) -> Observation:
    """
    Units are assumed to be a certain set later down the pipeline. THis function serves to ensure that the correct units
    as being assumed.
    :param obs_params: Observation units relevant to the prediction function
    """
    lla_units = [u.deg, u.deg, u.km]
    spacial_units = [u.km, u.km, u.km]
    if obs_params.frame == Frames.LLA:
        desired_units = lla_units
    else:
        assert(obs_params.frame == Frames.ECI or Frames.ECEF)
        desired_units = spacial_units
    for i in range(3):
        if obs_params.position[i].unit is not desired_units[i]:
            obs_params.position[i] = obs_params.position[i].to(desired_units[i])
    return obs_params
