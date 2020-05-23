from src.dto import ObsParams
from src.enums import Frames
from src.frames import lla_to_ecef
import astropy.units as u


def convert_obs_params_from_lla_to_ecef(obs_params: ObsParams) -> ObsParams:
    """
    Converts Observer location from LLA to ECEF frame before calculations to limit total computational cost.
    :param obs_params: Observational params relevant to prediction function
    """
    assert obs_params.frame == Frames.LLA
    obs_params.frame = Frames.ECEF
    obs_params.position = lla_to_ecef(obs_params.position)
    return obs_params


def verify_units(obs_params: ObsParams) -> ObsParams:
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
            obs_params.position[i].to(desired_units[i]).value
    return obs_params
