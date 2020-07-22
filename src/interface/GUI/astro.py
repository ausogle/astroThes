from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from src.dto import Observation
from astropy.time import Time
import numpy as np
from src.interface.tle_dto import TLE
from src.dto import PropParams, FilterOutput
from src.perturbation_util import *
from src.enums import Perturbations
from src.interface.string_conversions import dms_to_dd
from src.core import milani

tag_string = "Developed by Austin Ogle  Ver. 0.0.1  7/20/2020"


class MainWindow(Screen):
    state = np.zeros(6)
    epoch = Time("2000-01-01T00:00:00.000", format='isot', scale='utc')
    prop_params = PropParams(epoch)
    delta_x_apr = np.zeros(6)
    p_apr = np.zeros((6, 6))
    observations = []

    tag = ObjectProperty(None)

    def on_enter(self, *args):
        self.tag.text = tag_string

        self.state_label.text = "Satellite state:    " + str(self.state)
        self.epoch_label.text = "Epoch: " + self.epoch.fits
        self.params_label.text = "Included Perturbations: " + self.prop_params.tostring()
        self.delta_x_apr_label.text = "A priori \u0394 x:    " + str(self.delta_x_apr)
        self.p_apr_label.text = "A priori covariance:    "
        self.p_apr_value.text = str(self.p_apr)
        for observation in self.observations:
            self.observations_output.text += observation.tostring() + "\n"

    def set_default_values(self):
        self.state = np.zeros(6)
        self.epoch = Time("2000-01-01T00:00:00.000", format='isot', scale='utc')
        self.prop_params = PropParams(self.epoch)
        self.delta_x_apr = np.zeros(6)
        self.p_apr = np.zeros((6, 6))
        self.observations = []
        self.on_enter()

    def run_clicked(self):
        # a_priori = FilterOutput(delta_x=self.delta_x_apr, p=self.p_apr)
        # output = milani(self.state, self.observations, self.prop_params, a_priori)
        # sm.get_screen("results").output = output
        print("hi")


class AddCoreValues(Screen):
    state = ObjectProperty(None)
    tle = ObjectProperty(None)
    is_isot = ObjectProperty(None)
    is_jd = ObjectProperty(None)
    tle_active = ObjectProperty(None)
    include_j2 = ObjectProperty(None)
    include_j3 = ObjectProperty(None)
    include_solar = ObjectProperty(None)
    include_lunar = ObjectProperty(None)
    include_srp = ObjectProperty(None)
    include_drag = ObjectProperty(None)

    srp_A = ObjectProperty(None)
    srp_C_r = ObjectProperty(None)
    srp_m = ObjectProperty(None)
    drag_A = ObjectProperty(None)
    drag_C_d = ObjectProperty(None)
    drag_m = ObjectProperty(None)

    format = "isot"

    def on_enter(self, *args):
        self.epoch.hint_text = "YYYY-MM-DDTHH:MM:SS.SSS"
        self.is_isot.active = True
        self.format = 'isot'
        self.tle.readonly = True

    def j2_active(self, state):
        if state:
            sm.get_screen('main').prop_params.add_perturbation(Perturbations.J2, build_j2())
        else:
            del sm.get_screen('main').prop_params.perturbations[Perturbations.J2]

    def j3_active(self, state):
        if state:
            sm.get_screen('main').prop_params.add_perturbation(Perturbations.J3, build_j3())
        else:
            del sm.get_screen('main').prop_params.perturbations[Perturbations.J3]

    def solar_active(self, active):
        if active:
            if self.validate_values() is True:
                epoch = self.get_epoch_from_box()
                sm.get_screen('main').prop_params.add_perturbation(Perturbations.Sun, build_solar_third_body(epoch))
            else:
                invalid_entry("Ensure epoch meets expected format")
                self.include_solar.active = False
        else:
            try:
                del sm.get_screen('main').prop_params.perturbations[Perturbations.Sun]
            except KeyError:
                thing = "Hih"

    def lunar_active(self, active):
        if active:
            if self.validate_values() is True:
                epoch = self.get_epoch_from_box()
                sm.get_screen('main').prop_params.add_perturbation(Perturbations.Moon, build_lunar_third_body(epoch))
            else:
                invalid_entry("Ensure epoch meets expected format")
                self.include_solar.active = False
        else:
            try:
                del sm.get_screen('main').prop_params.perturbations[Perturbations.Sun]
            except KeyError:
                thing = "Hih"

    def drag_active(self, active):
        if active:
            self.drag_A.readonly = False
            self.drag_A.background_color = 1, 1, 1, 1
            self.drag_A.hint_text_color = .6, .6, .6, 1
            self.drag_C_d.readonly = False
            self.drag_C_d.background_color = 1, 1, 1, 1
            self.drag_C_d.hint_text_color = .6, .6, .6, 1
            self.drag_m.readonly = False
            self.drag_m.background_color = 1, 1, 1, 1
            self.drag_m.hint_text_color = .6, .6, .6, 1
        else:
            self.drag_A.readonly = True
            self.drag_A.background_color = .6, .6, .6, 1
            self.drag_A.hint_text_color = 1, 1, 1, 1
            self.drag_C_d.readonly = True
            self.drag_C_d.background_color = .6, .6, .6, 1
            self.drag_C_d.hint_text_color = 1, 1, 1, 1
            self.drag_m.readonly = True
            self.drag_m.background_color = .6, .6, .6, 1
            self.drag_m.hint_text_color = 1, 1, 1, 1
            try:
                del sm.get_screen('main').prop_params.perturbations[Perturbations.Drag]
            except KeyError:
                thing = "Hih"

    def srp_active(self, active):
        if active:
            self.srp_A.readonly = False
            self.srp_A.background_color = 1, 1, 1, 1
            self.srp_A.hint_text_color = .6, .6, .6, 1
            self.srp_C_r.readonly = False
            self.srp_C_r.background_color = 1, 1, 1, 1
            self.srp_C_r.hint_text_color = .6, .6, .6, 1
            self.srp_m.readonly = False
            self.srp_m.background_color = 1, 1, 1, 1
            self.srp_m.hint_text_color = .6, .6, .6, 1
        else:
            self.srp_A.readonly = True
            self.srp_A.background_color = .6, .6, .6, 1
            self.srp_A.hint_text_color = 1, 1, 1, 1
            self.srp_C_r.readonly = True
            self.srp_C_r.background_color = .6, .6, .6, 1
            self.srp_C_r.hint_text_color = 1, 1, 1, 1
            self.srp_m.readonly = True
            self.srp_m.background_color = .6, .6, .6, 1
            self.srp_m.hint_text_color = 1, 1, 1, 1
            try:
                del sm.get_screen('main').prop_params.perturbations[Perturbations.SRP]
            except KeyError:
                thing = "Hih"

    def get_epoch_from_box(self):
        if self.format == 'jd':
            epoch = Time(float(self.epoch.text), format=self.format, scale='utc')
        elif self.format == 'isot':
            epoch = Time(self.epoch.text, format=self.format, scale='utc')
        return epoch

    def tle_state(self, state):
        if state:
            self.tle.readonly = False
            self.tle.background_color = [1, 1, 1, 1]
            self.tle.hint_text = ""
        else:
            self.tle.readonly = True
            self.tle.background_color = [.6, .6, .6, 1]
            self.tle.hint_text = "This box is read only until the checkbox has been activated"
            self.tle.hint_text_color = [1, 1, 1, 1]

    def isot_active(self):
        self.epoch.hint_text = "YYYY-MM-DDTHH:MM:SS.SSS"
        self.format = 'isot'

    def jd_active(self):
        self.epoch.hint_text = "2000000.000"
        self.format = 'jd'

    def update_values(self):
        if self.validate_values():
            if self.include_srp.active:
                sm.get_screen("main").prop_params.add_perturbation(Perturbations.SRP,
                                                                   build_srp(float(self.srp_C_r.text),
                                                                             float(self.srp_A.text),
                                                                             float(self.srp_m.text),
                                                                             self.get_epoch_from_box()))
            if self.include_drag.active:
                sm.get_screen("main").prop_params.add_perturbation(Perturbations.Drag,
                                                                   build_basic_drag(float(self.drag_C_d.text),
                                                                                    float(self.drag_A.text),
                                                                                    float(self.drag_m.text)))
            if self.tle_active.active is False:
                sm.get_screen('main').state = np.fromstring(self.state.text, sep=",").reshape(6)
                sm.get_screen('main').epoch = self.get_epoch_from_box()
            else:
                tle = TLE.from_lines(self.tle.text)
                sm.get_screen('main').state, sm.get_screen('main').epoch = tle.to_state()
            sm.current = 'main'

    def validate_values(self):
        if self.tle_active.active is False:
            try:
                np.fromstring(self.state.text, sep=",").reshape(6)
            except ValueError:
                invalid_entry("The state entry is invalid. Separate entries by commas.")
                return False
            try:
                if self.format == 'jd':
                    Time(float(self.epoch.text), format=self.format, scale='utc')
                elif self.format == 'isot':
                    Time(self.epoch.text, format=self.format, scale='utc')
            except ValueError:
                invalid_entry("The epoch format is incorrect. See hint")
                return False
            return True
        else:
            try:
                TLE.from_lines(self.tle.text)
            except ValueError:
                invalid_entry("The TLE format is incorrect.")
                return False
            return True


class AddAPrioriValues(Screen):
    delta_x_apr = ObjectProperty(None)
    p_apr = ObjectProperty(None)
    state = ObjectProperty(None)

    def update_values(self):
        if self.state.text == "" and self.p_apr.text == "":
            sm.current = 'main'
        elif self.validate_values() is True:
            sm.get_screen('main').delta_x_apr = np.fromstring(self.state.text, sep=",")
            p_lines = self.p_apr.text.replace("\n", ",")
            p = np.fromstring(p_lines, sep=",").reshape((6, 6))
            sm.get_screen('main').p_apr = p
            sm.current = "main"

    def clear_values(self):
        sm.get_screen('main').delta_x_apr = np.zeros(6)
        sm.get_screen('main').p_apr = np.zeros((6, 6))

    def validate_values(self):
        if self.state.text == "" and self.p_apr.text == "":
            return True
        try:
            p_lines = self.p_apr.text.replace("\n", ",")
            p = np.fromstring(p_lines, sep=",").reshape((6, 6))
        except ValueError:
            invalid_entry("Covariance Matrix input didn't match \nthe expected format. See hint")
            return False
        try:
            np.fromstring(self.state.text, sep=",").reshape((6, 1))
        except ValueError:
            invalid_entry("\u0394 x (a priori) didn't match the expected format. See hint")
            return False
        return True


class AddObservation(Screen):
    new_string = ObjectProperty(None)
    epoch = ObjectProperty(None)
    ra = ObjectProperty(None)
    dec = ObjectProperty(None)
    sigma_ra = ObjectProperty(None)
    sigma_dec = ObjectProperty(None)
    observation_output = ObjectProperty(None)
    is_isot = ObjectProperty(None)
    is_jd = ObjectProperty(None)
    lat = ObjectProperty(None)
    lon = ObjectProperty(None)
    alt = ObjectProperty(None)
    is_dd = ObjectProperty(None)
    is_dms = ObjectProperty(None)

    epoch_format = 'isot'
    location_format = "dd"

    def on_enter(self, *args):
        self.epoch.hint_text = "YYYY-MM-DDTHH:MM:SS.SSS"
        self.is_isot.active = True
        self.epoch_format = 'isot'
        self.is_dd.active = True

    def add_observation(self):
        if self.validate_values() is True:
            obs_pos = self.get_obs_pos()
            obs = Observation(obs_pos, None, self.epoch.text, np.array([float(self.ra.text), float(self.dec.text)]),
                              None, np.array([float(self.sigma_ra.text), float(self.sigma_dec.text)]))
            sm.get_screen('main').observations.append(obs)
            self.clear_values()
        self.observation_output.text = ""
        for observation in sm.get_screen('main').observations:
            self.observation_output.text += observation.tostring() + "\n"

    def clear_observations(self):
        sm.get_screen('main').observations = []
        self.observation_output.text = ""
        sm.get_screen('main').observations_output.text = ""

    def clear_values(self):
        self.lat.text = ""
        self.lon.text = ""
        self.alt.text = ""
        self.epoch.text = ""
        self.ra.text = ""
        self.dec.text = ""
        self.sigma_ra.text = ""
        self.sigma_dec.text = ""

    def validate_values(self):
        try:
            float(self.ra.text)
            float(self.dec.text)
            float(self.sigma_dec.text)
            float(self.sigma_ra.text)
        except ValueError:
            invalid_entry("Invalid entries for measurements and their uncertainties.\n D"
                          "ecimal degrees is the expected format ")
            return False
        try:
            if self.epoch_format == 'jd':
                sm.get_screen('main').epoch = Time(float(self.epoch.text), format=self.epoch_format, scale='utc')
            elif self.epoch_format == 'isot':
                sm.get_screen('main').epoch = Time(self.epoch.text, format=self.epoch_format, scale='utc')
        except ValueError:
            invalid_entry("The epoch format is incorrect. See hint")
            return False
        try:
            if self.is_dd.active is True:
                float(self.lat.text)
                float(self.lon.text)
            else:
                dms_to_dd(self.lat.text)
                dms_to_dd(self.lon.text)
        except ValueError:
            invalid_entry("The lat/lon inputs do not match expected format")
        try:
            float(self.alt.text)
        except ValueError:
            invalid_entry("The alt input does not match expected format")
        return True

    def get_obs_pos(self):
        if self.is_dd.active is True:
            lat = float(self.lat.text)
            lon = float(self.lon.text)
        else:
            lat = dms_to_dd(self.lat.text)
            lon = dms_to_dd(self.lon.text)
        return [lat * u.deg, lon * u.deg, float(self.alt.text) * u.km]

    def isot_active(self):
        self.epoch.hint_text = "YYYY-MM-DDTHH:MM:SS.SSS"
        self.epoch_format = 'isot'

    def jd_active(self):
        self.epoch.hint_text = "2000000.000"
        self.epoch_format = 'jd'

    def dd_active(self):
        self.lat.hint_text = "00.000"
        self.lon.hint_text = "00.000"

    def dms_active(self):
        self.lat.hint_text = "00 00\' 00\""
        self.lon.hint_text = "00 00\' 00\""


class ResultsScreen(Screen):
    output_text = ObjectProperty(None)
    output = FilterOutput()

    def on_enter(self, *args):
        self.output_text.text = self.output.tostring()


def invalid_entry(string):
    pop = Popup(title='Invalid Entry',
                content=Label(text=string),
                size_hint=(None, None), size=(400, 400))

    pop.open()


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("astro.kv")

sm = WindowManager()

screens = [MainWindow(name="main"), AddObservation(name="obs"), AddAPrioriValues(name='apr'),
           AddCoreValues(name='addcore'), ResultsScreen(name='results')]
for screen in screens:
    sm.add_widget(screen)

sm.current = "main"


class BatchLeastSquaresFilterApp(App):
    def build(self):
        return sm


if __name__ == "__main__":
    BatchLeastSquaresFilterApp().run()