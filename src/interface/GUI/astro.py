from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from src.dto import Observation
from astropy.time import Time
import numpy as np


class MainWindow(Screen):
    state = np.zeros(6)
    epoch = Time("2000-01-01T00:00:00.000", format='isot', scale='utc')
    prop_params = "This is gonna be a mess"
    delta_x_apr = np.zeros(6)
    p_apr = np.zeros((6, 6))
    observations = []

    def on_enter(self, *args):
        self.state_label.text = "Satellite state:    " + str(self.state)
        self.epoch_label.text = "Epoch: " + self.epoch.fits
        self.params_label.text = "Propagation Parameters to be included: " + self.prop_params
        self.delta_x_apr_label.text = "A priori \u0394 x:    " + str(self.delta_x_apr)
        self.p_apr_label.text = "A priori covariance:    "
        self.p_apr_value.text = str(self.p_apr)
        for observation in self.observations:
            self.observations_output.text += observation.tostring() + "\n"

    def set_default_values(self):
        self.state = np.zeros(6)
        self.epoch = "YYYY-MM-DDTHH:MM:SS.SSS"
        self.prop_params = "This is gonna be a mess"
        self.delta_x_apr = np.zeros(6)
        self.p_apr = np.zeros((6, 6))
        self.observations = []
        self.on_enter()

    def run_clicked(self):
        pop = Popup(title='Huge Success',
                          content=Label(text='We did it'),
                          size_hint=(None, None), size=(200, 200))

        pop.open()


class AddCoreValues(Screen):
    state = ObjectProperty
    prop_params = ObjectProperty(None)
    tle = ObjectProperty(None)

    def update_values(self):
        update = True
        try:
            sm.get_screen('main').state = np.fromstring(self.state.text, sep=",").reshape(6)
        except ValueError:
            update = False
            invalid_entry("The state entry is invalid. Separate entries by commas.")
        try:
            sm.get_screen('main').epoch = Time(self.epoch.text, format='isot', scale='utc')
        except ValueError:
            update = False
            invalid_entry("The epoch input requires the ISOT format. See hint")
        try:
            sm.get_screen('main').prop_params = self.prop_params.text
        except ValueError:
            update = False
            invalid_entry("The prop_params is incorrect")
        if update is True:
            sm.current = 'main'
        # or include business about TLE


class AddAPrioriValues(Screen):
    delta_x_apr = ObjectProperty(None)
    p_apr = ObjectProperty(None)
    state = ObjectProperty(None)

    def update_values(self):
        if self.validate_values() is False:
            self.clear_values()
        else:
            sm.get_screen('main').delta_x_apr = np.fromstring(self.state.text, sep=",")
            p_lines = self.p_apr.text.replace("\n", ",")
            p = np.fromstring(p_lines, sep=",").reshape((6, 6))
            sm.get_screen('main').p_apr = p

    def clear_values(self):
        sm.get_screen('main').delta_x_apr = np.zeros(6)
        sm.get_screen('main').p_apr = np.zeros((6, 6))

    def validate_values(self):
        try:
            p_lines = self.p_apr.text.replace("\n", ",")
            p = np.fromstring(p_lines, sep=",").reshape((6, 6))
        except ValueError:
            return False
        if np.fromstring(self.state.text, sep=",").shape != 6:
            return False
        return True


class AddObservation(Screen):
    new_string = ObjectProperty(None)
    obs_pos = ObjectProperty(None)
    epoch = ObjectProperty(None)
    ra = ObjectProperty(None)
    dec = ObjectProperty(None)
    sigma_ra = ObjectProperty(None)
    sigma_dec = ObjectProperty(None)
    observation_output = ObjectProperty(None)

    def add_observation(self):
        if self.validate_values() is True:
            obs = Observation(self.obs_pos.text, None, self.epoch.text, np.array([float(self.ra.text), float(self.dec.text)]),
                              None, np.array([float(self.sigma_ra.text), float(self.sigma_dec.text)]))
            sm.get_screen('main').observations.append(obs)
            self.clear_values()
        self.observation_output.text = ""
        for observation in sm.get_screen('main').observations:
            self.observation_output.text += observation.tostring() + "\n"

    def clear_observations(self):
        sm.get_screen('main').observations = []

    def clear_values(self):
        self.obs_pos.text = ""
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
            return False
        return True


def invalid_entry(string):
    pop = Popup(title='Huge Success',
                content=Label(text=string),
                size_hint=(None, None), size=(400, 400))

    pop.open()


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("astro.kv")

sm = WindowManager()

screens = [MainWindow(name="main"), AddObservation(name="obs"), AddAPrioriValues(name='apr'), AddCoreValues(name='addcore')]
for screen in screens:
    sm.add_widget(screen)

sm.current = "main"


class MyMainApp(App):
    def build(self):
        return sm


if __name__ == "__main__":
    MyMainApp().run()