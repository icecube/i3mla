from threeML.plugin_prototype import PluginPrototype
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from typing import Optional
from astromodels import Model
import numpy as np


class ProfilellhLike(PluginPrototype):
    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        spline: Optional[callable] = None,
        fill_value: Optional[float] = 1e30,
    ):
        nuisance_parameters = {}

        """
        A generic plugin for profile likelihood. Give either a pandas dataframe in format of parameter1,2...,-llh or a spline which return -llh.
        :param name:
        :type name: str
        :param df:
        :type df: pandas dataframe
        :param spline:
        :type spline: callable
        :returns:
        """

        super(ProfilellhLike, self).__init__(name, nuisance_parameters)
        if spline is not None:
            self.spline = spline
            self.df = None
        else:
            self.df = df
            self.par_name = list(df.columns)
            self.par_name.pop()
            listofpoint = []
            shape = []
            for n in self.par_name:
                points = np.unique(df[n])
                listofpoint.append(points)
                shape.append(points.shape[0])
            llh = np.reshape(df["llh"].values, shape)
            self.spline = RegularGridInterpolator(
                listofpoint, llh, bounds_error=False, fill_value=fill_value
            )

    @property
    def likelihood_model(self) -> Model:

        if self._likelihood_model is None:

            log.error(f"plugin {self._name} does not have a likelihood model")

            raise RuntimeError()

        return self._likelihood_model

    def set_model(self, likelihood_model_instance: Model) -> None:
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        :param likelihood_model_instance: instance of Model
        :type likelihood_model_instance: astromodels.Model
        """

        self._likelihood_model = likelihood_model_instance

    def get_log_like(self) -> float:
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """
        current_value = []
        for name in self.par_name:
            value = self._likelihood_model.parameters[name].value
            current_value.append(value)
        llh = -self.spline(current_value)[0]
        return llh

    def inner_fit(self) -> float:
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """

        return self.get_log_like()
