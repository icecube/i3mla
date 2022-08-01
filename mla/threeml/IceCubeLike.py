"""Docstring"""
from __future__ import print_function
from __future__ import division
from past.utils import old_div
import collections
import scipy
import numpy as np
import numpy.lib.recfunctions as rf
from astromodels import Gaussian_on_sphere
from astromodels.core.sky_direction import SkyDirection
from astromodels.core.spectral_component import SpectralComponent
from astromodels.core.tree import Node
from astromodels.core.units import get_units
from astromodels.sources.source import Source, SourceType
from astromodels.utils.pretty_list import dict_to_list
from astromodels.core.memoization import use_astromodels_memoization
from astromodels import PointSource, ExtendedSource
import astropy.units as u
from threeML.plugin_prototype import PluginPrototype
from mla.threeml import data_handlers
from mla.threeml import sob_terms
from mla import sob_terms as sob_terms_base
from mla import test_statistics
from mla.params import Params
from mla import analysis
from mla import sources
from mla import minimizers
from mla import trial_generators

__all__ = ["NeutrinoPointSource"]
r"""This IceCube plugin is currently under develop by Kwok Lung Fan"""


class NeutrinoPointSource(PointSource):
    """
    Class for NeutrinoPointSource. It is inherited from astromodels PointSource class.
    """

    def __init__(
        self,
        source_name,
        ra=None,
        dec=None,
        spectral_shape=None,
        l=None,
        b=None,
        components=None,
        sky_position=None,
        energy_unit=u.GeV,
    ):
        """Constructor for NeutrinoPointSource

        More info ...

        Args:
            source_name:Name of the source
            ra: right ascension in degree
            dec: declination in degree
            spectral_shape: Shape of the spectrum.Check 3ML example for more detail.
            l: galactic longitude in degree
            b: galactic   in degree
            components: Spectral Component.Check 3ML example for more detail.
            sky_position: sky position
            energy_unit: Unit of the energy
        """
        # Check that we have all the required information

        # (the '^' operator acts as XOR on booleans)

        # Check that we have one and only one specification of the position

        assert (
            (ra is not None and dec is not None)
            ^ (l is not None and b is not None)
            ^ (sky_position is not None)
        ), "You have to provide one and only one specification for the position"

        # Gather the position

        if not isinstance(sky_position, SkyDirection):

            if (ra is not None) and (dec is not None):

                # Check that ra and dec are actually numbers

                try:

                    ra = float(ra)
                    dec = float(dec)

                except (TypeError, ValueError):

                    raise AssertionError(
                        "RA and Dec must be numbers. If you are confused by this message,"
                        " you are likely using the constructor in the wrong way. Check"
                        " the documentation."
                    )

                sky_position = SkyDirection(ra=ra, dec=dec)

            else:

                sky_position = SkyDirection(l=l, b=b)

        self._sky_position = sky_position

        # Now gather the component(s)

        # We need either a single component, or a list of components, but not both
        # (that's the ^ symbol)

        assert (spectral_shape is not None) ^ (components is not None), (
            "You have to provide either a single "
            "component, or a list of components "
            "(but not both)."
        )

        if spectral_shape is not None:

            components = [SpectralComponent("main", spectral_shape)]

        Source.__init__(self, components, src_type=SourceType.POINT_SOURCE)

        # A source is also a Node in the tree

        Node.__init__(self, source_name)

        # Add the position as a child node, with an explicit name

        self._add_child(self._sky_position)

        # Add a node called 'spectrum'

        spectrum_node = Node("spectrum")
        spectrum_node._add_children(list(self._components.values()))

        self._add_child(spectrum_node)

        # Now set the units
        # Now sets the units of the parameters for the energy domain

        current_units = get_units()

        # Components in this case have energy as x and differential flux as y

        x_unit = energy_unit
        y_unit = (energy_unit * current_units.area * current_units.time) ** (-1)

        # Now set the units of the components
        for component in list(self._components.values()):

            component.shape.set_units(x_unit, y_unit)

    def __call__(self, x, tag=None):
        """
        Overwrite the function so it always return 0.
        It is because it should not produce any EM signal.
        """
        if isinstance(x, u.Quantity):
            if isinstance(x, (float, int)):
                return 0 * (u.keV ** -1 * u.cm ** -2 * u.second ** -1)
            return np.zeros((len(x))) * (
                u.keV ** -1 * u.cm ** -2 * u.second ** -1
            )  # It is zero so the unit doesn't matter
        else:
            if isinstance(x, (float, int)):
                return 0
            return np.zeros((len(x)))

    def call(self, x, tag=None):
        """
        Calling the spectrum

        Args:
            x: Energy

        return
            differential flux
        """
        if tag is None:

            # No integration nor time-varying or whatever-varying

            if isinstance(x, u.Quantity):

                # Slow version with units

                results = [
                    component.shape(x) for component in list(self.components.values())
                ]

                # We need to sum like this (slower) because using
                # np.sum will not preserve the units (thanks astropy.units)

                return sum(results)

            else:

                # Fast version without units, where x is supposed to be in the same
                # units as currently defined in units.get_units()

                results = [
                    component.shape(x) for component in list(self.components.values())
                ]

                return np.sum(results, 0)

        else:

            # Time-varying or energy-varying or whatever-varying

            integration_variable, a, b = tag

            if b is None:

                # Evaluate in a, do not integrate

                with use_astromodels_memoization(False):

                    integration_variable.value = a

                    res = self.__call__(x, tag=None)

                return res

            else:

                # Integrate between a and b

                integrals = np.zeros(len(x))

                # TODO: implement an integration scheme avoiding the for loop

                with use_astromodels_memoization(False):

                    reentrant_call = self.__call__

                    for i, e in enumerate(x):

                        def integral(y):

                            integration_variable.value = y

                            return reentrant_call(e, tag=None)

                        # Now integrate
                        integrals[i] = scipy.integrate.quad(integral, a, b, epsrel=1e-5)[
                            0
                        ]

                return old_div(integrals, (b - a))


class NeutrinoExtendedSource(ExtendedSource):
    def __init__(self, source_name, spatial_shape, spectral_shape=None, components=None):

        # Check that we have all the required information
        # and set the units

        current_u = get_units()

        if isinstance(spatial_shape, Gaussian_on_sphere):

            # Now gather the component(s)

            # We need either a single component, or a list of components, but not both
            # (that's the ^ symbol)

            assert (spectral_shape is not None) ^ (components is not None), (
                "You have to provide either a single "
                "component, or a list of components "
                "(but not both)."
            )

            if spectral_shape is not None:

                components = [SpectralComponent("main", spectral_shape)]

            # Components in this case have energy as x and differential flux as y

            diff_flux_units = (current_u.energy * current_u.area * current_u.time) ** (-1)

            # Now set the units of the components
            for component in components:

                component.shape.set_units(current_u.energy, diff_flux_units)

            # Set the units of the brightness
            spatial_shape.set_units(
                current_u.angle, current_u.angle, current_u.angle ** (-2)
            )

        else:

            print("Only support Gaussian_on_sphere")

            raise RuntimeError()

        # Here we have a list of components

        Source.__init__(self, components, SourceType.EXTENDED_SOURCE)

        # A source is also a Node in the tree

        Node.__init__(self, source_name)

        # Add the spatial shape as a child node, with an explicit name
        self._spatial_shape = spatial_shape
        self._add_child(self._spatial_shape)

        # Add the same node also with the name of the function
        # self._add_child(self._shape, self._shape.__name__)

        # Add a node called 'spectrum'

        spectrum_node = Node("spectrum")
        spectrum_node._add_children(list(self._components.values()))

        self._add_child(spectrum_node)

    @property
    def spatial_shape(self):
        """
        A generic name for the spatial shape.
        :return: the spatial shape instance
        """

        return self._spatial_shape

    def get_spatially_integrated_flux(self, energies):

        """
        Returns total flux of source at the given energy
        :param energies: energies (array or float)
        :return: differential flux at given energy
        """

        if not isinstance(energies, np.ndarray):
            energies = np.array(energies, ndmin=1)

        # Get the differential flux from the spectral components

        results = [
            self.spatial_shape.get_total_spatial_integral(energies)
            * component.shape(energies)
            for component in self.components.values()
        ]

        if isinstance(energies, u.Quantity):

            # Slow version with units

            # We need to sum like this (slower) because using
            # np.sum will not preserve the units (thanks astropy.units)

            differential_flux = sum(results)

        else:

            # Fast version without units, where x is supposed to be in the
            # same units as currently defined in units.get_units()

            differential_flux = np.sum(results, 0)

        return differential_flux

    def __call__(self, lon, lat, energies):
        """
        Returns brightness of source at the given position and energy
        :param lon: longitude (array or float)
        :param lat: latitude (array or float)
        :param energies: energies (array or float)
        :return: differential flux at given position and energy
        """

        lat = np.array(lat, ndmin=1)
        lon = np.array(lon, ndmin=1)
        energies = np.array(energies, ndmin=1)
        if isinstance(self.spatial_shape, Gaussian_on_sphere):
            if isinstance(energies, u.Quantity):

                # Slow version with units

                # We need to sum like this (slower) because
                # using np.sum will not preserve the units (thanks astropy.units)

                result = np.zeros((lat.shape[0], energies.shape[0])) * (
                    u.keV ** -1 * u.cm ** -2 * u.second ** -1 * u.degree ** -2
                )

            else:

                # Fast version without units, where x is supposed to be in the
                # same units as currently defined in units.get_units()

                result = np.zeros((lat.shape[0], energies.shape[0]))

        return np.squeeze(result)

    def call(self, energies):
        """Returns total flux of source at the given energy"""
        return self.get_spatially_integrated_flux(energies)

    @property
    def has_free_parameters(self):
        """
        Returns True or False whether there is any parameter in this source
        :return:
        """

        for component in list(self._components.values()):

            for par in list(component.shape.parameters.values()):

                if par.free:

                    return True

        for par in list(self.spatial_shape.parameters.values()):

            if par.free:

                return True

        return False

    @property
    def free_parameters(self):
        """
        Returns a dictionary of free parameters for this source
        We use the parameter path as the key because it's
        guaranteed to be unique, unlike the parameter name.
        :return:
        """
        free_parameters = collections.OrderedDict()

        for component in list(self._components.values()):

            for par in list(component.shape.parameters.values()):

                if par.free:

                    free_parameters[par.path] = par

        for par in list(self.spatial_shape.parameters.values()):

            if par.free:

                free_parameters[par.path] = par

        return free_parameters

    @property
    def parameters(self):
        """
        Returns a dictionary of all parameters for this source.
        We use the parameter path as the key because it's
        guaranteed to be unique, unlike the parameter name.
        :return:
        """
        all_parameters = collections.OrderedDict()

        for component in list(self._components.values()):

            for par in list(component.shape.parameters.values()):

                all_parameters[par.path] = par

        for par in list(self.spatial_shape.parameters.values()):

            all_parameters[par.path] = par

        return all_parameters

    def _repr__base(self, rich_output=False):
        """
        Representation of the object
        :param rich_output: if True, generates HTML, otherwise text
        :return: the representation
        """

        # Make a dictionary which will then be transformed in a list

        repr_dict = collections.OrderedDict()

        key = "%s (extended source)" % self.name

        repr_dict[key] = collections.OrderedDict()
        repr_dict[key]["shape"] = self._spatial_shape.to_dict(minimal=True)
        repr_dict[key]["spectrum"] = collections.OrderedDict()

        for component_name, component in list(self.components.items()):
            repr_dict[key]["spectrum"][component_name] = component.to_dict(minimal=True)

        return dict_to_list(repr_dict, rich_output)

    def get_boundaries(self):
        """
        Returns the boundaries for this extended source
        :return: a tuple of tuples ((min. lon, max. lon), (min lat, max lat))
        """
        return self._spatial_shape.get_boundaries()


class Spectrum(object):
    r"""
    A class that converter a astromodels model
    instance to a spectrum object with __call__ method.
    """

    def __init__(self, likelihood_model_instance, A=1):
        r"""Constructor of the class"""
        self.model = likelihood_model_instance
        self.norm = A
        for source_name, source in likelihood_model_instance.point_sources.items():
            if isinstance(source, NeutrinoPointSource):
                self.neutrinosource = source_name
                self.point = True
        for source_name, source in likelihood_model_instance.extended_sources.items():
            if isinstance(source, NeutrinoExtendedSource):
                self.neutrinosource = source_name
                self.point = False

    def __call__(self, energy, **kwargs):
        r"""Evaluate spectrum at E"""
        if self.point:
            return self.model.point_sources[self.neutrinosource].call(energy) * self.norm
        else:
            return (
                self.model.extended_sources[self.neutrinosource].call(energy) * self.norm
            )

    def validate(self):
        pass

    def __str__(self):
        r"""String representation of class"""
        return "SpectrumConverter class doesn't support string representation now"

    def copy(self):
        r"""Return copy of this class"""
        c = type(self).__new__(type(self))
        c.__dict__.update(self.__dict__)
        return c


class IceCubeLike(PluginPrototype):
    def __init__(
        self,
        name: str,
        data: np.ndarray,
        data_handlers: data_handlers.ThreeMLDataHandler,
        llh: test_statistics.LLHTestStatisticFactory,
        source: sources.PointSource = None,
        livetime: float = None,
        fix_flux_norm: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        r"""Constructor of the class.
        Args:
            name: name for the plugin
            data: data of experiment
            data_handlers: mla.threeml.data_handlers ThreeMLDataHandler object
            llh: test_statistics.LLHTestStatistic object. Used to evaluate the ts
            source: injection location(only when need injection)
            livetime: livetime(calculated using livetime within time profile if None)
            fix_flux_norm: only fit the spectrum shape
            verbose: print the output or not

        """
        nuisance_parameters = {}
        super(IceCubeLike, self).__init__(name, nuisance_parameters)
        self.parameter = kwargs
        self.fix_flux_norm = fix_flux_norm
        self.fit_ns = 0
        self.fit_likelihood = 0
        if livetime is None:
            for term in llh.sob_term_factories:
                if isinstance(term, sob_terms_base.TimeTermFactory):
                    self.livetime = (
                        data_handlers.contained_livetime(
                            term.signal_time_profile.range[0],
                            term.signal_time_profile.range[1],
                        )
                        * 3600
                        * 24
                    )
        else:
            self.livetime = livetime
        if source is None:
            config = sources.PointSource.generate_config()
            config["ra"] = 0
            config["dec"] = 0
            source = sources.PointSource(config=config)
        self.injected_source = source
        trial_config = trial_generators.SingleSourceTrialGenerator.generate_config()
        self.trial_generator = trial_generators.SingleSourceTrialGenerator(
            trial_config, data_handlers, source
        )
        analysis_config = analysis.SingleSourceLLHAnalysis.generate_default_config(
            minimizer_class=minimizers.GridSearchMinimizer
        )
        self.analysis = analysis.SingleSourceLLHAnalysis(
            config=analysis_config,
            minimizer_class=minimizers.GridSearchMinimizer,
            sob_term_factories=llh.sob_term_factories,
            data_handler_source=(data_handlers, source),
        )

        self.sob_term_factories = llh.sob_term_factories
        for term in llh.sob_term_factories:
            if isinstance(term, sob_terms_base.SpatialTermFactory):
                self.spatial_sob_factory = term
            if isinstance(term, sob_terms.ThreeMLPSEnergyTermFactory0:
                self.energy_sob_factory = term
        self.verbose = verbose
        self._data = data
        self._ra = np.rad2deg(source.config["ra"])
        self._dec = np.rad2deg(source.config["dec"])
        self._sigma = np.rad2deg(source.sigma)
        self.test_statistic = self.analysis.test_statistic_factory(
            Params.from_dict({"ns": 0}), data
        )
        for key in self.test_statistic.sob_terms.keys():
            if isinstance(self.test_statistic.sob_terms[key], sob_terms.ThreeMLPSEnergyTerm):
                self.energyname = key
        return

    def set_model(self, likelihood_model_instance):
        r"""Setting up the model"""
        if likelihood_model_instance is None:

            return

        for source_name, source in likelihood_model_instance.point_sources.items():
            if isinstance(source, NeutrinoPointSource):
                self.source_name = source_name
                ra = source.position.get_ra()
                dec = source.position.get_dec()
                if self._ra == ra and self._dec == dec:
                    self.llh_model = likelihood_model_instance
                    self.energy_sob_factory.spectrum = Spectrum(likelihood_model_instance)
                    self.test_statistic = self.analysis.test_statistic_factory(
                        Params.from_dict({"ns": 0}), self._data
                    )
                else:
                    self._ra = ra
                    self._dec = dec
                    config = sources.PointSource.generate_config()
                    config["ra"] = np.deg2rad(ra)
                    config["dec"] = np.deg2rad(dec)
                    mlasource = sources.PointSource(config=config)
                    self.analysis.data_handler_source = (
                        self.analysis.data_handler_source[0],
                        mlasource,
                    )
                    self.llh_model = likelihood_model_instance
                    self.energy_sob_factory.source = mlasource
                    self.energy_sob_factory.spectrum = Spectrum(likelihood_model_instance)
                    self.test_statistic = self.analysis.test_statistic_factory(
                        Params.from_dict({"ns": 0}), self._data
                    )
        for source_name, source in likelihood_model_instance.extended_sources.items():
            if isinstance(source, NeutrinoExtendedSource):
                self.source_name = source_name
                ra = source.spatial_shape.lon0.value
                dec = source.spatial_shape.lat0.value
                sigma = source.spatial_shape.sigma.value
                if self._ra == ra and self._dec == dec and self._sigma == sigma:
                    self.llh_model = likelihood_model_instance
                    self.energy_sob_factory.spectrum = Spectrum(likelihood_model_instance)
                    self.test_statistic = self.analysis.test_statistic_factory(
                        Params.from_dict({"ns": 0}), self._data
                    )
                else:
                    self._ra = ra
                    self._dec = dec
                    self._sigma = sigma
                    config = sources.GaussianExtendedSource.generate_config()
                    config["ra"] = np.deg2rad(ra)
                    config["dec"] = np.deg2rad(dec)
                    config["sigma"] = np.deg2rad(sigma)
                    mlasource = sources.GaussianExtendedSource(config=config)
                    self.analysis.data_handler_source = (
                        self.analysis.data_handler_source[0],
                        mlasource,
                    )
                    self.llh_model = likelihood_model_instance
                    self.energy_sob_factory.source = mlasource
                    self.energy_sob_factory.spectrum = Spectrum(likelihood_model_instance)
                    self.test_statistic = self.analysis.test_statistic_factory(
                        Params.from_dict({"ns": 0}), self._data
                    )

        if self.source_name is None:
            print("No point sources in the model")
            return

    def inject_background_and_signal(self, **kwargs) -> None:
        """docstring"""
        self._data = self.trial_generator(**kwargs)
        self.test_statistic = self.analysis.test_statistic_factory(
            Params.from_dict({"ns": 0}), self._data
        )
        return

    def update_data(self, data) -> None:
        """docstring"""
        self._data = data
        self.test_statistic = self.analysis.test_statistic_factory(
            Params.from_dict({"ns": 0}), data
        )
        return

    def update_injection(self, source: sources.PointSource):
        """docstring"""
        self.trial_generator.source = source
        return

    def update_model(self):
        """docstring"""
        spectrum = Spectrum(self.llh_model)
        self.energy_sob_factory.spectrum = spectrum
        self.test_statistic.sob_terms[self.energyname].update_sob_hist(
            self.energy_sob_factory
        )
        return

    def get_ns(self):
        """docstring"""
        ns = (
            self.energy_sob_factory.spectrum(
                self.analysis.data_handler_source[0].sim["trueE"]
            )
            * self.analysis.data_handler_source[0].sim["ow"]
            * self.livetime
        ).sum()
        return ns

    def get_log_like(self, verbose=None):
        """docstring"""
        if verbose is None:
            verbose = self.verbose
        self.update_model()
        if self.fix_flux_norm:
            llh = self.test_statistic()  # doesn't matter here
            if verbose:
                ns = self.test_statistic.best_ns
                print(ns, llh)
        else:
            ns = self.get_ns()
            if ns > self.test_statistic.n_events:
                if verbose:
                    print(ns, 0)
                return 0
            llh = self.test_statistic([ns], fitting_ns=True)
            self.fit_ns = ns
            self.fit_likelihood = llh
            if verbose:
                print(ns, llh)

        return -llh

    def get_number_of_data_points(self):
        """docstring"""
        return self.test_statistic.n_events

    def get_current_fit_ns(self):
        """docstring"""
        return self.fit_ns

    def inner_fit(self):
        return self.get_log_like()

    @property
    def data(self) -> np.ndarray:
        """Getter for data."""
        return self._data

    @property
    def ra(self) -> float:
        """Getter for ra."""
        return self._ra

    @property
    def dec(self) -> float:
        """Getter for ra."""
        return self._dec


class icecube_analysis(object):
    """Docstring"""

    def __init__(self, listoficecubelike):
        """Docstring"""
        self.listoficecubelike = listoficecubelike
        self.livetime_ratio = []  # livetime ratio between sample
        self.effA_ratio = []
        self.totallivetime = []
        self._p = []
        self.mc_index = []
        self.init_mc_array()

    def init_mc_array(self):
        """Docstring"""
        for i, sample in enumerate(self.listoficecubelike):
            self.livetime_ratio.append(sample.livetime)
            self.effA_ratio.append(
                sample.analysis.data_handler_source[0].sim["weight"].sum()
            )
        self.livetime_ratio = np.array(self.livetime_ratio)
        self.totallivetime = self.livetime_ratio.sum()
        self.livetime_ratio /= np.sum(self.livetime_ratio)
        self.effA_ratio /= np.sum(self.effA_ratio)
        for i, sample in enumerate(self.listoficecubelike):
            sim = sample.analysis.data_handler_source[0].sim
            mc_array = rf.append_fields(
                np.empty(len(sim["weight"])),
                "p",
                sim["weight"] / sim["weight"].sum() * self.livetime_ratio[i],
                usemask=False,
            )
            mc_array = rf.append_fields(
                mc_array,
                "index",
                np.arange(len(mc_array)),
                usemask=False,
            )
            mc_array = rf.append_fields(
                mc_array,
                "sample",
                np.ones(len(sim["weight"])) * i,
                usemask=False,
            )
            self.mc_index.append(mc_array)

        self.mc_index = np.array(self.mc_index)

    def injection(self, n_signal=0, flux_norm=None, poisson=False):
        """docstring"""
        if flux_norm is not None:
            for i, icecubeobject in enumerate(self.listoficecubelike):
                time_intergrated = flux_norm * icecubeobject.livetime
                icecubeobject.trial_generator.config["fixed_ns"] = False
                tempdata = icecubeobject.trial_generator(time_intergrated)
                self.listoficecubelike[i].update_data(tempdata)
        else:
            if poisson:

                ratio_injection = self.livetime_ratio * self.effA_ratio
                ratio_injection = (ratio_injection / ratio_injection.sum()) * n_signal
                for i, icecubeobject in enumerate(self.listoficecubelike):
                    icecubeobject.trial_generator.config["fixed_ns"] = True
                    injection_signal = np.random.poisson(ratio_injection[i])
                    tempdata = icecubeobject.trial_generator(injection_signal)
                    self.listoficecubelike[i].update_data(tempdata)
            else:
                print("No fix number injection implemented")

    def cal_injection_ns(self, flux_norm):
        """Docstring"""
        ns = 0
        for icecubeobject in self.listoficecubelike:
            time_intergrated = flux_norm * icecubeobject.livetime
            tempns = (
                time_intergrated
                * icecubeobject.analysis.data_handler_source[0].sim["weight"].sum()
            )
            ns = ns + tempns
        return ns

    def cal_injection_fluxnorm(self, ns):
        """Docstring"""
        totalweight = 0
        for icecubeobject in self.listoficecubelike:
            tempweight = (
                icecubeobject.analysis.data_handler_source[0].sim["weight"].sum()
                * icecubeobject.livetime
            )
            totalweight = totalweight + tempweight
        return ns / totalweight
