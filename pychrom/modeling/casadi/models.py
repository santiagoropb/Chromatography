from pychrom.modeling.casadi.discretization import backward_dcdx, backward_dcdx2
from pychrom.modeling.casadi.smoothing import smooth_named_functions
from pychrom.modeling.results_object import ResultsDataSet
import casadi as ca
import numpy as np
import xarray as xr
import logging
import abc


logger = logging.getLogger(__name__)


class CasadiModel(object):

    def __init__(self):
        self.x = None
        self.s = None
        self.r = None
        self.grid_t = None

        self.ns = None
        self.nx = None
        self.nr = None

        self.x_to_id = None
        self.s_to_id = None
        self.r_to_id = None

        self.id_to_x = None
        self.id_to_s = None
        self.id_to_r = None

        self.t = None
        self.C = None
        self.Cp = None
        self.Q = None

        self.states = None
        self.algebraics = None
        self.ode = None
        self.alg_eq = None
        self.init_conditions = None
        self.parameters = None


class CasadiColumn(object):

    def __init__(self, column):
        """

        :param model: chromatography model
        :type model: ChromatographyModel
        """

        self._column = column
        self._inlet = getattr(self._column._model(), self._column.left_connection)
        self._outlet = getattr(self._column._model(), self._column.right_connection)

        self.m = CasadiModel()

    def setup_base(self, lspan, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with casadi
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """

        """
                Creates base sets and params for modeling a chromatography column with casadi
                :param tspan:
                :param lspan:
                :param rspan:
                :return: None
                """

        # defines space discrete points
        self.m.x = [x for x in lspan]

        if rspan is not None:
            self.m.r = [r for r in rspan]
        else:
            self.m.r = []

        # defines list components
        self.m.s = self._column.list_components()

        # defines lengths
        self.m.nx = len(self.m.x)
        self.m.nr = len(self.m.r)
        self.m.ns = len(self.m.s)

        # define maps to indices
        self.m.x_to_id = dict()
        self.m.id_to_x = dict()
        for i, x in enumerate(self.m.x):
            self.m.x_to_id[x] = i
            self.m.id_to_x[i] = x

        self.m.s_to_id = dict()
        self.m.id_to_s = dict()
        for i, s in enumerate(self.m.s):
            self.m.s_to_id[s] = i
            self.m.id_to_s[i] = s

        self.m.r_to_id = dict()
        self.m.id_to_r = dict()
        for i, r in enumerate(self.m.r):
            self.m.r_to_id[r] = i
            self.m.id_to_r[i] = r

        # defines container for states
        self.m.states = []

        # defines container for algebraics
        self.m.algebraics = []

        # defines container for odes
        self.m.ode = []

        # defines container for algebraic equations
        self.m.alg_eq = []

        # defines container with initial conditions
        self.m.init_conditions = []

        # defines container with parameters
        self.parameters = []

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with casadi
        :return: None
        """
        # time variable
        self.m.t = ca.SX.sym("t")

        # velocity parameter
        self.m.v = ca.SX.sym("v")

        # dispersion
        self.m.diff = ca.SX.sym("diff")

        # append parameters
        self.m.parameters = [self.m.v, self.m.diff]

    @abc.abstractmethod
    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """

    @abc.abstractmethod
    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """

    @abc.abstractmethod
    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """

    @abc.abstractmethod
    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

    @abc.abstractmethod
    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

    @abc.abstractmethod
    def initialize_variables(self, trajectories=None):
        """

        :param trajectories: results set
        :return:
        """

    @abc.abstractmethod
    def store_values_in_data_set(self, solution):
        """
        store variable values in results object
        :return: ResultsDataSet
        """

    def create_inlet_functions(self):

        section_functions = dict()
        break_times = []

        end_time = 0.0
        for n, sec in self._inlet.sections(ordered=True):
            end_time = sec.start_time_sec
            break_times.append(sec.start_time_sec)

        end_time += 100  # easy fix
        break_times.append(end_time)

        for name in self._column.list_components():
            list_functions = list()
            for n, sec in self._inlet.sections(ordered=True):
                list_functions.append(sec.f)
                component_function = smooth_named_functions(list_functions, break_times, name, tee=False)
            section_functions[name] = component_function

        return section_functions

    def build_model(self, lspan=None, rspan=None, **kwargs):
        """

        :return:
        """
        # create sets and parameters
        self.setup_base(lspan, rspan)
        # create variables
        self.build_variables(**kwargs)
        # create constraints
        self.build_mobile_phase_balance(**kwargs)
        self.build_stationary_phase_balance(**kwargs)
        self.build_adsorption_equations(**kwargs)
        self.build_boundary_conditions(**kwargs)
        self.build_initial_conditions(**kwargs)

    def model(self):
        """
        Return an instance of the pyomo model for the chromatography column
        :return: casadi model
        """
        return self.m


class ConvectionModel(CasadiColumn):

    def __init__(self, column):
        super().__init__(column)

    def setup_base(self, lspan, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with casadi
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """

        super().setup_base(lspan, rspan, **kwargs)

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

        super().build_variables(**kwargs)

        # concentration variable
        self.m.C = ca.SX.sym("C", self.m.ns, self.m.nx)

        for i in range(1, self.m.nx):
            for j in range(self.m.ns):
                self.m.states.append(self.m.C[j, i])

        for j in range(self.m.ns):
            self.m.algebraics.append(self.m.C[j, 0])

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """

        for j, x in enumerate(self.m.x):
            if j > 0:
                for s in self.m.s:
                    expr = -self.m.v * backward_dcdx(self.m, s, x)
                    self.m.ode.append(expr)

    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """
        pass

    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """
        pass

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """
        inlet_functions = self.create_inlet_functions()

        for i, s in enumerate(self.m.s):
            expr = inlet_functions[s](s, self.m.t)
            self.m.alg_eq.append(self.m.C[i, 0]-expr)

    def initialize_variables(self, trajectories=None):
        """

        :param trajectories: results set
        :return:
        """
        pass

    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """
        for i in range(1, self.m.nx):
            for s in self.m.s:
                ic = self._column.init_c(s)
                self.m.init_conditions.append(ic)

    def store_values_in_data_set(self, solution):
        """
        store variable values in results object
        :return: ResultsDataSet
        """

        # loading solution to x array
        nt = len(self.m.grid_t)
        nx = self.m.nx
        ns = self.m.ns

        result_set = ResultsDataSet()
        result_set.components = np.array(self.m.s)
        result_set.times = np.array(self.m.grid_t)
        result_set.col_locs = np.array(self.m.x)

        conc = np.zeros((ns, nt, nx))

        for j, t in enumerate(self.m.grid_t):
            for k, x in enumerate(self.m.x):
                for w, s in enumerate(self.m.s):
                    if k > 0:
                        conc[w, j, k] = solution['xf'][(k*ns+w) - ns, j]

        for j, t in enumerate(self.m.grid_t):
            for w, s in enumerate(self.m.s):
                conc[w, j, 0] = solution['zf'][w, j]

        result_set.C = xr.DataArray(conc,
                                    coords=[self.m.s, self.m.grid_t, self.m.x],
                                    dims=['component', 'time', 'location'])
        return result_set


class DispersionModel(CasadiColumn):

    def __init__(self, column):
        super().__init__(column)

    def setup_base(self, lspan, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with casadi
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """

        super().setup_base(lspan, rspan, **kwargs)

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

        super().build_variables(**kwargs)

        # concentration variable
        self.m.C = ca.SX.sym("C", self.m.ns, self.m.nx)

        for i in range(1, self.m.nx-1):
            for j in range(self.m.ns):
                self.m.states.append(self.m.C[j, i])

        for j in range(self.m.ns):
            self.m.algebraics.append(self.m.C[j, 0])

        for j in range(self.m.ns):
            self.m.algebraics.append(self.m.C[j, self.m.nx-1])

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """

        for j, x in enumerate(self.m.x):
            if j != 0 and j != self.m.nx-1:
                for s in self.m.s:
                    if j>1:
                        expr = -self.m.v * backward_dcdx(self.m, s, x) + self.m.diff*backward_dcdx2(self.m, s, x)
                    else:
                        expr = -self.m.v * backward_dcdx(self.m, s, x)
                    self.m.ode.append(expr)

    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """
        pass

    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """
        pass

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """
        inlet_functions = self.create_inlet_functions()

        for i, s in enumerate(self.m.s):
            expr = inlet_functions[s](s, self.m.t)
            self.m.alg_eq.append(self.m.C[i, 0]-expr)

        last_x = self.m.x[-1]
        for i, s in enumerate(self.m.s):
            self.m.alg_eq.append(backward_dcdx(self.m, s, last_x))

    def initialize_variables(self, trajectories=None):
        """

        :param trajectories: results set
        :return:
        """
        pass

    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """
        for i in range(1, self.m.nx-1):
            for s in self.m.s:
                ic = self._column.init_c(s)
                self.m.init_conditions.append(ic)

    def store_values_in_data_set(self, solution):
        """
        store variable values in results object
        :return: ResultsDataSet
        """

        # loading solution to x array
        nt = len(self.m.grid_t)
        nx = self.m.nx
        ns = self.m.ns

        result_set = ResultsDataSet()
        result_set.components = np.array(self.m.s)
        result_set.times = np.array(self.m.grid_t)
        result_set.col_locs = np.array(self.m.x)

        conc = np.zeros((ns, nt, nx))

        for j, t in enumerate(self.m.grid_t):
            for k, x in enumerate(self.m.x):
                for w, s in enumerate(self.m.s):
                    if 0 < k < self.m.nx-1:
                        conc[w, j, k] = solution['xf'][(k*ns+w) - ns, j]

        for j, t in enumerate(self.m.grid_t):
            for w, s in enumerate(self.m.s):
                conc[w, j, 0] = solution['zf'][w, j]

        for j, t in enumerate(self.m.grid_t):
            for w, s in enumerate(self.m.s):
                conc[w, j, self.m.nx-1] = solution['zf'][w+self.m.ns, j]

        result_set.C = xr.DataArray(conc,
                                    coords=[self.m.s, self.m.grid_t, self.m.x],
                                    dims=['component', 'time', 'location'])
        return result_set


