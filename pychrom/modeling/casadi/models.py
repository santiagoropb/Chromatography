from pychrom.modeling.casadi.discretization import backward_dcdx, backward_dcdx2
from pychrom.modeling.casadi.smoothing import smooth_named_functions
from pychrom.modeling.results_object import ResultsDataSet
from pychrom.core.binding_model import SMABinding, LinearBinding
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
        self.q_expressions = None

        # for easier ordering
        self.c_states = None
        self.c_algebraics = None
        self.c_ode = None
        self.c_alg_eq = None
        self.c_init_conditions = None

        # for easier ordering
        self.q_states = None
        self.q_algebraics = None
        self.q_ode = None
        self.q_alg_eq = None
        self.q_init_conditions = None

        # for sma
        self.q_free_sites = None
        self.q_free_sites_expr = None

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
        self.m.c_states = []
        self.m.q_states = []

        # defines container for algebraics
        self.m.algebraics = []
        self.m.c_algebraics = []
        self.m.q_algebraics = []

        # defines container for odes
        self.m.ode = []
        self.m.c_ode = []
        self.m.q_ode = []

        # defines container for algebraic equations
        self.m.alg_eq = []
        self.m.c_alg_eq = []
        self.m.q_alg_eq = []

        # define containers for sma
        self.m.q_free_sites_expr = []
        self.m.q_free_sites = []

        # defines container with initial conditions
        self.m.init_conditions = []
        self.m.c_init_conditions = []
        self.m.q_init_conditions = []

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

    def build_model(self, lspan, rspan=None, **kwargs):
        """

        :return:
        """
        # create sets and parameters
        self.setup_base(lspan, rspan)
        # create variables
        self.build_variables(**kwargs)
        # create constraints
        self.build_boundary_conditions(**kwargs)
        self.build_adsorption_equations(**kwargs)
        self.build_mobile_phase_balance(**kwargs)
        self.build_stationary_phase_balance(**kwargs)
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
                self.m.c_states.append(self.m.C[j, i])

        for j in range(self.m.ns):
            self.m.c_algebraics.append(self.m.C[j, 0])

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """

        for j, x in enumerate(self.m.x):
            if j > 0:
                for s in self.m.s:
                    expr = -self.m.v * backward_dcdx(self.m, s, x)
                    self.m.c_ode.append(expr)

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
            self.m.c_alg_eq.append(self.m.C[i, 0]-expr)

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
                self.m.c_init_conditions.append(ic)

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
                                    dims=['component', 'time', 'col_loc'])
        return result_set

    def build_model(self, lspan, rspan=None, **kwargs):
        """

        :return:
        """
        super().build_model(lspan)

        # concatenate state variables
        for e in self.m.c_states:
            self.m.states.append(e)

        # concatenate algebraic variables
        for e in self.m.c_algebraics:
            self.m.algebraics.append(e)

        # concatenate odes
        for e in self.m.c_ode:
            self.m.ode.append(e)

        # concatenate algebraic equations
        for e in self.m.c_alg_eq:
            self.m.alg_eq.append(e)


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
                self.m.c_states.append(self.m.C[j, i])

        for j in range(self.m.ns):
            self.m.c_algebraics.append(self.m.C[j, 0])

        for j in range(self.m.ns):
            self.m.c_algebraics.append(self.m.C[j, self.m.nx-1])

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
                    self.m.c_ode.append(expr)

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
            self.m.c_alg_eq.append(self.m.C[i, 0]-expr)

        last_x = self.m.x[-1]
        for i, s in enumerate(self.m.s):
            self.m.c_alg_eq.append(backward_dcdx(self.m, s, last_x))

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
                self.m.c_init_conditions.append(ic)

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
                                    dims=['component', 'time', 'col_loc'])
        return result_set

    def build_model(self, lspan, rspan=None, **kwargs):
        """

        :return:
        """
        super().build_model(lspan)

        # concatenate state variables
        for e in self.m.c_states:
            self.m.states.append(e)

        # concatenate algebraic variables
        for e in self.m.c_algebraics:
            self.m.algebraics.append(e)

        # concatenate odes
        for e in self.m.c_ode:
            self.m.ode.append(e)

        # concatenate algebraic equations
        for e in self.m.c_alg_eq:
            self.m.alg_eq.append(e)


class IdealConvectiveColumn(ConvectionModel):

    def __init__(self, column):
        super().__init__(column)
        self.dc_dt_factor = dict()
        self.dc_dt_rhs_addition = dict()
        self.dq_dt_expression = dict()

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
        self.m.Q = ca.SX.sym("Q", self.m.ns, self.m.nx)

        salt_name = self._column.salt

        if salt_name is None:
            salt_name = ''

        if salt_name != '':
            self.m.fs = ca.SX.sym("FS", self.m.nx)

        s_no_salt = [s for s in self.m.s if s != salt_name]
        if self._column.binding_model.is_kinetic:
            # define states for all components that are not salt
            for i in range(self.m.nx):
                for s in s_no_salt:
                    j = self.m.s_to_id[s]
                    self.m.q_states.append(self.m.Q[j, i])

            # define algebraic for salt component
            if salt_name != '':
                j = self.m.s_to_id[salt_name]
                for i in range(self.m.nx):
                    self.m.q_algebraics.append(self.m.Q[j, i])
                    self.m.q_free_sites.append(self.m.fs[i])

        else:
            # define algebraic for all components
            for i in range(self.m.nx):
                for j in range(self.m.ns):
                    self.m.q_algebraics.append(self.m.Q[j, i])

            if salt_name != '':
                for i in range(self.m.nx):
                    self.m.q_free_sites.append(self.m.fs[i])

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """

        bm = self._column.binding_model
        is_kinetic = self._column.binding_model.is_kinetic
        if is_kinetic:
            for j, x in enumerate(self.m.x):
                if j > 0:
                    for k, s in enumerate(self.m.s):
                        expr = -self.m.v * backward_dcdx(self.m, s, x) - self.dq_dt_expression[k, j]
                        self.m.c_ode.append(expr)
        else:
            for j, x in enumerate(self.m.x):
                if j > 0:
                    for k, s in enumerate(self.m.s):
                        expr = -self.m.v * backward_dcdx(self.m, s, x)
                        expr += self.dc_dt_rhs_addition[k, j]
                        expr /= (1.0+self.dc_dt_factor[k, j])
                        self.m.c_ode.append(expr)

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
        F = (1.0 - self._column.column_porosity) / self._column.column_porosity
        bm = self._column.binding_model

        if isinstance(bm, SMABinding):
            self.build_adsorption_equations2(**kwargs)
            return True

        is_kinetic = bm.is_kinetic

        self.m.q_expressions = dict()
        for i in range(self.m.nx):
            c_var = dict()
            q_var = dict()
            for k, s in enumerate(self.m.s):
                c_var[s] = self.m.C[k, i]
                q_var[s] = self.m.Q[k, i]

            for k, s in enumerate(self.m.s):
                expr = bm.f_ads(s, c_var, q_var)
                self.m.q_expressions[k, i] = expr
                self.dq_dt_expression[k, i] = expr

        if is_kinetic:
            for i in range(self.m.nx):
                for k, s in enumerate(self.m.s):
                    k = self.m.s_to_id[s]
                    expr = self.m.q_expressions[k, i]
                    self.m.q_ode.append(expr)
        else:
            for i in range(self.m.nx):
                for k, s in enumerate(self.m.s):
                    expr = self.m.q_expressions[k, i]
                    self.m.q_alg_eq.append(expr)

                    if isinstance(bm, LinearBinding):
                        self.dc_dt_factor[k, i] = bm.ka(s)/bm.kd(s)*F
                        self.dc_dt_rhs_addition[k, i] = 0.0
                    else:
                        self.dc_dt_factor[k, i] = 0.0
                        self.dc_dt_rhs_addition[k, i] = 0.0

    def build_adsorption_equations2(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with casadi
        :return: boolean
        """
        F = (1.0 - self._column.column_porosity) / self._column.column_porosity
        bm = self._column.binding_model
        is_kinetic = bm.is_kinetic
        salt_name = self._column.salt

        s_no_salt = [s for s in self.m.s if s != salt_name]
        self.m.q_expressions = dict()
        f_sites_expressions = dict()
        for i in range(self.m.nx):
            c_var = dict()
            q_var = dict()

            for k, s in enumerate(self.m.s):
                c_var[s] = self.m.C[k, i]
                q_var[s] = self.m.Q[k, i]

            f_sites = self.m.fs[i]
            for k, s in enumerate(self.m.s):
                expr = bm.f_ads_given_free_sites(s, c_var, q_var, f_sites)
                self.m.q_expressions[k, i] = expr
                self.dq_dt_expression[k, i] = expr

            f_sites_expressions[i] = bm.f_ads_given_free_sites('free_sites', c_var, q_var, f_sites)
            self.m.q_free_sites_expr.append(f_sites-f_sites_expressions[i])

        for i in range(self.m.nx):
            k_salt = self.m.s_to_id[salt_name]
            self.dq_dt_expression[k_salt, i] = 0.0
            for s in s_no_salt:
                k = self.m.s_to_id[s]
                self.dq_dt_expression[k_salt, i] -= self.dq_dt_expression[k, i]*bm.nu(s)

        if is_kinetic:
            for i in range(self.m.nx):
                for s in s_no_salt:
                    k = self.m.s_to_id[s]
                    expr = self.m.q_expressions[k, i]
                    self.m.q_ode.append(expr)

            # take care of the salt as algebraic
            k = self.m.s_to_id[salt_name]
            for i in range(self.m.nx):
                expr = self.m.q_expressions[k, i]
                self.m.q_alg_eq.append(expr)

        else:
            for i in range(self.m.nx):
                for k, s in enumerate(self.m.s):
                    expr = self.m.q_expressions[k, i]
                    self.m.q_alg_eq.append(expr)

                    if isinstance(bm, LinearBinding):
                        self.dc_dt_factor[k, i] = bm.ka(s) / bm.kd(s) * F
                        self.dc_dt_rhs_addition[k, i] = 0.0
                    else:
                        self.dc_dt_factor[k, i] = 0.0
                        self.dc_dt_rhs_addition[k, i] = 0.0

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """
        #super().build_boundary_conditions(**kwargs)

        inlet_functions = self.create_inlet_functions()

        for i, s in enumerate(self.m.s):
            expr = inlet_functions[s](s, self.m.t)
            self.m.c_alg_eq.append(self.m.C[i, 0] - expr)

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
        super().build_initial_conditions(**kwargs)

        bm = self._column.binding_model
        is_kinetic = bm.is_kinetic
        salt_name = self._column.salt
        if salt_name is None:
            salt_name = ''

        s_no_salt = [s for s in self.m.s if s != salt_name]
        if is_kinetic:
            for i in range(self.m.nx):
                for s in s_no_salt:
                    ic = self._column.init_q(s)
                    self.m.q_init_conditions.append(ic)

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
                                    dims=['component', 'time', 'col_loc'])
        return result_set

    def build_model(self, lspan, rspan=None, **kwargs):
        """

        :return:
        """
        super().build_model(lspan)

        bm = self._column.binding_model

        # concatenate state variables
        for e in self.m.q_states:
            self.m.states.append(e)

        # concatenate algebraic variables
        for e in self.m.q_algebraics:
            self.m.algebraics.append(e)

        # concatenate odes
        for e in self.m.q_ode:
            self.m.ode.append(e)

        # concatenate algebraic equations
        for e in self.m.q_alg_eq:
            self.m.alg_eq.append(e)

        if isinstance(bm, SMABinding):
            for e in self.m.q_free_sites:
                self.m.algebraics.append(e)

            for e in self.m.q_free_sites_expr:
                print(e)
                self.m.alg_eq.append(e)

        print(self.m.algebraics)

        m = self.m

        """
        print("concentration variables and equations")
        print(len(m.c_states), len(m.c_ode))
        for i, v in enumerate(m.c_states):
            e = m.c_ode[i]
            print(v, e, sep="\t")

        print(len(m.c_algebraics), len(m.c_alg_eq))
        for i, v in enumerate(m.c_algebraics):
            e = m.c_alg_eq[i]
            print(v, e, sep="\t")

        print(" variables and equations")
        print(len(m.q_states), len(m.q_ode))
        for i, v in enumerate(m.q_states):
            e = m.q_ode[i]
            print(v, e, sep="\t")

        print(len(m.q_algebraics), len(m.q_alg_eq))
        for i, v in enumerate(m.q_algebraics):
            e = m.q_alg_eq[i]
            print(v, e, sep="\t")

        """

        print("state variables and odes")
        print(len(m.states), len(m.ode))
        for i, v in enumerate(m.states):
            e = m.ode[i]
            print(v, e, sep="\t")

        print("algebraic variables and equations")
        print(len(m.algebraics), len(m.alg_eq))
        for i, v in enumerate(m.algebraics):
            e = m.alg_eq[i]
            print(v, e, sep="\t")