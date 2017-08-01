from pychrom.modeling.pyomo.var import define_C_vars, define_Q_vars, define_free_sites_vars
from pychrom.modeling.pyomo.smoothing import PieceWiseNamedFunction, smooth_named_functions
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.results_object import ResultsDataSet
from pychrom.modeling.pyomo.discretization import backward_dcdx, backward_dcdx2
from itertools import combinations
import pyomo.environ as pe
import pyomo.dae as dae
import xarray as xr
import numpy as np
import logging
import abc


logger = logging.getLogger(__name__)


class PyomoColumn(abc.ABC):

    def __init__(self, column):
        """

        :param model: chromatography model
        :type model: ChromatographyModel
        """

        self._column = column
        self._inlet = getattr(self._column._model(), self._column.left_connection)
        self._outlet = getattr(self._column._model(), self._column.right_connection)

        self.m = pe.ConcreteModel()


    @abc.abstractmethod
    def setup_base(self, tspan, lspan=None, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with pyomo
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """

    @abc.abstractmethod
    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

    @abc.abstractmethod
    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

    @abc.abstractmethod
    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

    @abc.abstractmethod
    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
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
    def store_values_in_data_set(self):
        """
        store variable values in results object
        :return: ResultsDataSet
        """

    def build_pyomo_model(self, tspan, lspan=None, rspan=None, **kwargs):
        """

        :return:
        """
        # create sets and parameters
        self.setup_base(tspan, lspan, rspan)
        # create variables
        self.build_variables(**kwargs)
        # create constraints
        self.build_mobile_phase_balance(**kwargs)
        self.build_stationary_phase_balance(**kwargs)
        self.build_adsorption_equations(**kwargs)
        self.build_boundary_conditions(**kwargs)
        self.build_initial_conditions(**kwargs)
        self.build_objective()

    def build_objective(self):
        self.m.obj = pe.Objective(expr=0.0)

    def pyomo_model(self):
        """
        Return an instance of the pyomo model for the chromatography column
        :return: pyomo model
        """
        return self.m


class ConvectionModel(PyomoColumn):

    def __init__(self, column):
        super().__init__(column)

    def setup_base(self, tspan, lspan=None, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with pyomo
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """

        if lspan is None:
            lspan = [0.0, self._column.length]
        else:
            lmax = max(lspan)
            lmin = min(lspan)
            if lmax > self._column.length:
                msg = 'all elements in lspan need to be less than the length '
                msg += 'of the column'
                raise RuntimeError(msg)
            if lmin < 0:
                msg = 'all elements in lspan need to be possitive'
                raise RuntimeError(msg)

        tao = tspan
        z = lspan

        # define sets
        self.m.s = self._column.list_components()
        self.m.t = dae.ContinuousSet(initialize=tao)
        self.m.x = dae.ContinuousSet(initialize=z)

        # define scaling values
        self.m.sc = pe.Param(self.m.s, initialize=1.0, mutable=True)
        self.m.sq = pe.Param(self.m.s, initialize=1.0, mutable=True)

        # for SMA scaling
        bm = self._column.binding_model
        if isinstance(bm, SMABinding):
            lamda = self._column.binding_model.lamda
            self.m.sf = pe.Param(initialize=lamda, mutable=True)
        else:
            self.m.sf = pe.Param(initialize=1.0, mutable=True)

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """
        scale_vars = kwargs.pop('scale_c', False)
        define_C_vars(self.m,
                      scale_vars=scale_vars,
                      with_second_der=False)

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: None
        """

        # mobile phase mass balance
        def rule_mass_balance(m, s, t, x):
            if x == m.x.first() or t == m.t.first():
                return pe.Constraint.Skip
            lhs = m.dCdt[s, t, x]
            rhs = -self._column.velocity*m.dCdx[s, t, x]
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                   self.m.t,
                                                   self.m.x,
                                                   rule=rule_mass_balance)
        return True

    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        inlet_functions = self.create_inlet_functions()
        lin = self.m.x.first()

        def rule_inlet_bc(m, s, tt):
            if tt == m.t.first():
                return pe.Constraint.Skip
            lhs = m.C[s, tt, lin]
            rhs = inlet_functions[s](s, tt)
            return lhs == rhs

        self.m.inlet = pe.Constraint(self.m.s,
                                     self.m.t,
                                     rule=rule_inlet_bc)

    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        t0 = self.m.t.first()

        def rule_init_c(m, s, x):
            return m.C[s, t0, x] == self._column.init_c(s)

        self.m.init_c = pe.Constraint(self.m.s,
                                      self.m.x,
                                      rule=rule_init_c)

    def create_inlet_functions(self, smooth=False):

        section_functions = dict()
        break_times = []

        for n, sec in self._inlet.sections(ordered=True):
            break_times.append(sec.start_time_sec)

        end_time = self.m.t.bounds()[1]
        break_times.append(end_time)

        for name in self._column.list_components():
            list_functions = list()
            for n, sec in self._inlet.sections(ordered=True):
                list_functions.append(sec.f)
            if smooth:
                component_function = smooth_named_functions(list_functions, break_times, name)
            else:
                component_function = PieceWiseNamedFunction(list_functions, break_times, name)
            section_functions[name] = component_function

        return section_functions

    def initialize_variables(self, trajectories=None):

        inlet_functions = self.create_inlet_functions()

        # get the concentration variable
        cvar = 'phi' if hasattr(self.m, 'phi') else 'C'
        pyomo_c_var = getattr(self.m, cvar)

        # defaults initializes all with initial conditions
        if trajectories is None:
            for s in self.m.s:
                for t in self.m.t:
                    for x in self.m.x:
                        # if the variable was not scale sc will be 1.0
                        if x == self.m.x.first():
                            pyomo_c_var[s, t, x].value = inlet_functions[s](s, t)
                        else:
                            pyomo_c_var[s, t, x].value = self._column.init_c(s) / pe.value(self.m.sc[s])

        else:
            if hasattr(trajectories, 'C'):
                for s in self.m.s:
                    Cn = trajectories.C.sel(component=s)
                    for t in self.m.t:
                        for x in self.m.x:
                            val = Cn.sel(time=t, col_loc=x, method='nearest')
                            pyomo_c_var[s, t, x].value = float(val) / pe.value(self.m.sc[s])

    def store_values_in_data_set(self):

        nt = len(self.m.t)
        ns = len(self.m.s)
        nx = len(self.m.x)

        sorted_x = sorted(self.m.x)
        sorted_s = sorted(self.m.s)
        sorted_t = sorted(self.m.t)

        conc = np.zeros((ns, nt, nx))

        for i, s in enumerate(sorted_s):
            for j, t in enumerate(sorted_t):
                for k, x in enumerate(sorted_x):
                    conc[i, j, k] = pe.value(self.m.C[s, t, x])

        result_set = ResultsDataSet()
        result_set.components = np.array(sorted_s)
        result_set.times = np.array(sorted_t)
        result_set.col_locs = np.array(sorted_x)

        # store concentrations
        result_set.C = xr.DataArray(conc,
                                    coords=[result_set.components,
                                            result_set.times,
                                            result_set.col_locs],
                                    dims=['component',
                                          'time',
                                          'col_loc'])
        return result_set


class DispersionModel(ConvectionModel):

    def __init__(self, column):

        super().__init__(column)

    def setup_base(self, tspan, lspan=None, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with pyomo
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """
        super().setup_base(tspan, lspan, rspan, **kwargs)

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

        super().build_variables(**kwargs)

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: None
        """

        u = self._column.velocity
        diff = self._column.dispersion
        # mobile phase mass balance

        def rule_mass_balance(m, s, t, x):

            if x == m.x.first():
                return pe.Constraint.Skip
            if x == m.x.last():
                return pe.Constraint.Skip
            if t == m.t.first():
                return pe.Constraint.Skip

            x_list = sorted(m.x)
            lhs = m.dCdt[s, t, x]
            rhs = -u*m.dCdx[s, t, x] + diff*backward_dcdx2(m, s, t, x, x_list)
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s, self.m.t, self.m.x, rule=rule_mass_balance)
        return True

    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        super().build_boundary_conditions(**kwargs)

        def rule_outlet_bc(m, s, t):
            lout = m.x.last()
            if t == m.t.first():
                return pe.Constraint.Skip
            lhs = m.dCdx[s, t, lout]
            rhs = 0.0
            return lhs == rhs

        self.m.outlet = pe.Constraint(self.m.s, self.m.t, rule=rule_outlet_bc)

    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        super().build_initial_conditions(**kwargs)

    def initialize_variables(self, trajectories=None):

        super().initialize_variables(trajectories)

    def store_values_in_data_set(self):

        result_set = super().store_values_in_data_set()
        return result_set


class IdealConvectiveColumn(ConvectionModel):

    def __init__(self, column):

        super().__init__(column)

    def setup_base(self, tspan, lspan=None, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with pyomo
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """
        super().setup_base(tspan, lspan, rspan, **kwargs)

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

        super().build_variables(**kwargs)

        scale_vars = kwargs.pop('scale_q', False)
        define_Q_vars(self.m,
                      scale_vars=scale_vars,
                      index_radius=False)
        if isinstance(self._column.binding_model, SMABinding):
            define_free_sites_vars(self.m, scale_vars=True, index_radius=False)

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: None
        """

        F =  (1.0 - self._column.column_porosity) / self._column.column_porosity
        u = self._column.velocity

        # mobile phase mass balance
        def rule_mass_balance(m, s, t, x):

            if x == m.x.first():
                return pe.Constraint.Skip
            if t == m.t.first():
                return pe.Constraint.Skip

            lhs = m.dCdt[s, t, x]
            rhs = -u*m.dCdx[s, t, x] - F * self.m.dQdt[s, t, x]
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                   self.m.t,
                                                   self.m.x,
                                                   rule=rule_mass_balance)
        return True

    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """
        bm = self._column.binding_model
        if isinstance(bm, SMABinding):
            self.build_adsorption_equations2(**kwargs)
            return True

        binding = self._column.binding_model

        def rule_adsorption(m, s, t, x):
            if t == m.t.first():
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._column.list_components():
                c_var[n] = m.C[n, t, x]
                q_var[n] = m.Q[n, t, x]

            if self._column.is_salt(s):
                lhs = self.m.Q[s, t, x]
                rhs = binding.f_ads(s, c_var, q_var)
            else:
                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]
                else:
                    lhs = 0.0
                rhs = binding.f_ads(s, c_var, q_var)

            return lhs == rhs

        self.m.adsorption = pe.Constraint(self.m.s, self.m.t, self.m.x, rule=rule_adsorption)

    def build_adsorption_equations2(self, **kwargs):

        binding = self._column.binding_model
        salt_name = self._column.salt

        def rule_adsorption(m, s, t, x):
            if t == m.t.first():
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._column.list_components():
                c_var[n] = m.C[n, t, x]
                q_var[n] = m.Q[n, t, x]

            if s == salt_name:
                lhs = self.m.Q[s, t, x]
                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])

            elif s == 'free_sites':
                lhs = self.m.free_sites[t, x]
                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])

            else:
                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]
                else:
                    lhs = 0.0

                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])
            return lhs == rhs

        species = set(self.m.s)
        species.add('free_sites')
        self.m.adsorption = pe.Constraint(species, self.m.t, self.m.x, rule=rule_adsorption)

        return True

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        super().build_boundary_conditions(**kwargs)

    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        super().build_initial_conditions(**kwargs)

        def rule_init_q(m, s, x):
            return m.Q[s, 0.0, x] == self._column.init_q(s)

        self.m.init_q = pe.Constraint(self.m.s, self.m.x, rule=rule_init_q)


    def initialize_variables(self, trajectories=None):

        super().initialize_variables(trajectories)

        # get the concentration variable
        qvar = 'gamma' if hasattr(self.m, 'gamma') else 'Q'
        pyomo_q_var = getattr(self.m, qvar)

        # defaults initializes all with initial conditions
        if trajectories is None:
            for s in self.m.s:
                for t in self.m.t:
                    for x in self.m.x:
                        pyomo_q_var[s, t, x].value = self._column.init_q(s) / pe.value(self.m.sq[s])
        else:
            if hasattr(trajectories, 'Q'):
                r = self._column.particle_radius
                for s in self.m.s:
                    Qn = trajectories.Q.sel(component=s)
                    for t in self.m.t:
                        for x in self.m.x:
                            val = Qn.sel(time=t, col_loc=x, par_loc=r, method='nearest')
                            pyomo_q_var[s, t, x].value = float(val) / pe.value(self.m.sq[s])

    def store_values_in_data_set(self):
        result_set = super().store_values_in_data_set()

        #self.m.numerator.pprint()
        #self.m.denominator.pprint()

        nt = len(self.m.t)
        ns = len(self.m.s)
        nx = len(self.m.x)

        conc = np.zeros((ns, nt, nx, 1))

        for i, s in enumerate(result_set.components):
            for j, t in enumerate(result_set.times):
                for k, x in enumerate(result_set.col_locs):
                    conc[i, j, k, 0] = pe.value(self.m.Q[s, t, x])

        # store concentrations
        r = self._column.particle_radius
        result_set.par_locs = np.array([r])
        result_set.Q = xr.DataArray(conc,
                                    coords=[result_set.components,
                                            result_set.times,
                                            result_set.col_locs,
                                            result_set.par_locs],
                                    dims=['component',
                                          'time',
                                          'col_loc',
                                          'par_loc'])

        return result_set


class IdealDispersiveColumn(DispersionModel):

    def __init__(self, column):

        super().__init__(column)

    def setup_base(self, tspan, lspan=None, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with pyomo
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """
        super().setup_base(tspan, lspan, rspan, **kwargs)

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

        super().build_variables(**kwargs)

        scale_vars = kwargs.pop('scale_q', False)
        define_Q_vars(self.m,
                      scale_vars=scale_vars,
                      index_radius=False)
        if isinstance(self._column.binding_model, SMABinding):
            define_free_sites_vars(self.m, scale_vars=True, index_radius=False)

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: None
        """

        F = (1.0 - self._column.column_porosity) / self._column.column_porosity
        u = self._column.velocity
        diff = self._column.dispersion

        # mobile phase mass balance

        def rule_mass_balance(m, s, t, x):

            if x == m.x.first():
                return pe.Constraint.Skip
            if x == m.x.last():
                return pe.Constraint.Skip
            if t == m.t.first():
                return pe.Constraint.Skip

            x_list = sorted(m.x)
            lhs = m.dCdt[s, t, x]
            rhs = -u*m.dCdx[s, t, x] - F * self.m.dQdt[s, t, x] + diff*backward_dcdx2(m, s, t, x, x_list)
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                   self.m.t,
                                                   self.m.x,
                                                   rule=rule_mass_balance)
        return True

    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        bm = self._column.binding_model
        if isinstance(bm, SMABinding):
            self.build_adsorption_equations2(**kwargs)
            return True

        binding = self._column.binding_model

        def rule_adsorption(m, s, t, x):
            if t == m.t.first():
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._column.list_components():
                c_var[n] = m.C[n, t, x]
                q_var[n] = m.Q[n, t, x]

            if self._column.is_salt(s):
                lhs = self.m.Q[s, t, x]
                rhs = binding.f_ads(s, c_var, q_var)
            else:
                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]
                else:
                    lhs = 0.0
                rhs = binding.f_ads(s, c_var, q_var)

            return lhs == rhs

        self.m.adsorption = pe.Constraint(self.m.s, self.m.t, self.m.x, rule=rule_adsorption)

    def build_adsorption_equations2(self, **kwargs):

        binding = self._column.binding_model
        salt_name = self._column.salt

        def rule_adsorption(m, s, t, x):
            if t == m.t.first():
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._column.list_components():
                c_var[n] = m.C[n, t, x]
                q_var[n] = m.Q[n, t, x]

            if s == salt_name:
                lhs = self.m.Q[s, t, x]
                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])

            elif s == 'free_sites':
                lhs = self.m.free_sites[t, x]
                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])

            else:
                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]
                else:
                    lhs = 0.0

                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])
            return lhs == rhs

        species = set(self.m.s)
        species.add('free_sites')
        self.m.adsorption = pe.Constraint(species, self.m.t, self.m.x, rule=rule_adsorption)

        return True

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        super().build_boundary_conditions(**kwargs)

    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        super().build_initial_conditions(**kwargs)

        def rule_init_q(m, s, x):
            return m.Q[s, 0.0, x] == self._column.init_q(s)

        self.m.init_q = pe.Constraint(self.m.s, self.m.x, rule=rule_init_q)

    def initialize_variables(self, trajectories=None):

        super().initialize_variables(trajectories)

        # get the concentration variable
        qvar = 'gamma' if hasattr(self.m, 'gamma') else 'Q'
        pyomo_q_var = getattr(self.m, qvar)

        # defaults initializes all with initial conditions
        if trajectories is None:
            for s in self.m.s:
                for t in self.m.t:
                    for x in self.m.x:
                        pyomo_q_var[s, t, x].value = self._column.init_q(s) / pe.value(self.m.sq[s])
        else:
            if hasattr(trajectories, 'Q'):
                r = self._column.particle_radius
                for s in self.m.s:
                    Qn = trajectories.Q.sel(component=s)
                    for t in self.m.t:
                        for x in self.m.x:
                            val = Qn.sel(time=t, col_loc=x, par_loc=r, method='nearest')
                            pyomo_q_var[s, t, x].value = float(val) / pe.value(self.m.sq[s])

    def store_values_in_data_set(self):
        result_set = super().store_values_in_data_set()

        nt = len(self.m.t)
        ns = len(self.m.s)
        nx = len(self.m.x)

        conc = np.zeros((ns, nt, nx, 1))

        for i, s in enumerate(result_set.components):
            for j, t in enumerate(result_set.times):
                for k, x in enumerate(result_set.col_locs):
                    conc[i, j, k, 0] = pe.value(self.m.Q[s, t, x])

        # store concentrations
        r = self._column.particle_radius
        result_set.par_locs = np.array([r])
        result_set.Q = xr.DataArray(conc,
                                    coords=[result_set.components,
                                            result_set.times,
                                            result_set.col_locs,
                                            result_set.par_locs],
                                    dims=['component',
                                          'time',
                                          'col_loc',
                                          'par_loc'])

        return result_set

class OptimalConvectiveColumn(ConvectionModel):

    def __init__(self, column):

        super().__init__(column)

    def setup_base(self, tspan, lspan=None, rspan=None, **kwargs):
        """
        Creates base sets and params for modeling a chromatography column with pyomo
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """
        super().setup_base(tspan, lspan, rspan, **kwargs)

    def build_variables(self, **kwargs):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

        super().build_variables(**kwargs)

        scale_vars = kwargs.pop('scale_q', False)
        define_Q_vars(self.m,
                      scale_vars=scale_vars,
                      index_radius=False)
        if isinstance(self._column.binding_model, SMABinding):
            define_free_sites_vars(self.m, scale_vars=True, index_radius=False)

        n_moments = kwargs.pop('n_moments', 1)

        bm = self._column.binding_model
        no_salt_list = [cname for cname in self.m.s if not bm.is_salt(cname)]
        if n_moments >= 1:
            """
            # defines states for total concentrations
            self.m.Cts = pe.Var(no_salt_list, self.m.t, initialize=1.0)
            self.m.dCts = dae.DerivativeVar(self.m.Cts, wrt=self.m.t)

            def rule_total_concentration(m, s, t):
                if t == m.t.first():
                    return pe.Constraint.Skip
                return m.dCts[s, t] == m.C[s, t, m.x.last()]

            self.m.c_integral = pe.Constraint(no_salt_list, self.m.t, rule=rule_total_concentration)

            def init_cts_rule(m, s):
                return m.Cts[s, m.t.first()] == 0.0

            self.m.init_cts = pe.Constraint(no_salt_list, rule=init_cts_rule)
            """
            # defines states for retention times integral
            self.m.Trs = pe.Var(no_salt_list, self.m.t, initialize=1.0)
            self.m.dTrs = dae.DerivativeVar(self.m.Trs, wrt=self.m.t)

            def rule_total_time(m, s, t):
                if t == m.t.first():
                    return pe.Constraint.Skip
                return m.dTrs[s, t] == m.C[s, t, m.x.last()] * t

            self.m.t_ode = pe.Constraint(no_salt_list, self.m.t, rule=rule_total_time)

            def init_trs_rule(m, s):
                return m.Trs[s, m.t.first()] == 0.0

            self.m.init_trs = pe.Constraint(no_salt_list, rule=init_trs_rule)

            # retention times variable
            self.m.mean_t = pe.Var(no_salt_list, initialize=1.0)

            def rule_retention(m, s):
                return 10.0 * m.mean_t[s] == m.Trs[s, m.t.last()]

            self.m.retention = pe.Constraint(no_salt_list, rule=rule_retention)

        if n_moments >= 2:

            # defines states for standard deviation
            self.m.sigmas = pe.Var(no_salt_list, self.m.t)
            self.m.dsigmas = dae.DerivativeVar(self.m.sigmas, wrt=self.m.t)

            def rule_sigmas(m, s, t):
                if t == m.t.first():
                    return pe.Constraint.Skip
                return m.dsigmas[s, t] == m.C[s, t, m.x.last()] * (t - m.mean_t[s]) ** 2

            self.m.sigma_ode = pe.Constraint(no_salt_list, self.m.t, rule=rule_sigmas)

            def init_sigmas_rule(m, s):
                return m.sigmas[s, m.t.first()] == 0.0

            self.m.init_sigmas = pe.Constraint(no_salt_list, rule=init_sigmas_rule)

            # retention times variable
            self.m.variance_t = pe.Var(no_salt_list, initialize=1.0)

            def rule_variance(m, s):
                return 10.0 * m.variance_t[s] == m.sigmas[s, m.t.last()]

            self.m.variance = pe.Constraint(no_salt_list, rule=rule_variance)

        if n_moments ==3:

            # defines states for standard deviation
            self.m.skews = pe.Var(no_salt_list, self.m.t)
            self.m.dskews = dae.DerivativeVar(self.m.skews, wrt=self.m.t)

            def rule_skews(m, s, t):
                if t == m.t.first():
                    return pe.Constraint.Skip
                eps = 1e-4
                std = (m.variance_t[s] ** 2 + eps) ** 0.5
                return m.dskews[s, t] == m.C[s, t, m.x.last()] * ((t - m.mean_t[s]) / std) ** 3

            self.m.skew_integral = pe.Constraint(no_salt_list, self.m.t, rule=rule_skews)

            def init_skew_rule(m, s):
                return m.skews[s, m.t.first()] == 0.0

            self.m.init_skews = pe.Constraint(no_salt_list, rule=init_skew_rule)

            # skew variable
            self.m.skew_t = pe.Var(no_salt_list, initialize=1.0)

            def rule_skewness(m, s):
                return 10.0 * m.skew_t[s] == m.skews[s, m.t.last()]

            self.m.skewness = pe.Constraint(no_salt_list, rule=rule_skewness)

        if n_moments > 3:
            raise RuntimeError("The number of moments needs to be less or equal to 3")

    def build_mobile_phase_balance(self, **kwargs):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: None
        """

        F =  (1.0 - self._column.column_porosity) / self._column.column_porosity
        u = self._column.velocity

        # mobile phase mass balance
        def rule_mass_balance(m, s, t, x):

            if x == m.x.first():
                return pe.Constraint.Skip
            if t == m.t.first():
                return pe.Constraint.Skip

            lhs = m.dCdt[s, t, x]
            rhs = -u*m.dCdx[s, t, x] - F * self.m.dQdt[s, t, x]
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                   self.m.t,
                                                   self.m.x,
                                                   rule=rule_mass_balance)
        return True

    def build_stationary_phase_balance(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_adsorption_equations(self, **kwargs):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """
        bm = self._column.binding_model
        if isinstance(bm, SMABinding):
            self.build_adsorption_equations2(**kwargs)
            return True

        binding = self._column.binding_model

        def rule_adsorption(m, s, t, x):
            if t == m.t.first():
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._column.list_components():
                c_var[n] = m.C[n, t, x]
                q_var[n] = m.Q[n, t, x]

            if self._column.is_salt(s):
                lhs = self.m.Q[s, t, x]
                rhs = binding.f_ads(s, c_var, q_var)
            else:
                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]
                else:
                    lhs = 0.0
                rhs = binding.f_ads(s, c_var, q_var)

            return lhs == rhs

        self.m.adsorption = pe.Constraint(self.m.s, self.m.t, self.m.x, rule=rule_adsorption)

    def build_adsorption_equations2(self, **kwargs):

        binding = self._column.binding_model
        salt_name = self._column.salt

        def rule_adsorption(m, s, t, x):
            if t == m.t.first():
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._column.list_components():
                c_var[n] = m.C[n, t, x]
                q_var[n] = m.Q[n, t, x]

            if s == salt_name:
                lhs = self.m.Q[s, t, x]
                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])

            elif s == 'free_sites':
                lhs = self.m.free_sites[t, x]
                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])

            else:
                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]
                else:
                    lhs = 0.0

                rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])
            return lhs == rhs

        species = set(self.m.s)
        species.add('free_sites')
        self.m.adsorption = pe.Constraint(species, self.m.t, self.m.x, rule=rule_adsorption)

        return True

    def build_boundary_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        #super().build_boundary_conditions(**kwargs)

        inlet_functions = self.create_inlet_functions()
        lin = self.m.x.first()

        bm = self._column.binding_model
        no_salt_list = [cname for cname in self.m.s if not bm.is_salt(cname)]

        if len(no_salt_list) == len(self.m.s):
            salt_name = "no_salt_name"
        else:
            sno = set(no_salt_list)
            sw = set(list(self.m.s))
            ns = sw.difference(sno)
            salt_name = ns.pop()

        def rule_inlet_bc(m, s, tt):
            if tt == m.t.first():
                return pe.Constraint.Skip
            if bm.is_salt(s):
                return 50.0 <= m.C[s, tt, lin] <= 80.0

            lhs = m.C[s, tt, lin]
            rhs = inlet_functions[s](s, tt)
            return lhs == rhs

        self.m.inlet = pe.Constraint(self.m.s,
                                     self.m.t,
                                     rule=rule_inlet_bc)

    def build_objective(self):
        bm = self._column.binding_model
        no_salt_list = [cname for cname in self.m.s if not bm.is_salt(cname)]
        if isinstance(bm, SMABinding):
            # combination of components into tuples
            combined_components = list(combinations(no_salt_list, 2))
            n_combinations = len(combined_components)
            self.m.resolution = pe.Var(combined_components)
            self.m.list_resolutions = pe.ConstraintList()

            obj = 0.0
            for i, p in enumerate(combined_components):
                s1 = p[0]
                s2 = p[1]
                eps = 1e-5
                w1 = 4 * (self.m.variance_t[s1] ** 2 + eps) ** 0.5
                w2 = 4 * (self.m.variance_t[s2] ** 2 + eps) ** 0.5
                self.m.list_resolutions.add(self.m.resolution[p] == 2.0 * (self.m.mean_t[s1] - self.m.mean_t[s2]) / (w1 + w2))
                obj -= self.m.resolution[p]**2
            self.m.obj = pe.Objective(expr=obj)

    def build_initial_conditions(self, **kwargs):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        super().build_initial_conditions(**kwargs)

        def rule_init_q(m, s, x):
            return m.Q[s, 0.0, x] == self._column.init_q(s)

        self.m.init_q = pe.Constraint(self.m.s, self.m.x, rule=rule_init_q)

    def initialize_variables(self, trajectories=None):

        super().initialize_variables(trajectories)

        # get the concentration variable
        qvar = 'gamma' if hasattr(self.m, 'gamma') else 'Q'
        pyomo_q_var = getattr(self.m, qvar)

        # defaults initializes all with initial conditions
        if trajectories is None:
            for s in self.m.s:
                for t in self.m.t:
                    for x in self.m.x:
                        pyomo_q_var[s, t, x].value = self._column.init_q(s) / pe.value(self.m.sq[s])
        else:
            if hasattr(trajectories, 'Q'):
                r = self._column.particle_radius
                for s in self.m.s:
                    Qn = trajectories.Q.sel(component=s)
                    for t in self.m.t:
                        for x in self.m.x:
                            val = Qn.sel(time=t, col_loc=x, par_loc=r, method='nearest')
                            pyomo_q_var[s, t, x].value = float(val) / pe.value(self.m.sq[s])

    def store_values_in_data_set(self):
        result_set = super().store_values_in_data_set()

        #self.m.numerator.pprint()
        #self.m.denominator.pprint()

        nt = len(self.m.t)
        ns = len(self.m.s)
        nx = len(self.m.x)

        conc = np.zeros((ns, nt, nx, 1))

        for i, s in enumerate(result_set.components):
            for j, t in enumerate(result_set.times):
                for k, x in enumerate(result_set.col_locs):
                    conc[i, j, k, 0] = pe.value(self.m.Q[s, t, x])

        # store concentrations
        r = self._column.particle_radius
        result_set.par_locs = np.array([r])
        result_set.Q = xr.DataArray(conc,
                                    coords=[result_set.components,
                                            result_set.times,
                                            result_set.col_locs,
                                            result_set.par_locs],
                                    dims=['component',
                                          'time',
                                          'col_loc',
                                          'par_loc'])

        if hasattr(self.m, 'mean_t'):
            result_set.mean_t = dict()
            for k, v in self.m.mean_t.items():
                result_set.mean_t[k] = v.value
        if hasattr(self.m, 'variance_t'):
            result_set.variance_t = dict()
            for k, v in self.m.variance_t.items():
                result_set.variance_t[k] = v.value
        if hasattr(self.m, 'skew_t'):
            result_set.skew_t = dict()
            for k, v in self.m.skew_t.items():
                result_set.skew_t[k] = v.value

        return result_set