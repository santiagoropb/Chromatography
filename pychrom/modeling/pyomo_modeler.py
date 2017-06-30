from pychrom.modeling.results_object import ResultsDataSet
from pychrom.core.unit_operation import Column
import pyomo.environ as pe
import pyomo.dae as dae
from pychrom.core.registrar import Registrar
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import warnings
import logging
import os

logger = logging.getLogger(__name__)


class PyomoModeler(object):

    def __init__(self, model):
        """

        :param model: chromatography model
        :type model: ChromatographyModel
        """

        self._model = model
        columns = self._model.list_unit_operations(unit_type=Column)
        n_col = len(columns)
        assert n_col == 1, 'PyomoModeler only supports models with 1 column for now'
        if not self._model.is_fully_specified(print_out=True):
            raise RuntimeError('PyomoModeler requires a fully specified chromatography model')

        self._model._sort_sections_by_time()

        self._column = getattr(self._model, columns[0])
        self._inlet = getattr(self._model, self._column.left_connection)
        self._outlet = getattr(self._model, self._column.right_connection)

        self.m = pe.ConcreteModel()

        self.wq = True
        self.dimensionless = True

    def _build_sets_ideal_model(self,
                                 tspan,
                                 lspan=None):

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

        if self.dimensionless:
            u = self._column.velocity
            l = self._column.length
            scale = u/l
            tao = [t*scale for t in tspan]

            scale = 1.0/l
            z = [x*scale for x in lspan]
        else:
            tao = tspan
            z = lspan

        self.m.s = self._model.list_components()
        self.m.t = dae.ContinuousSet(initialize=tao,)
        self.m.x = dae.ContinuousSet(initialize=z)

        self.m.scale_c = pe.Param(self.m.s,
                                  initialize=1.0,
                                  mutable=True)

        self.m.scale_q = pe.Param(self.m.s,
                                  initialize=1.0,
                                  mutable=True)

    def _build_variables_ideal_model(self):

        # auxiliary aliases
        x = self.m.x
        t = self.m.t
        s = self.m.s

        # mobile phase scaled concentration
        self.m.phi = pe.Var(s, t, x)
        self.m.dphidx = dae.DerivativeVar(self.m.phi, wrt=self.m.x)
        self.m.dphidt = dae.DerivativeVar(self.m.phi, wrt=self.m.t)

        def rule_rescale_c(m, i, j, k):
            return m.phi[i, j, k] * self.m.scale_c[i]

        def rule_rescale_dcdx(m, i, j, k):
            return m.dphidx[i, j, k] * self.m.scale_c[i]

        def rule_rescale_dcdt(m, i, j, k):
            return m.dphidt[i, j, k] * self.m.scale_c[i]

        # mobile phase concentration
        self.m.C = pe.Expression(s, t, x, rule=rule_rescale_c)
        self.m.dCdx = pe.Expression(s, t, x, rule=rule_rescale_dcdx)
        self.m.dCdt = pe.Expression(s, t, x, rule=rule_rescale_dcdt)

        self.m.C.pprint()

        if self.wq:
            # stationary phase scaled concentration
            self.m.gamma = pe.Var(s, t, x)
            self.m.dgdt = dae.DerivativeVar(self.m.gamma, wrt=self.m.t)

            def rule_rescale_q(m, i, j, k):
                return m.gamma[i, j, k] * m.scale_q[i]

            def rule_rescale_dqdt(m, i, j, k):
                return m.dgdt[i, j, k] * m.scale_q[i]

            # stationary phase concentration variable
            self.m.Q = pe.Expression(s, t, x, rule=rule_rescale_q)
            self.m.dQdt = pe.Expression(s, t, x, rule=rule_rescale_dqdt)

    def _build_mass_conservation_ideal_model(self):

        F = (1.0 - self._column.column_porosity) / self._column.column_porosity

        # mobile phase mass balance
        def rule_mass_balance(m, s, t, x):
            if x == m.x.bounds()[0] or t == m.t.bounds()[0]:
                return pe.Constraint.Skip
            lhs = m.dCdt[s, t, x] + m.dCdx[s, t, x]
            if self.wq:
                lhs += F*self.m.dQdt[s, t, x]
            rhs = 0.0
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                   self.m.t,
                                                   self.m.x,
                                                   rule=rule_mass_balance)

    def _build_bc_ideal_model(self):

        lin = self.m.x.bounds()[0]
        section = None

        for n, sec in self._inlet.sections():
            if sec.section_id == 0:
                section = sec

        def rule_inlet_bc(m, s, t):
            lhs = m.C[s, t, lin]
            rhs = section.a0(s)
            return lhs == rhs

        self.m.inlet = pe.Constraint(self.m.s,
                                     self.m.t,
                                     rule=rule_inlet_bc)

    def _build_ic_ideal_model(self):

        def rule_init_c(m, s, x):
            if x == self.m.x.bounds()[0]:
                return pe.Constraint.Skip
            return m.C[s, 0.0, x] == self._column.init_c(s)
        self.m.init_c = pe.Constraint(self.m.s,
                                      self.m.x,
                                      rule=rule_init_c)

        if self.wq:
            def rule_init_q(m, s, x):
                return m.Q[s, 0.0, x] == self._column.init_q(s)

            self.m.init_q = pe.Constraint(self.m.s,
                                          self.m.x,
                                          rule=rule_init_q)

    def _build_adsorption_equations(self):

        dl_factor = 1.0
        if self.dimensionless:
            dl_factor = self._column.velocity*self._column.length

        binding = self._column.binding_model
        salt_scale = self.m.scale_q[self._model.salt]

        def rule_adsorption(m, s, t, x):
            if t == 0:
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._model.list_components():
                c_var[n] = m.C[n, t, x]
                q_var[n] = m.Q[n, t, x]

            if self._model.is_salt(s):
                lhs = self.m.Q[s, t, x]
                rhs = binding.f_ads(s, c_var, q_var)
            else:
                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]*dl_factor
                else:
                    lhs = 0.0
                rhs = binding.f_ads(s, c_var, q_var, q_ref=salt_scale)
            return lhs == rhs

        self.m.adsorption = pe.Constraint(self.m.s,
                                          self.m.t,
                                          self.m.x,
                                          rule=rule_adsorption)

    def build_ideal_model(self,
                          tspan,
                          lspan=None,
                          q_scale=None,
                          c_scale=None):

        self._build_sets_ideal_model(tspan,
                                     lspan=lspan)

        # change scales in pyomo model
        if isinstance(c_scale, dict):
            for k, v in c_scale.items():
                self.m.scale_c[k] = v

        if self.wq:
            if isinstance(q_scale, dict):
                for k, v in q_scale.items():
                    self.m.scale_q[k] = v

        self._build_variables_ideal_model()

        self._build_mass_conservation_ideal_model()

        self._build_bc_ideal_model()

        self._build_ic_ideal_model()

        if self.wq:
            self._build_adsorption_equations()

        self.m.pprint()

    def discretize_space_ideal_model(self):

        # Discretize using Finite Difference and Collocation
        discretizer = pe.TransformationFactory('dae.finite_difference')

        discretizer.apply_to(self.m,
                             nfe=50,
                             wrt=self.m.x,
                             scheme='BACKWARD')

    def discretize_time_ideal_model(self):

        discretizer = pe.TransformationFactory('dae.collocation')
        discretizer.apply_to(self.m, nfe=20, ncp=2, wrt=self.m.t)

    def initialize_variables_ideal_model(self, init_trajectories=None):

        L = self._column.length
        u = self._column.velocity
        t_factor = u / L

        if init_trajectories is None:
            for s in self.m.s:
                for t in self.m.t:
                    for x in self.m.x:
                        self.m.phi[s, t, x].value = self._column.init_c(s)/pe.value(self.m.scale_c[s])
                        if self.wq:
                            self.m.gamma[s, t, x].value = self._column.init_q(s)/pe.value(self.m.scale_q[s])
        else:
            for s in self.m.s:
                Cn = init_trajectories.C.sel(component=s)
                Qn = init_trajectories.Q.sel(component=s)
                for t in self.m.t:
                    tt = t*t_factor
                    for x in self.m.x:
                        xx = x/L
                        val = Cn.sel(time=tt, location=xx, method='nearest')
                        self.m.phi[s, t, x].value = float(val)/pe.value(self.m.scale_c[s])
                        val = Qn.sel(time=tt, location=xx, method='nearest')
                        self.m.gamma[s, t, x].value = float(val)/pe.value(self.m.scale_q[s])

    def run_sim(self,
                solver='ipopt',
                solver_opts=None):

        opt = pe.SolverFactory(solver)
        if isinstance(solver_opts, dict):
            for k, v in solver_opts.items():
                opt.options[k] = v

        results = opt.solve(self.m, tee=True)

        return self._parse_results()

    def _parse_results(self):

        nt = len(self.m.t)
        ns = len(self.m.s)
        nx = len(self.m.x)

        sorted_x = sorted(self.m.x)
        sorted_s = sorted(self.m.s)
        sorted_t = sorted(self.m.t)

        conc = np.zeros((ns, nt, nx))
        if self.wq:
            q_array = np.zeros((ns, nt, nx))

        for i, s in enumerate(sorted_s):
            for j, t in enumerate(sorted_t):
                for k, x in enumerate(sorted_x):
                    conc[i, j, k] = pe.value(self.m.C[s, t, x])
                    if self.wq:
                        q_array[i, j, k] = pe.value(self.m.Q[s, t, x])

        result_set = ResultsDataSet()
        result_set.components = np.array(sorted_s)
        if not self.dimensionless:
            result_set.times = np.array(sorted_t)
            result_set.col_locs = np.array(sorted_x)
        else:
            L = self._column.length
            u = self._column.velocity
            t_factor = L/u
            result_set.times = np.array([t*t_factor for t in sorted_t])
            result_set.col_locs = np.array([x*L for x in sorted_x])

        # store concentrations
        result_set.C = xr.DataArray(conc,
                                    coords=[result_set.components,
                                            result_set.times,
                                            result_set.col_locs],
                                    dims=['component',
                                          'time',
                                          'location'])

        if self.wq:
            result_set.Q = xr.DataArray(q_array,
                                        coords=[result_set.components,
                                                result_set.times,
                                                result_set.col_locs],
                                        dims=['component',
                                              'time',
                                              'location'])

        return result_set

