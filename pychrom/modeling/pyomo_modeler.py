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

    def _build_sets_ideal_model(self,
                                 tspan,
                                 lspan=None,
                                 dimensionless=True):

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

        if dimensionless:
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
        self.m.t = dae.ContinuousSet(initialize=tao)
        self.m.x = dae.ContinuousSet(initialize=z)

        self.m.scale_c = pe.Param(self.m.s,
                                  initialize=1.0,
                                  mutable=True)

        self.m.scale_q = pe.Param(self.m.s,
                                  initialize=1.0,
                                  mutable=True)

    def _build_variables_ideal_model(self, with_dispersion=False):

        # auxiliary aliases
        x = self.m.x
        t = self.m.t
        s = self.m.s

        # mobile phase concentration variable
        self.m.C = pe.Var(s, t, x)
        self.m.dCdx = dae.DerivativeVar(self.m.C, wrt=self.m.x)
        if with_dispersion:
            self.m.dCdx2 = dae.DerivativeVar(self.m.C, wrt=(self.m.x, self.m.x))
        self.m.dCdt = dae.DerivativeVar(self.m.C, wrt=self.m.t)
        # stationary phase concentration variable
        self.m.Q = pe.Var(s, t, x)
        self.m.dQdt = dae.DerivativeVar(self.m.Q, wrt=self.m.t)

    def _build_mass_conservation_ideal_model(self,
                                           dimensionless=True,
                                           with_dispersion=False):

        diff_x = self._column.dispersion
        inv_pe = diff_x
        if dimensionless:
            inv_pe /= (self._column.velocity*self._column.length)

        F = (1.0 - self._column.column_porosity) / self._column.column_porosity

        # mobile phase mass balance
        def rule_mass_balance(m, s, t, x):
            if x in m.x.bounds() or t == m.t.bounds()[0]:
                return pe.Constraint.Skip
            lhs = m.scale_c[s]*m.dCdt[s, t, x] + m.scale_c[s]*m.dCdx[s, t, x]
            lhs += m.scale_q[s]*F*self.m.dQdt[s, t, x]
            if with_dispersion:
                rhs = m.scale_c[s]*inv_pe*self.m.dCdx2[s, t, x]
            else:
                rhs = 0.0
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                   self.m.t,
                                                   self.m.x,
                                                   rule=rule_mass_balance)

    def _build_bc_ideal_model(self,
                            dimensionless=True,
                            dankwerts=False,
                            with_dispersion=False):

        diff_x = self._column.dispersion
        if dimensionless:
            diff_x /= self._column.length

        if not dankwerts:
            diff_x = 0.0

        lin = self.m.x.bounds()[0]
        lout = self.m.x.bounds()[1]
        section = None

        for n, sec in self._inlet.sections():
            if sec.section_id == 0:
                section = sec

        def rule_inlet_bc(m, s, t):
            lhs = self._column.velocity*m.C[s, t, lin]*m.scale_c[s]
            rhs = self._column.velocity*section.a0(s) - diff_x*m.dCdx[s, t, lin]*m.scale_c[s]
            return lhs == rhs

        self.m.inlet = pe.Constraint(self.m.s,
                                     self.m.t,
                                     rule=rule_inlet_bc)

        if with_dispersion:
            def rule_outlet_bc(m, s, t):
                return m.dCdx[s, t, lout] == 0.0

            self.m.outlet = pe.Constraint(self.m.s,
                                          self.m.t,
                                          rule=rule_outlet_bc)

    def _build_ic_ideal_model(self):

        def rule_init_c(m, s, x):
            if x in self.m.x.bounds():
                return pe.Constraint.Skip
            return m.scale_c[s]*m.C[s, 0.0, x] == self._column.init_c(s)
        self.m.init_c = pe.Constraint(self.m.s,
                                      self.m.x,
                                      rule=rule_init_c)

        def rule_init_q(m, s, x):
            return m.scale_q[s]*m.Q[s, 0.0, x] == self._column.init_q(s)

        self.m.init_q = pe.Constraint(self.m.s,
                                      self.m.x,
                                      rule=rule_init_q)

    def _build_adsorption_equations(self,
                                  dimensionless=True):

        dl_factor = 1.0
        if dimensionless:
            dl_factor = self._column.velocity*self._column.length

        binding = self._column.binding_model
        salt_scale = self.m.scale_q[self._model.salt]

        def rule_adsorption(m, s, t, x):
            if t == 0:
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self._model.list_components():
                c_var[n] = m.C[n, t, x] * m.scale_c[n]
                q_var[n] = m.Q[n, t, x] * m.scale_q[n]

            if self._model.is_salt(s):
                lhs = self.m.Q[s, t, x] * m.scale_q[s]
                rhs = binding.f_ads(s, c_var, q_var)
            else:

                if binding.is_kinetic:
                    lhs = self.m.dQdt[s, t, x]*dl_factor*m.scale_q[s]
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
                          dimensionless=True,
                          with_dispersion=False,
                          dankwerts=False,
                          q_scale=None,
                          c_scale=None):

        self._build_sets_ideal_model(tspan,
                                     lspan=lspan,
                                     dimensionless=dimensionless)

        # change scales in pyomo model
        if isinstance(q_scale, dict):
            for k, v in q_scale.items():
                self.m.scale_q[k] = v
        if isinstance(c_scale, dict):
            for k, v in c_scale.items():
                self.m.scale_c[k] = v

        self._build_variables_ideal_model(with_dispersion=with_dispersion)

        self._build_mass_conservation_ideal_model(dimensionless=dimensionless,
                                                  with_dispersion=with_dispersion)

        self._build_bc_ideal_model(dimensionless=dimensionless,
                                   dankwerts=dankwerts)

        self._build_ic_ideal_model()

        self._build_adsorption_equations(dimensionless=dimensionless)

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
        discretizer.apply_to(self.m, nfe=40, ncp=3, wrt=self.m.t)

    def run_sim(self,
                solver='ipopt',
                solver_opts=None):

        opt = pe.SolverFactory(solver)
        if isinstance(solver_opts, dict):
            for k, v in solver_opts.items():
                opt.options[k] = v

        results = opt.solve(self.m, tee=True)