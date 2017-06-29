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

        self._column = getattr(self._model, columns[0])
        self._inlet = getattr(self._model, self._column.left_connection)
        self._outlet = getattr(self._model, self._column.right_connection)

        self.m = pe.ConcreteModel()

    def _create_sets_ideal_model(self,
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

    def _create_variables_ideal_model(self, with_dispersion=False):

        # auxiliary aliases
        x = self.m.z
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

    def _add_mass_conservation_ideal_model(self, dimensionles=True, with_dispersion=False):

        diff_x = self._column.dispersion
        inv_pe = diff_x
        if dimensionles:
            inv_pe /= (self._column.velocity*self._column.length)
        F = (1.0 - self._column.column_porosity) / self._column.column_porosity

        # mobile phase mass balance
        def rule_mass_balance_with_dispersion(m, s, t, x):
            if x in m.x.bounds() or t == m.t.bounds()[0]:
                return pe.Constraint.Skip
            return m.dCdt[s, t, x] + m.dCdx[s, t, x] + F*self.m.dQdt[s, t, x] == inv_pe*self.m.dCdx2[s, t, x]

        def rule_mass_balance_no_dispersion(m, s, t, x):
            if x in m.x.bounds() or t == m.t.bounds()[0]:
                return pe.Constraint.Skip
            return m.dCdt[s, t, x] + m.dCdx[s, t, x] + F * self.m.dQdt[s, t, x] == 0.0

        if with_dispersion:
            self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                       self.m.t,
                                                       self.m.x,
                                                       rule=rule_mass_balance_with_dispersion())
        else:
            self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                       self.m.t,
                                                       self.m.x,
                                                       rule=rule_mass_balance_no_dispersion())