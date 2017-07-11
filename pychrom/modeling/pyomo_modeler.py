from pychrom.modeling.pyomo.var import define_C_vars, define_Q_vars
from pychrom.modeling.results_object import ResultsDataSet
from pychrom.modeling.pyomo.ideal_model import IdealColumn
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

        self.dimensionless = True
        self.wq = False

    def build_model(self,
                    tspan,
                    model_type,
                    lspan=None,
                    rspan=None,
                    q_scale=None,
                    c_scale=None,
                    free_vars_scale=None):

        if model_type == 'IdealModel':
            pcolumn = IdealColumn(self._column,
                                  dimensionless=self.dimensionless,
                                  with_q=self.wq)

            pcolumn.build_pyomo_model(tspan,
                                      lspan=lspan,
                                      rspan=None)

            self.m = pcolumn.pyomo_model()

            # change scales in pyomo model
            if isinstance(c_scale, dict):
                for k, v in c_scale.items():
                    self.m.sc[k] = v

            if isinstance(q_scale, dict):
                for k, v in q_scale.items():
                    self.m.sq[k] = v
            #self.m.pprint()

        else:
            raise NotImplementedError("Model not supported yet")


    def discretize_space(self):

        # Discretize using Finite Difference and Collocation
        discretizer = pe.TransformationFactory('dae.finite_difference')

        discretizer.apply_to(self.m,
                             nfe=50,
                             wrt=self.m.x,
                             scheme='BACKWARD')

    def discretize_time(self):

        discretizer = pe.TransformationFactory('dae.collocation')
        discretizer.apply_to(self.m, nfe=30, ncp=2, wrt=self.m.t)

    def initialize_variables(self, init_trajectories=None):

        L = self._column.length
        u = self._column.velocity
        t_factor = u / L

        if init_trajectories is None:
            for s in self.m.s:
                for t in self.m.t:
                    for x in self.m.x:
                        self.m.phi[s, t, x].value = self._column.init_c(s)/pe.value(self.m.sc[s])
                        if self.wq:
                            self.m.gamma[s, t, x].value = self._column.init_q(s)/pe.value(self.m.sq[s])
        else:
            for s in self.m.s:
                Cn = init_trajectories.C.sel(component=s)
                Qn = init_trajectories.Q.sel(component=s)
                for t in self.m.t:
                    tt = t*t_factor
                    for x in self.m.x:
                        xx = x/L
                        val = Cn.sel(time=tt, location=xx, method='nearest')
                        self.m.phi[s, t, x].value = float(val)/pe.value(self.m.sc[s])
                        val = Qn.sel(time=tt, location=xx, method='nearest')
                        self.m.gamma[s, t, x].value = float(val)/pe.value(self.m.sq[s])

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

