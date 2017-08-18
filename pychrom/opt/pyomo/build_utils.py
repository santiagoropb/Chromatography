from pychrom.modeling.pyomo.smoothing import PieceWiseNamedFunction
from pychrom.modeling.pyomo.smoothing import smooth_named_functions
from pychrom.modeling.results_object import ResultsDataSet
import pyomo.environ as pe
import pyomo.dae as dae
import xarray as xr
import numpy as np


# definition of variables
def define_c_vars(model, with_second_der=False, scale_vars=False):

    if scale_vars:

        model.phi = pe.Var(model.s, model.t, model.x)
        model.dphidx = dae.DerivativeVar(model.phi, wrt=model.x)
        model.dphidt = dae.DerivativeVar(model.phi, wrt=model.t)

        model.sc = pe.Param(model.s, initialize=1.0, mutable=True)

        def rule_rescale_c(m, i, j, k):
            return m.phi[i, j, k] * m.sc[i]

        def rule_rescale_dcdx(m, i, j, k):
            return m.dphidx[i, j, k] * m.sc[i]

        def rule_rescale_dcdt(m, i, j, k):
            return m.dphidt[i, j, k] * m.sc[i]

        model.C = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_c)
        model.dCdx = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dcdx)
        model.dCdt = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dcdt)

    else:
        model.C = pe.Var(model.s, model.t, model.x)
        model.dCdx = dae.DerivativeVar(model.C, wrt=model.x)
        model.dCdt = dae.DerivativeVar(model.C, wrt=model.t)
        if with_second_der:
            model.dCdx2 = dae.DerivativeVar(model.C, wrt=(model.x, model.x))


def define_q_vars(model, scale_vars=False, index_radius=False, with_first_der=False, with_second_der=False):

    if not index_radius:
        if scale_vars:
            model.gamma = pe.Var(model.s, model.t, model.x)
            model.dgdt = dae.DerivativeVar(model.gamma, wrt=model.t)

            model.sq = pe.Param(model.s, initialize=1.0, mutable=True)

            def rule_rescale_q(m, i, j, k):
                return m.gamma[i, j, k] * m.sq[i]

            def rule_rescale_dqdt(m, i, j, k):
                return m.dgdt[i, j, k] * m.sq[i]

            model.Q = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_q)
            model.dQdt = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dqdt)
        else:
            model.Q = pe.Var(model.s, model.t, model.x)
            model.dQdt = dae.DerivativeVar(model.Q, wrt=model.t)

    else:
        if scale_vars:
            model.gamma = pe.Var(model.s, model.t, model.x, model.r)
            model.dgdt = dae.DerivativeVar(model.gamma, wrt=model.t)

            model.sq = pe.Param(model.s, initialize=1.0, mutable=True)

            def rule_rescale_q(m, i, j, k, w):
                return m.gamma[i, j, k, w] * m.sq[i]

            def rule_rescale_dqdt(m, i, j, k, w):
                return m.dgdt[i, j, k, w] * m.sq[i]

            # stationary phase concentration variable
            model.Q = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_q)
            model.dQdt = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dqdt)

            if with_first_der:
                model.dgdr = dae.DerivativeVar(model.gamma, wrt=model.r)

                def rule_rescale_dqdr(m, i, j, k, w):
                    return m.dgdr[i, j, k, w] * m.sq[i]

                model.dQdr = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dqdr)

            if with_second_der:
                model.dgdr2 = dae.DerivativeVar(model.gamma, wrt=(model.r, model.r))

                def rule_rescale_dqdr2(m, i, j, k, w):
                    return m.dgdr2[i, j, k, w] * m.sq[i]

                model.dQdr2 = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dqdr2)

        else:
            model.Q = pe.Var(model.s, model.t, model.x, model.r)
            model.dQdt = dae.DerivativeVar(model.Q, wrt=model.t)

            if with_first_der:
                model.dQdr = dae.DerivativeVar(model.Q, wrt=model.r)
            if with_second_der:
                model.dQdr2 = dae.DerivativeVar(model.Q, wrt=(model.r, model.r))


# discretization functions
def backward_dcdx(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx] - x_list[idx - 1]
    return (m.C[s, t, x_list[idx]] - m.C[s, t, x_list[idx - 1]]) / dx


def backward_dcdx2(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx] - x_list[idx-1]
    if idx-2 > 0:
        return (m.C[s, t, x_list[idx]] - 2 * m.C[s, t, x_list[idx-1]] + m.C[s, t, x_list[idx-2]])/dx**2
    else:  # silently ignores first point
        return 0.0


def forward_dcdx(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx+1] - x_list[idx]
    return (m.C[s, t, x_list[idx+1]] - m.C[s, t, x_list[idx]]) / dx


def forward_dcdx2(m, s, t, x, x_list):
    idx = x_list.index(x)
    dx = x_list[idx+1] - x_list[idx]
    if idx + 2 < len(x_list)-1:
        return (m.C[s, t, x_list[idx+2]] - 2 * m.C[s, t, x_list[idx+1]] + m.C[s, t, x_list[idx]])/dx**2
    else:  # silently ignores last point
        return 0.0


class PyomoColumn(object):

    def __init__(self, column):
        self._column = column
        self._inlet = getattr(self._column._model(), self._column.left_connection)
        self._outlet = getattr(self._column._model(), self._column.right_connection)

        self.m = pe.ConcreteModel()
        self.m.scaled_st = False

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
                component_function = smooth_named_functions(list_functions, break_times, name, tee=False)
            else:
                component_function = PieceWiseNamedFunction(list_functions, break_times, name)
            section_functions[name] = component_function

        return section_functions

    def build_model(self, tspan, lspan=None, rspan=None, **kwargs):

        scale_st = kwargs.pop('scale_st', False)
        self.m.scaled_st = scale_st

        if lspan is None:
            lspan = [0.0, self._column.length]
        # defines space discrete points
        inner_lspan = []
        if scale_st:
            for x in lspan:
                if x > self._column.length:
                    raise RuntimeError("entry grater than column length")
                inner_lspan.append(x / self._column.length)
        else:
            for x in lspan:
                if x > self._column.length:
                    raise RuntimeError("entry grater than column length")
                inner_lspan.append(x)

        if scale_st:
            v = 1.0
            t_scale = self._column.velocity / self._column.length
        else:
            v = self._column.velocity
            t_scale = 1.0

        inner_tspan = []
        for t in tspan:
            inner_tspan.append(t*t_scale)


        # define sets
        self.m.s = self._column.list_components()
        self.m.t = dae.ContinuousSet(initialize=inner_tspan)
        self.m.x = dae.ContinuousSet(initialize=inner_lspan)

        # define varibles
        define_c_vars(self.m)
        define_q_vars(self.m)

        f = (1.0 - self._column.column_porosity) / self._column.column_porosity

        # write mobile phase balances
        def rule_mobile_phase(m, i, j, k):
            if j == m.x.first() or j == m.t.first():
                return pe.Constraint.Skip
            lhs = m.dCdt[i, j, k]
            rhs = -v * m.dCdx[i, j, k] - f * m.dQdt[i, j, k]
            return lhs == rhs

        self.m.mobile_balance = pe.Constraint(self.m.s, self.m.t, self.m.x,
                                              rule=rule_mobile_phase)

        # define inlet boundary condition
        inlet_functions = self.create_inlet_functions()

        def rule_inlet_bc(m, i, j):
            if j == m.t.first():
                return pe.Constraint.Skip
            lhs = m.C[i, j, m.x.first()]
            rhs = inlet_functions[i](i, j/t_scale)
            return lhs == rhs

        self.m.inlet = pe.Constraint(self.m.s, self.m.t,
                                     rule=rule_inlet_bc)

        # define initial condition c
        def rule_init_c(m, i, k):
            t0 = m.t.first()
            return m.C[i, t0, k] == self._column.init_c(i)

        self.m.ic_c = pe.Constraint(self.m.s, self.m.x,
                                    rule=rule_init_c)

        # define adsorption equations
        bm = self._column.binding_model

        def rule_adsorption(m, i, j, k):
            if j == m.t.first() and not bm.is_salt(i):
                return pe.Constraint.Skip

            c_var = dict()
            q_var = dict()
            for n in self.m.s:
                c_var[n] = m.C[n, j, k]
                q_var[n] = m.Q[n, j, k]

            if bm.is_salt(i):
                lhs = m.Q[i, j, k]
            else:
                lhs = m.dQdt[i, j, k]*t_scale
            rhs = bm.f_ads(i, c_var, q_var)
            return lhs == rhs

        self.m.binding_balance = pe.Constraint(self.m.s, self.m.t, self.m.x, rule=rule_adsorption)

        # define initial conditions q
        def rule_init_q(m, i, k):
            if bm.is_salt(i):
                return pe.Constraint.Skip
            t0 = self.m.t.first()
            return m.Q[i, t0, k] == self._column.init_q(i)

        self.m.ic_q = pe.Constraint(self.m.s, self.m.x,
                                    rule=rule_init_q)

    def initialize_variables(self, trajectories=None):

        inlet_functions = self.create_inlet_functions()

        if self.m.scaled_st:
            t_scale = self._column.length / self._column.velocity
            x_scale = self._column.length
        else:
            t_scale = 1.0
            x_scale = 1.0

        # defaults initializes all with initial conditions
        if trajectories is None:
            for i in self.m.s:
                for j in self.m.t:
                    for k in self.m.x:
                        if k == self.m.x.first():
                            self.m.C[i, j, k].value = inlet_functions[i](i, j*t_scale)
                        else:
                            self.m.C[i, j, k].value = self._column.init_c(i)
                        #self.m.Q[i, j, k].value = self._column.init_q(i)
        else:
            if hasattr(trajectories, 'C'):
                for i in self.m.s:
                    Cn = trajectories.C.sel(component=i)
                    for j in self.m.t:
                        for k in self.m.x:
                            val = Cn.sel(time=j*t_scale, col_loc=k*x_scale, method='nearest')
                            self.m.C[i, j, k].value = float(val)

            if hasattr(trajectories, 'Q'):
                r = self._column.particle_radius
                for i in self.m.s:
                    Qn = trajectories.Q.sel(component=i)
                    for j in self.m.t:
                        for k in self.m.x:
                            val = Qn.sel(time=j * t_scale, col_loc=k * x_scale, par_loc=r, method='nearest')
                            self.m.Q[i, j, k].value = float(val)

            if hasattr(trajectories, 'dCdt'):
                for i in self.m.s:
                    Cn = trajectories.dCdt.sel(component=i)
                    for j in self.m.t:
                        for k in self.m.x:
                            val = Cn.sel(time=j * t_scale, col_loc=k * x_scale, method='nearest')
                            self.m.dCdt[i, j, k].value = float(val)

            if hasattr(trajectories, 'dCdx'):
                for i in self.m.s:
                    Cn = trajectories.dCdx.sel(component=i)
                    for j in self.m.t:
                        for k in self.m.x:
                            val = Cn.sel(time=j * t_scale, col_loc=k * x_scale, method='nearest')
                            self.m.dCdx[i, j, k].value = float(val)

            if hasattr(trajectories, 'dQdt'):
                r = self._column.particle_radius
                for i in self.m.s:
                    Qn = trajectories.dQdt.sel(component=i)
                    for j in self.m.t:
                        for k in self.m.x:
                            val = Qn.sel(time=j * t_scale, col_loc=k * x_scale, par_loc=r, method='nearest')
                            self.m.dQdt[i, j, k].value = float(val)

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

        if self.m.scaled_st:
            result_set.times = np.array(sorted_t) * self._column.length / self._column.velocity
            result_set.col_locs = np.array(sorted_x) * self._column.length
        else:
            result_set.times = np.array(sorted_t)
            result_set.col_locs = np.array(sorted_x)

        # store c concentrations
        result_set.C = xr.DataArray(conc,
                                    coords=[result_set.components,
                                            result_set.times,
                                            result_set.col_locs],
                                    dims=['component',
                                          'time',
                                          'col_loc'])

        # add Qs
        conc = np.zeros((ns, nt, nx, 1))
        for i, s in enumerate(result_set.components):
            for j, t in enumerate(sorted_t):
                for k, x in enumerate(sorted_x):
                    val = pe.value(self.m.Q[s, t, x])
                    if val<1e-10:
                        conc[i, j, k, 0] = 0.0
                    else:
                        conc[i, j, k, 0] = val
        # store q concentrations
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

    def discretize_space(self, nfe):

        # Discretize using Finite Difference
        discretizer = pe.TransformationFactory('dae.finite_difference')
        discretizer.apply_to(self.m, nfe=nfe, wrt=self.m.x, scheme='BACKWARD')

    def discretize_time(self, nfe, ncp=1):

        # Discretize using Finite elements and collocation
        discretizer = pe.TransformationFactory('dae.collocation')
        discretizer.apply_to(self.m, nfe=nfe, ncp=ncp, wrt=self.m.t)

    def solve(self, solver='ipopt', solver_opts=None, tee=True):

        opt = pe.SolverFactory(solver)
        if isinstance(solver_opts, dict):
            for k, v in solver_opts.items():
                opt.options[k] = v

        opt.solve(self.m, tee=tee)

        results_set = self.store_values_in_data_set()
        return results_set