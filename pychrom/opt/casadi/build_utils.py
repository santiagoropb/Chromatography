from pychrom.modeling.casadi.smoothing import smooth_named_functions
from pychrom.modeling.results_object import ResultsDataSet
import casadi as ca
import numpy as np
import xarray as xr
import logging


def backward_dcdx(m, s, x):
    idx = m.x.index(x)
    dx = m.x[idx] - m.x[idx - 1]
    i = m.s_to_id[s]
    j = idx
    jb = idx - 1
    return (m.C[i, j] - m.C[i, jb]) / dx


def backward_dcdx2(m, s, x):
    idx = m.x.index(x)
    dx = m.x[idx] - m.x[idx - 1]
    i = m.s_to_id[s]
    j = idx
    jb = idx - 1
    jbb = idx - 2
    return (m.C[i, j] - 2 * m.C[i, jb] + m.C[i, jbb])/dx**2


class CasadiModel(object):

    def __init__(self):
        self.x = None
        self.s = None
        self.r = None
        self.grid_t = None
        self.scaled_st = False

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
        self.nominal_c = None
        self.nominal_q = None

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
        self._column = column
        self._inlet = getattr(self._column._model(), self._column.left_connection)
        self._outlet = getattr(self._column._model(), self._column.right_connection)

        self.m = CasadiModel()

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
        Creates base sets and params for modeling a chromatography column with casadi
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """

        nominal_c = kwargs.pop('nominal_c', None)
        nominal_q = kwargs.pop('nominal_q', None)
        scale_st = kwargs.pop('scale_st', False)

        ################################## define sets and maps ##################################
        # defines space discrete points
        self.m.x = []
        self.m.scaled_st = scale_st
        if scale_st:
            for x in lspan:
                if x > self._column.length:
                    raise RuntimeError("entry grater than column length")
                self.m.x.append(x/self._column.length)
        else:
            for x in lspan:
                if x > self._column.length:
                    raise RuntimeError("entry grater than column length")
                self.m.x.append(x)

        self.m.r = []
        if rspan is not None:
            if scale_st:
                for r in rspan:
                    if r > self._column.particle_radius:
                        raise RuntimeError("entry grater than particle radius")
                    self.m.r.append(r / self._column.particle_radius)
            else:
                for r in rspan:
                    if r > self._column.particle_radius:
                        raise RuntimeError("entry grater than particle radius")
                    self.m.r.append(r)
        else:
            if scale_st:
                self.m.r = [0.0,  1.0]
            else:
                self.m.r = [0.0, self._column.particle_radius]

        if scale_st:
            t_scale = self._column.length/self._column.velocity
        else:
            t_scale = 1.0

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

        ################################## define containers ##################################
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

        # defines container for algebraic equations
        self.m.alg_eq = []


        # defines container with initial conditions
        self.m.state_ic = []
        self.m.algebraic_ic = []

        ################################## define variables ##################################

        bm = self._column.binding_model
        F = (1.0 - self._column.column_porosity) / self._column.column_porosity
        s_no_salt = [s for s in self.m.s if not bm.is_salt(s)]

        if len(s_no_salt) == len(self.m.s):
            salt_name = "no_salt_name"
        else:
            sno = set(s_no_salt)
            sw = set(self.m.s)
            ns = sw.difference(sno)
            salt_name = ns.pop()

        self.m.C = dict()
        for i in range(self.m.nx):
            for j in range(self.m.ns):
                s = self.m.id_to_s[j]
                name = "C_{}_{}".format(s, i)
                self.m.C[j, i] = ca.SX.sym(name)

        self.m.Q = dict()
        for i in range(self.m.nx):
            for j in range(self.m.ns):
                s = self.m.id_to_s[j]
                name = "Q_{}_{}".format(s, i)
                self.m.Q[j, i] = ca.SX.sym(name)

        self.m.t = ca.SX.sym("t")

        # derivatives q
        self.m.dQ = dict()
        for i in range(self.m.nx):
            for j in range(self.m.ns):
                s = self.m.id_to_s[j]
                name = "dQ_{}_{}".format(s, i)
                self.m.dQ[j, i] = ca.SX.sym(name)


        ################################## Defines scaling parameters ##########################
        init_c = dict()
        self.m.nominal_c = dict()
        if isinstance(nominal_c, dict):
            for s in self.m.s:
                if s not in nominal_c.keys():
                    self.m.nominal_c[s] = 1.0
                else:
                    self.m.nominal_c[s] = nominal_c[s]
                init_c[s] = self._column.init_c(s)/self.m.nominal_c[s]
        else:
            for s in self.m.s:
                init_c[s] = self._column.init_c(s)
                self.m.nominal_c[s] = 1.0

        init_q = dict()
        self.m.nominal_q = dict()
        if isinstance(nominal_q, dict):
            for s in self.m.s:
                if s not in nominal_q.keys():
                    self.m.nominal_q[s] = 1.0
                else:
                    self.m.nominal_q[s] = nominal_q[s]
                init_q[s] = self._column.init_q(s) / self.m.nominal_q[s]
        else:
            for s in self.m.s:
                init_q[s] = self._column.init_q(s)
                self.m.nominal_q[s] = 1.0


        ################################## build mobile phase equations ##################################
        if scale_st:
            v = 1.0
        else:
            v = self._column.velocity

        for i in range(self.m.nx):
            if i > 0:
                for j in range(self.m.ns):
                    s = self.m.id_to_s[j]
                    x = self.m.id_to_x[i]
                    expr = -v * backward_dcdx(self.m, s, x)-F*self.m.dQ[j, i]
                    self.m.ode.append(expr/self.m.nominal_c[s])
                    self.m.states.append(self.m.C[j, i])
                    ic = init_c[s]
                    self.m.state_ic.append(ic)

        ################################## build boundary conditions c ##################################

        inlet_functions = self.create_inlet_functions()
        for j in range(self.m.ns):
            s = self.m.id_to_s[j]
            expr = inlet_functions[s](s, self.m.t*t_scale)
            #expr = 1.0
            self.m.alg_eq.append(self.m.C[j, 0] - expr)
            self.m.algebraics.append(self.m.C[j, 0])
            ic = init_c[s]
            self.m.algebraic_ic.append(ic)

        ################################## build adsorption equaitons ##################################

        if salt_name != "no_salt_name":
            to_loop = s_no_salt
        else:
            to_loop = self.m.s

        # add all but not salt
        self.m.dQ_exprs = dict()
        for i in range(self.m.nx):
            # store symbolics in dictionaries
            qvars = dict()
            cvars = dict()
            for j in range(self.m.ns):
                s = self.m.id_to_s[j]
                qvars[s] = self.m.Q[j, i]
                cvars[s] = self.m.C[j, i]

            for s in to_loop:
                j = self.m.s_to_id[s]
                expr = bm.f_ads(s, cvars, qvars, smoothing=True)/self.m.nominal_q[s]
                self.m.ode.append(expr)
                self.m.dQ_exprs[j, i] = expr
                self.m.states.append(self.m.Q[j, i])
                ic = init_q[s]
                self.m.state_ic.append(ic)

        # takes care of salt
        if salt_name != "no_salt_name":
            # add salt to algebraics
            for i in range(self.m.nx):
                # store symbolics in dictionaries
                qvars = dict()
                cvars = dict()
                for j in range(self.m.ns):
                    s = self.m.id_to_s[j]
                    qvars[s] = self.m.Q[j, i]
                    cvars[s] = self.m.C[j, i]

                j = self.m.s_to_id[salt_name]
                expr = bm.f_ads(salt_name, cvars, qvars, smoothing=True)
                self.m.alg_eq.append(expr-self.m.Q[j, i])
                self.m.algebraics.append(self.m.Q[j, i])
                ic = init_q[salt_name]
                self.m.algebraic_ic.append(ic)

                self.m.dQ_exprs[j, i] = 0.0
                for s in to_loop:
                    jj = self.m.s_to_id[s]
                    self.m.dQ_exprs[j, i] -= self.m.dQ_exprs[jj, i]*bm.nu(s)
        # substitute expressions
        old_expr = []
        new_expr = []
        for i in range(self.m.nx):
            for j in range(self.m.ns):
                old_expr.append(self.m.dQ[j, i])
                new_expr.append(self.m.dQ_exprs[j, i])

        for i, expr in enumerate(self.m.ode):
            self.m.ode[i] = ca.substitute(expr, ca.vertcat(*old_expr), ca.vertcat(*new_expr))

        # substitute scaling parameters
        old_expr = []
        new_expr = []
        for i in range(self.m.nx):
            for j in range(self.m.ns):
                s = self.m.id_to_s[j]
                if self.m.nominal_c[s] != 1.0:
                    old_expr.append(self.m.C[j, i])
                    new_expr.append(self.m.C[j, i]*self.m.nominal_c[s])
                if self.m.nominal_q[s] != 1.0:
                    old_expr.append(self.m.Q[j, i])
                    new_expr.append(self.m.Q[j, i] * self.m.nominal_q[s])
        if new_expr:
            for i, expr in enumerate(self.m.ode):
                self.m.ode[i] = ca.substitute(expr, ca.vertcat(*old_expr), ca.vertcat(*new_expr))
            for i, expr in enumerate(self.m.alg_eq):
                self.m.alg_eq[i] = ca.substitute(expr, ca.vertcat(*old_expr), ca.vertcat(*new_expr))
        """
        print("STATES")
        for i, v in enumerate(self.m.states):
            print(v, self.m.ode[i], self.m.state_ic[i])

        print("ALGEBRAICS")
        for i, v in enumerate(self.m.algebraics):
            print(v, self.m.alg_eq[i], self.m.algebraic_ic[i])
        """
        return self.m

    def store_values_in_data_set(self, solution, store_ders=True):
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
        if self.m.scaled_st:
            result_set.times = np.array(self.m.grid_t)*self._column.length/self._column.velocity
            result_set.col_locs = np.array(self.m.x)*self._column.length
        else:
            result_set.times = np.array(self.m.grid_t)
            result_set.col_locs = np.array(self.m.x)


        conc = np.zeros((ns, nt, nx))

        for j, t in enumerate(self.m.grid_t):
            for k, x in enumerate(self.m.x):
                for w, s in enumerate(self.m.s):
                    if k > 0:
                        conc[w, j, k] = solution['xf'][(k * ns + w) - ns, j]*self.m.nominal_c[s]

        for j, t in enumerate(self.m.grid_t):
            for w, s in enumerate(self.m.s):
                conc[w, j, 0] = solution['zf'][w, j]*self.m.nominal_c[s]

        result_set.C = xr.DataArray(conc,
                                    coords=[self.m.s, result_set.times, result_set.col_locs],
                                    dims=['component', 'time', 'col_loc'])

        bm = self._column.binding_model
        s_no_salt = [s for s in self.m.s if not bm.is_salt(s)]
        if len(s_no_salt) == len(self.m.s):
            salt_name = "no_salt_name"
        else:
            sno = set(s_no_salt)
            sw = set(self.m.s)
            nos = sw.difference(sno)
            salt_name = nos.pop()

        # get the Qs
        concQ = np.zeros((ns, nt, nx, 1))
        if salt_name == "no_salt_name":
            for j, t in enumerate(self.m.grid_t):
                for k, x in enumerate(self.m.x):
                    for w, s in enumerate(self.m.s):
                        val = solution['xf'][(k * ns + w) + (nx-1)*ns, j] * self.m.nominal_c[s]
                        if val < 1e-10:
                            concQ[w, j, k, 0] = 0.0
                        else:
                            concQ[w, j, k, 0] = val
        else:
            for j, t in enumerate(self.m.grid_t):
                for k, x in enumerate(self.m.x):
                    for w, s in enumerate(s_no_salt):
                        sid = self.m.s_to_id[s]
                        val = solution['xf'][(k * (ns-1) + w) + (nx - 1) * ns, j] * self.m.nominal_c[s]
                        if val < 1e-10:
                            concQ[sid, j, k, 0] = 0.0
                        else:
                            concQ[sid, j, k, 0] = val

            sid = self.m.s_to_id[salt_name]
            for j, t in enumerate(self.m.grid_t):
                for k, x in enumerate(self.m.x):
                    val = solution['zf'][ns+k, j] * self.m.nominal_c[salt_name]
                    if val < 1e-10:
                        concQ[sid, j, k, 0] = 0.0
                    else:
                        concQ[sid, j, k, 0] = val
        # store q concentrations
        r = self._column.particle_radius
        result_set.par_locs = np.array([r])
        result_set.Q = xr.DataArray(concQ,
                                    coords=[result_set.components,
                                            result_set.times,
                                            result_set.col_locs,
                                            result_set.par_locs],
                                    dims=['component',
                                          'time',
                                          'col_loc',
                                          'par_loc'])
        if store_ders:
            # add derivatives
            dcdt = np.zeros((ns, nt, nx))
            for j, t in enumerate(self.m.grid_t):
                if j < nt - 1:
                    t0 = t
                    t1 = self.m.grid_t[j + 1]
                    if abs(t1 - t0) > 1e-6:
                        for k, x in enumerate(self.m.x):
                            for w, s in enumerate(self.m.s):
                                c0 = conc[w, j, k]
                                c1 = conc[w, j+1, k]
                                dcdt[w, j, k] = (c1-c0)/(t1-t0)

            result_set.dCdt = xr.DataArray(dcdt,
                                        coords=[self.m.s, result_set.times, result_set.col_locs],
                                        dims=['component', 'time', 'col_loc'])

            dcdx = np.zeros((ns, nt, nx))
            for k, x in enumerate(self.m.x):
                if k < nx - 1:
                    x0 = x
                    x1 = self.m.x[k + 1]
                    if abs(x1 - x0) > 1e-6:
                        for j, t in enumerate(self.m.grid_t):
                            for w, s in enumerate(self.m.s):
                                c0 = conc[w, j, k]
                                c1 = conc[w, j, k+1]
                                dcdt[w, j, k] = (c1 - c0) / (x1 - x0)

            result_set.dCdx = xr.DataArray(dcdx,
                                           coords=[self.m.s, result_set.times, result_set.col_locs],
                                           dims=['component', 'time', 'col_loc'])

            dQdt = np.zeros((ns, nt, nx, 1))
            for j, t in enumerate(self.m.grid_t):
                if j < nt - 1:
                    t0 = t
                    t1 = self.m.grid_t[j + 1]
                    if abs(t1 - t0) > 1e-6:
                        for k, x in enumerate(self.m.x):
                            for w, s in enumerate(self.m.s):
                                q0 = concQ[w, j, k, 0]
                                q1 = concQ[w, j+1, k, 0]
                                dQdt[w, j, k, 0] = (q1-q0)/(t1-t0)
            result_set.dQdt = xr.DataArray(dQdt,
                                        coords=[result_set.components,
                                                result_set.times,
                                                result_set.col_locs,
                                                result_set.par_locs],
                                        dims=['component',
                                              'time',
                                              'col_loc',
                                              'par_loc'])

        return result_set

    def solve(self, tspan):

        if self.m.scaled_st:
            t_scale = self._column.velocity/self._column.length
        else:
            t_scale = 1.0

        self.m.grid_t = [t*t_scale for t in tspan]
        # defines dae
        dae = {'x': ca.vertcat(*self.m.states),
               'z': ca.vertcat(*self.m.algebraics),
               'p': [],
               't': self.m.t,
               'ode': ca.vertcat(*self.m.ode),
               'alg': ca.vertcat(*self.m.alg_eq)}

        opts = {'grid': self.m.grid_t, 'output_t0': True}

        integrator = ca.integrator('I', 'idas', dae, opts)

        sol = integrator(x0=self.m.state_ic, z0=self.m.algebraic_ic)

        return self.store_values_in_data_set(sol)