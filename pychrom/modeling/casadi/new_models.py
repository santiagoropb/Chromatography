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


        ################################## define sets and maps ##################################
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
        scale_q = dict()
        scale_c = dict()

        for s in self.m.s:
            scale_c[s] = 1.0
            scale_q[s] = 1.0

        bm = self._column.binding_model
        s_no_salt = [s for s in self.m.s if not bm.is_salt(s)]

        if len(s_no_salt) == len(self.m.s):
            salt_name = "no_salt_name"
        else:
            sno = set(s_no_salt)
            sw = set(self.m.s)
            ns = sw.difference(sno)
            salt_name = ns.pop()
            scale_c[salt_name] = self._column.init_c(salt_name)
            scale_q[salt_name] = bm.lamda

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


        ################################## build mobile phase equations ##################################
        v = self._column.velocity
        for i in range(self.m.nx):
            if i > 0:
                for j in range(self.m.ns):
                    s = self.m.id_to_s[j]
                    x = self.m.id_to_x[i]
                    expr = -v * backward_dcdx(self.m, s, x)
                    self.m.ode.append(expr)
                    self.m.states.append(self.m.C[j, i])
                    ic = self._column.init_c(s)
                    self.m.state_ic.append(ic)

        ################################## build boundary conditions c ##################################

        inlet_functions = self.create_inlet_functions()
        for j in range(self.m.ns):
            s = self.m.id_to_s[j]
            expr = inlet_functions[s](s, self.m.t)
            #expr = 1.0
            self.m.alg_eq.append(self.m.C[j, 0] - expr)
            self.m.algebraics.append(self.m.C[j, 0])
            ic = self._column.init_c(s)
            self.m.algebraic_ic.append(ic)

        ################################## build adsorption equaitons ##################################


        if bm.is_kinetic:
            if salt_name != "no_salt_name":
                to_loop = s_no_salt
            else:
                to_loop = self.m.s

            # add all but not salt
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
                    expr = bm.f_ads(s, cvars, qvars, smoothing=False)
                    self.m.ode.append(expr)
                    self.m.states.append(self.m.Q[j, i])
                    ic = self._column.init_q(s)
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
                    expr = bm.f_ads(salt_name, cvars, qvars, smoothing=False)
                    self.m.alg_eq.append(expr-self.m.Q[j, i])
                    self.m.algebraics.append(self.m.Q[j, i])
                    ic = self._column.init_q(salt_name)
                    self.m.algebraic_ic.append(ic)
        else:
            # add algebraics
            for i in range(self.m.nx):
                # store symbolics in dictionaries
                qvars = dict()
                cvars = dict()
                for j in range(self.m.ns):
                    s = self.m.id_to_s[j]
                    qvars[s] = self.m.Q[j, i]
                    cvars[s] = self.m.C[j, i]

                for j in range(self.m.ns):
                    s = self.m.id_to_s[j]
                    expr = bm.f_ads(s, cvars, qvars, smoothing=False)
                    self.m.alg_eq.append(expr-self.m.Q[j, i])
                    self.m.algebraics.append(self.m.Q[j, i])
                    ic = self._column.init_q(s)
                    self.m.algebraic_ic.append(ic)

        print("STATES")
        for i, v in enumerate(self.m.states):
            print(v, self.m.ode[i], self.m.state_ic[i])

        print("ALGEBRAICS")
        for i, v in enumerate(self.m.algebraics):
            print(v, self.m.alg_eq[i], self.m.algebraic_ic[i])

        return self.m

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
                        conc[w, j, k] = solution['xf'][(k * ns + w) - ns, j]

        for j, t in enumerate(self.m.grid_t):
            for w, s in enumerate(self.m.s):
                conc[w, j, 0] = solution['zf'][w, j]

        result_set.C = xr.DataArray(conc,
                                    coords=[self.m.s, self.m.grid_t, self.m.x],
                                    dims=['component', 'time', 'col_loc'])
        return result_set


