from pychrom.modeling.pyomo.var import define_C_vars, define_Q_vars, define_free_sites_vars
from pychrom.modeling.pyomo.pyomo_column import PyomoColumn
from pychrom.utils.smoothing import PieceWiseNamedFunction, smooth_named_functions
import pyomo.environ as pe
import pyomo.dae as dae
import logging


logger = logging.getLogger(__name__)


class IdealColumn(PyomoColumn):

    def __init__(self, column, dimensionless=True, with_q=True):

        super().__init__(column, dimensionless=dimensionless)

        # tmp attribute
        self.wq = with_q

    def setup_base(self, tspan, lspan=None, rspan=None):
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

        if self.dimensionless:
            u = self._column.velocity
            l = self._column.length
            scale = u / l
            tao = [t * scale for t in tspan]

            scale = 1.0 / l
            z = [x * scale for x in lspan]
        else:
            tao = tspan
            z = lspan

        self.m.s = self._column.list_components()
        self.m.t = dae.ContinuousSet(initialize=tao, )
        self.m.x = dae.ContinuousSet(initialize=z)

        self.m.sc = pe.Param(self.m.s,
                                  initialize=1.0,
                                  mutable=True)

        self.m.sq = pe.Param(self.m.s,
                                  initialize=1.0,
                                  mutable=True)

        lamda = self._column.binding_model.lamda
        self.m.sf = pe.Param(initialize=lamda,
                                           mutable=True)

    def build_variables(self):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

        define_C_vars(self.m, scale_vars=True, with_second_der=False)
        if self.wq:
            define_Q_vars(self.m, scale_vars=True, index_radius=False)

            define_free_sites_vars(self.m, scale_vars=True, index_radius=False)

    def build_mobile_phase_balance(self):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: None
        """

        F = (1.0 - self._column.column_porosity) / self._column.column_porosity

        # mobile phase mass balance
        def rule_mass_balance(m, s, t, x):
            if x == m.x.bounds()[0] or t == m.t.bounds()[0]:
                return pe.Constraint.Skip
            lhs = m.dCdt[s, t, x] + m.dCdx[s, t, x]
            if self.wq:
                lhs += F * self.m.dQdt[s, t, x]
            rhs = 0.0
            return lhs == rhs

        self.m.mass_balance_mobile = pe.Constraint(self.m.s,
                                                   self.m.t,
                                                   self.m.x,
                                                   rule=rule_mass_balance)
        return True

    def build_stationary_phase_balance(self):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        return False

    def build_adsorption_equations(self):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

        if self.wq:
            dl_factor = 1.0
            if self.dimensionless:
                dl_factor = self._column.velocity * self._column.length

            binding = self._column.binding_model
            salt_name = self._column.salt
            salt_scale = self.m.sq[salt_name]

            def rule_adsorption(m, s, t, x):
                if t == 0:
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
                        lhs = self.m.dQdt[s, t, x] * dl_factor
                    else:
                        lhs = 0.0
                    rhs = binding.f_ads(s, c_var, q_var, q_ref=salt_scale)

                return lhs == rhs

            self.m.adsorption = pe.Constraint(self.m.s,
                                              self.m.t,
                                              self.m.x,
                                              rule=rule_adsorption)

            self.m.adsorption.pprint()

    def build_adsorption_equations2(self):

        if self.wq:
            dl_factor = 1.0
            if self.dimensionless:
                dl_factor = self._column.velocity * self._column.length

            binding = self._column.binding_model
            salt_name = self._column.salt

            def rule_adsorption(m, s, t, x):
                if t == 0:
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
                        lhs = self.m.dQdt[s, t, x] * dl_factor
                    else:
                        lhs = 0.0

                    rhs = binding.f_ads_given_free_sites(s, c_var, q_var, m.free_sites[t, x])
                return lhs == rhs

            species = set(self.m.s)
            species.add('free_sites')
            self.m.adsorption = pe.Constraint(species,
                                              self.m.t,
                                              self.m.x,
                                              rule=rule_adsorption)

            self.m.adsorption.pprint()
            return True
        else:
            return False

    def build_boundary_conditions(self):
        """
        Creates expressions for boundary conditions
        :return: None
        """

        lin = self.m.x.bounds()[0]
        section = None
        section_functions = dict()
        break_times = []

        if not self.dimensionless:
            for n, sec in self._inlet.sections(ordered=True):
                break_times.append(sec.start_time_sec)

        else:
            u = self._column.velocity
            l = self._column.length
            scale = u / l
            for n, sec in self._inlet.sections(ordered=True):
                t = sec.start_time_sec*scale
                break_times.append(t)

        # add end of time
        end_time = self.m.t.bounds()[1]
        break_times.append(end_time)

        smooth = False

        if smooth:
            for name in self._column.list_components():
                list_functions = list()
                for n, sec in self._inlet.sections(ordered=True):
                    list_functions.append(sec.f)
                component_function = smooth_named_functions(list_functions, break_times, name)
                section_functions[name] = component_function
        else:
            for name in self._column.list_components():
                list_functions = list()
                for n, sec in self._inlet.sections(ordered=True):
                    list_functions.append(sec.f)
                component_function = PieceWiseNamedFunction(list_functions, break_times, name)
                section_functions[name] = component_function

        def rule_inlet_bc(m, s, t):
            lhs = m.C[s, t, lin]
            rhs = section_functions[s](s, t)
            return lhs == rhs

        self.m.inlet = pe.Constraint(self.m.s,
                                     self.m.t,
                                     rule=rule_inlet_bc)
        self.m.inlet.pprint()

    def build_initial_conditions(self):
        """
        Creates expressions for boundary conditions
        :return: None
        """

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