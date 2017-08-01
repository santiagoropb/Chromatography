from pychrom.modeling.pyomo_modeler import PyomoModeler
from pychrom.modeling.cadet_modeler import CadetModeler
from pychrom.core import *
import pyomo.environ as pe
import pyomo.dae as dae
import numpy as np


def create_model():
    comps = ['A', 'B', 'C', 'D']
    GRM = GRModel(components=comps)

    # create sections
    GRM.load = Section(components=comps)
    for cname in comps:
        GRM.load.set_a0(cname, 1.0)

    GRM.load.set_a0('A', 50.0)
    GRM.load.set_a1('A', 0.0)
    GRM.load.start_time_sec = 0.0

    GRM.wash = Section(components=comps)
    GRM.wash.set_a0('A', 50.0)
    GRM.wash.start_time_sec = 10.0

    GRM.elute = Section(components=comps)
    GRM.elute.set_a0('A', 100.0)
    GRM.elute.set_a1('A', 0.2)
    GRM.elute.start_time_sec = 90.0

    # create inlet
    GRM.inlet = Inlet(components=comps)
    GRM.inlet.add_section('load')
    GRM.inlet.add_section('wash')
    GRM.inlet.add_section('elute')

    # create binding
    GRM.salt = 'A'
    GRM.binding = SMABinding(data="sma.yml")
    GRM.binding.is_kinetic = False

    # create column
    GRM.column = Column(data="column.yml")

    # create outlet
    GRM.outlet = Outlet(components=comps)

    # connect units
    GRM.connect_unit_operations('inlet', 'column')
    GRM.connect_unit_operations('column', 'outlet')
    return GRM


def compute_mass_integral(model):

    modeler = CadetModeler(model)
    modeler.discretize_column('column', ncol=50, npar=10)
    results = modeler.run_sim(tspan=np.linspace(0, 1500, 3000), retrive_c='all')
    time = list(results.C.coords['time'])
    tr = dict()
    for cname in model.list_components():
        if cname != 'salt':
            traj = results.C.sel(time=time, col_loc=model.column.length, component=cname)
            # compute total concentration
            c_integral = 0.0
            for i, t in enumerate(time):
                if i < len(time) - 1:
                    t1 = time[i]
                    t2 = time[i + 1]
                    dt = t2 - t1
                    c_integral += dt * traj.sel(time=t1)
            tr[cname] = float(c_integral)
    return tr


def add_integral_variables(model, modeler, n_moments=1, mass_integrals=None):

    m = modeler.pyomo_column.pyomo_model()
    no_salt_list = [cname for cname in model.list_components() if not model.is_salt(cname)]

    if mass_integrals is None:
        # defines states for total concentrations
        m.Cts = pe.Var(no_salt_list, m.t, initialize=1.0)
        m.dCts = dae.DerivativeVar(m.Cts, wrt=m.t)

        def rule_total_concentration(m,s,t):
            if t == m.t.first():
                return pe.Constraint.Skip
            return m.dCts[s, t] == m.C[s, t, m.x.last()]
        m.c_integral = pe.Constraint(no_salt_list, m.t, rule=rule_total_concentration)

        def init_cts_rule(m, s):
            return m.Cts[s, m.t.first()] == 0.0
        m.init_cts = pe.Constraint(no_salt_list, rule=init_cts_rule)

    if n_moments >= 1:
        # defines states for retention times integral
        m.Trs = pe.Var(no_salt_list, m.t, initialize=1.0)
        m.dTrs = dae.DerivativeVar(m.Trs, wrt=m.t)

        def rule_total_time(m,s,t):
            if t == m.t.first():
                return pe.Constraint.Skip
            return m.dTrs[s, t] == m.C[s, t, m.x.last()]*t
        m.t_integral = pe.Constraint(no_salt_list, m.t, rule=rule_total_time)

        def init_trs_rule(m, s):
            return m.Trs[s, m.t.first()] == 0.0
        m.init_trs = pe.Constraint(no_salt_list, rule=init_trs_rule)

        # retention times variable
        m.mean_t = pe.Var(no_salt_list, initialize=1.0)

        def rule_retention(m,s):
            if mass_integrals is None:
                return m.Cts[s, m.t.last()]*m.mean_t[s] == m.Trs[s, m.t.last()]
            return mass_integrals[s] * m.mean_t[s] == m.Trs[s, m.t.last()]
        m.retention = pe.Constraint(no_salt_list, rule=rule_retention)

    if n_moments >= 2:
        # defines states for standard deviation
        m.sigmas = pe.Var(no_salt_list, m.t)
        m.dsigmas = dae.DerivativeVar(m.sigmas, wrt=m.t)

        def rule_sigmas(m,s,t):
            if t == m.t.first():
                return pe.Constraint.Skip
            return m.dsigmas[s, t] == m.C[s, t, m.x.last()] * (t - m.mean_t[s]) ** 2
        m.sigma_integral = pe.Constraint(no_salt_list, m.t, rule=rule_sigmas)

        def init_sigmas_rule(m, s):
            return m.sigmas[s, m.t.first()] == 0.0
        m.init_sigmas = pe.Constraint(no_salt_list, rule=init_sigmas_rule)

        # variances variable
        m.variance_t = pe.Var(no_salt_list, initialize=1.0)

        def rule_variance(m,s):
            if mass_integrals is None:
                return m.Cts[s, m.t.last()]*m.variance_t[s] == m.sigmas[s, m.t.last()]
            return mass_integrals[s] * m.variance_t[s] == m.sigmas[s, m.t.last()]
        m.variance = pe.Constraint(no_salt_list, rule=rule_variance)

    if n_moments >= 3:
        # defines states for standard deviation
        m.skews = pe.Var(no_salt_list, m.t)
        m.dskews = dae.DerivativeVar(m.skews, wrt=m.t)

        def rule_skews(m,s,t):
            if t == m.t.first():
                return pe.Constraint.Skip
            eps = 1e-4
            std = (m.variance_t[s]**2+eps)**0.5
            return m.dskews[s, t] == m.C[s, t, m.x.last()] * ((t - m.mean_t[s])/std) ** 3
        m.skew_integral = pe.Constraint(no_salt_list, m.t, rule=rule_skews)

        def init_skew_rule(m, s):
            return m.skews[s, m.t.first()] == 0.0
        m.init_skews = pe.Constraint(no_salt_list, rule=init_skew_rule)

        # skew variable
        m.skew_t = pe.Var(no_salt_list, initialize=1.0)

        def rule_skewness(m, s):
            if mass_integrals is None:
                return m.Cts[s, m.t.last()]*m.skew_t[s] == m.skews[s, m.t.last()]
            return mass_integrals[s] * m.skew_t[s] == m.skews[s, m.t.last()]
        m.skewness = pe.Constraint(no_salt_list, rule=rule_skewness)

    if n_moments > 3:
        raise RuntimeError("only three moments supported")


def create_resolution_variables(model, modeler, combined_components):
    m = modeler.pyomo_column.pyomo_model()
    no_salt_list = [cname for cname in model.list_components() if not model.is_salt(cname)]

    m.resolution = pe.Var(combined_components)
    m.list_resolutions = pe.ConstraintList()

    for i, p in enumerate(combined_components):
        s1 = p[0]
        s2 = p[1]
        eps = 1e-5
        w1 = 4*(m.variance_t[s1] ** 2+eps) ** 0.5
        w2 = 4*(m.variance_t[s2] ** 2+eps) ** 0.5
        m.list_resolutions.add(m.resolution[p] == 2.0*(m.mean_t[s1]-m.mean_t[s2])/(w1+w2))




