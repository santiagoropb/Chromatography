from pychrom.modeling.pyomo_modeler import PyomoModeler
from pychrom.tmp.testing_utils import *
from itertools import combinations
import pyomo.environ as pe
import pyomo.dae as dae
import matplotlib.pyplot as plt
import numpy as np

def create_and_initialize_model(noise=False):
    GRM = create_model()

    # create a modeler
    modeler = PyomoModeler(GRM)
    tspan = [0.0]

    # add discontinuity points to time set
    for n, sec in GRM.inlet.sections():
        tspan.append(sec.start_time_sec)

    # add discontinuity points to time set
    for t in np.linspace(0.5, 11.0, 20):
        tspan.append(t)

    for name in GRM.list_components():
        nu = 2.0
        GRM.column.binding_model.set_nu(name, nu)

    if noise:
        bm = GRM.column.binding_model
        for cname in bm.list_components():
            if cname == 'B':
                kd = bm.kd(cname) + np.random.normal(0.0, 2)
                bm.set_kd(cname,kd)
        bm.pprint()

    tspan.append(1500.0)
    q_scale = {'A': 1200.0}
    c_scale = {'A': 50.0}

    modeler.build_model(tspan,
                        model_type='OptimalConvectiveModel',
                        q_scale=q_scale,
                        c_scale=c_scale,
                        options={'smooth':False})

    mass_integrals = {k: 10.0 for k in GRM.list_components()}
    add_integral_variables(GRM, modeler, n_moments=3, mass_integrals=mass_integrals)
    m = modeler.pyomo_column.pyomo_model()

    m.bconc = pe.Var(bounds=(0.0, 150.0))
    m.bconc.fix(100.0)
    m.slope = pe.Var(bounds=(0.0, 1.0))
    m.slope.fix(0.2)
    def rule_salt(m, tt):
        lin = m.x.first()
        if tt == m.t.first():
            return pe.Constraint.Skip
        else:
            if tt < GRM.elute.start_time_sec:
                return m.C['A', tt, lin] == m.bconc
            else:
                return m.C['A', tt, lin] == m.bconc + m.slope*tt

    m.inlet_salt = pe.Constraint(m.t, rule=rule_salt)

    combinations = [('D','C'), ('B','D'), ('B','C')]
    create_resolution_variables(GRM, modeler, combinations)

    #add_performance_variables(GRM, modeler, 0.0, 314.0, ['B'], mass_integrals)

    #m.obj = pe.Objective(expr=-m.resolution[('B', 'C')]**2-m.resolution[('B', 'D')]**2 + m.variance_t['B'])
    #m.obj = pe.Objective(expr=-m.resolution[('B', 'C')]-m.resolution[('B', 'D')] + (m.variance_t['B']+1e-4)**0.5)
    m.obj = pe.Objective(expr=-m.resolution[('B', 'C')]-m.resolution[('B', 'D')] )
    m.obj.deactivate()
    print("done building")
    modeler.discretize_space(30)
    print("done discretizing space")
    modeler.discretize_time(60)
    print("done discretizing time")

    modeler.initialize_variables()

    options = {'constr_viol_tol': 6e-4, 'halt_on_ampl_error': 'yes', 'bound_push':1e-6}
    results = modeler.solve(solver_opts=options)

    m.slope.fixed = False
    m.bconc.fixed = False
    m.obj.activate()
    results = modeler.solve(solver_opts=options)
    m.obj.deactivate()
    m.bconc.setlb(None)
    m.bconc.setub(None)
    m.slope.setlb(None)
    m.slope.setub(None)
    return modeler

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
textos = []


ef = pe.ConcreteModel()
ef.slope = pe.Var(bounds=(None, 1.0))
ef.bconc = pe.Var(bounds=(0.0, 150.0))
ef.non_anticipacivity = pe.ConstraintList()

n_samples = 2
#ef_obj = 0.0
avg_tb = 0.0
avg_tc = 0.0
avg_td = 0.0
for i in range(n_samples):
    modeler = create_and_initialize_model(noise=True)
    m = modeler.pyomo_column.pyomo_model()
    ef.add_component("b{}".format(i), m)
    #ef_obj += m.obj.expr
    avg_tb += m.mean_t['B']/n_samples
    avg_tc += m.mean_t['C']/n_samples
    avg_td += m.mean_t['D']/n_samples
    ef.non_anticipacivity.add(m.bconc == ef.bconc)
    ef.non_anticipacivity.add(ef.slope == m.slope)
    ef.bconc.value = m.bconc.value
    ef.slope.value = m.slope.value

ef_obj = -((avg_tb - avg_tc)**2+1e-4)**0.5-((avg_tb - avg_td)**2+1e-4)**0.5 + ((ef.bconc - 28.3388806643978)**2+1e-4)**0.5 + ((ef.slope - 0.0231777087420414)**2+1e-4)**0.5

ef.obj = pe.Objective(expr=ef_obj)

ef.obj.pprint()
ef.slope.pprint()
ef.bconc.pprint()
ef.non_anticipacivity.pprint()

options = {'constr_viol_tol': 6e-4, 'halt_on_ampl_error': 'yes', 'bound_push':1e-7, 'mu_init':1e-7}

solver = pe.SolverFactory('ipopt')

for k, v in options.items():
    solver.options[k] = v
solver.solve(ef, tee=True)

ef.slope.pprint()
ef.bconc.pprint()

