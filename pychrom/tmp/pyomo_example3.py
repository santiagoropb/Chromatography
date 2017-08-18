from pychrom.modeling.pyomo_modeler import PyomoModeler
from pychrom.tmp.testing_utils import *
from itertools import combinations
import pyomo.environ as pe
import pyomo.dae as dae
import matplotlib.pyplot as plt
import numpy as np

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
no_salt_list = [cname for cname in GRM.list_components() if not GRM.is_salt(cname)]

m.bconc = pe.Var(bounds=(0.0, 150.0))
m.bconc.fix(100)
m.slope = pe.Var(bounds=(0.0, 10.0))
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
#m.obj = pe.Objective(expr=-m.resolution[('B', 'C')]-m.resolution[('B', 'D')] )
m.obj = pe.Objective(expr=-m.resolution[('D', 'C')]-m.resolution[('B', 'C')])
#m.obj = pe.Objective(expr=-m.resolution[('B', 'D')]-m.resolution[('D', 'C')])
m.obj.deactivate()
print("done building")
modeler.discretize_space(30)
print("done discretizing space")
modeler.discretize_time(60)
print("done discretizing time")

modeler.initialize_variables()

options = {'constr_viol_tol': 6e-4, 'halt_on_ampl_error': 'yes'}
results = modeler.solve(solver_opts=options)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
textos = []

time = results.C.coords['time']
components = results.C.coords['component']

for cname in no_salt_list:
    print(cname, pe.value(m.mean_t[cname]),
          pe.value(m.variance_t[cname]**0.5),
          pe.value(m.skew_t[cname]))

"""
for t in m.t:
    if t > GRM.elute.start_time_sec:
        m.inlet_salt[t].deactivate()
"""
m.slope.fixed = False
m.bconc.fixed = False
m.obj.activate()
#results = modeler.solve(solver_opts=options)

for cname in no_salt_list:
    print(cname, pe.value(m.mean_t[cname]),
          pe.value(m.variance_t[cname]**0.5),
          pe.value(m.skew_t[cname]))
m.slope.pprint()
m.bconc.pprint()

trs = dict()
sigs = dict()
color = {'B':'b', 'C': 'y',  'D':'r'}
for cname in no_salt_list:
    trs[cname] = pe.value(m.mean_t[cname])
    sigs[cname] = pe.value(m.variance_t[cname]**0.5)

for cname in no_salt_list:
    traj = results.C.sel(time=time, col_loc=GRM.column.length, component=cname)
    plt.plot(time, traj, color=color[cname])
    peak_t = trs[cname]
    sd = sigs[cname]
    approx_peak = traj.sel(time=peak_t, method='nearest')
    plt.plot([peak_t, peak_t], [-0.01, approx_peak], color=color[cname])
    plt.plot([peak_t-2*sd, peak_t+2*sd], [-0.01, -0.01], color=color[cname])

plt.xlabel("time")
plt.ylabel("Concentration")
plt.show()

plt.figure()
Cs = []
traj = results.C.sel(time=time, col_loc=GRM.column.length, component='A')
plt.plot(time, traj)
plt.xlabel("time")
plt.ylabel("Salt Concentration")
plt.ylim([None,500])
plt.show()

for cname in results.components:
    to_plot = results.Q.sel(component=cname)
    to_plot.plot(cmap=plt.cm.gist_ncar)
    plt.show()

