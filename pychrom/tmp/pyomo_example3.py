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
    nu = 3.9
    GRM.column.binding_model.set_nu(name, nu)

tspan.append(1500.0)
q_scale = {'A': 1200.0}
c_scale = {'A': 50.0}

modeler.build_model(tspan,
                    model_type='IdealDispersiveModel',
                    q_scale=q_scale,
                    c_scale=c_scale,
                    options={'smooth':False})

add_integral_variables(GRM, modeler, n_moments=3, mass_integrals=None)

print("done building")
modeler.discretize_space(30)
print("done discretizing space")
modeler.discretize_time(60, 1)
print("done discretizing time")

modeler.initialize_variables()

options = {'constr_viol_tol': 6e-4, 'halt_on_ampl_error': 'yes'}
results = modeler.solve(solver_opts=options)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
textos = []

time = results.C.coords['time']
components = results.C.coords['component']

m = modeler.pyomo_column.pyomo_model()
no_salt_list = [cname for cname in GRM.list_components() if not GRM.is_salt(cname)]
for cname in no_salt_list:
    print(cname, pe.value(m.mean_t[cname]),
          pe.value(m.variance_t[cname]**0.5),
          pe.value(m.skew_t[cname]))


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
plt.show()



