from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.pyomo_modeler import PyomoModeler
from pychrom.modeling.cadet_modeler import CadetModeler
import matplotlib.animation as animation
from itertools import combinations
import pyomo.environ as pe
import pyomo.dae as dae

import matplotlib.pyplot as plt
import numpy as np

comps = ['A',
         'B',
         'C',
         'D']

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
    nu = 3.5
    #print(name, nu)
    if nu>0:
        GRM.column.binding_model.set_nu(name, nu)

tspan.append(1500.0)
q_scale = {'A': 1200.0}
c_scale = {'A': 50.0}

modeler.build_model(tspan,
                    #model_type='ConvectionModel',
                    #model_type='DispersionModel',
                    #model_type='IdealConvectiveModel',
                    model_type='IdealDispersiveModel',
                    q_scale=q_scale,
                    c_scale=c_scale,
                    options={'smooth':False})

m = modeler.pyomo_column.pyomo_model()

# testing
no_salt_list = [cname for cname in GRM.list_components() if not GRM.is_salt(cname)]

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
m.miu_t = pe.Var(no_salt_list, initialize=1.0)

def rule_retention(m,s):
    return m.Cts[s, m.t.last()]*m.miu_t[s] == m.Trs[s, m.t.last()]

m.retention = pe.Constraint(no_salt_list, rule=rule_retention)

# defines states for standard deviation
m.sigmas = pe.Var(no_salt_list, m.t)
m.dsigmas = dae.DerivativeVar(m.sigmas, wrt=m.t)

def rule_sigmas(m,s,t):
    if t == m.t.first():
        return pe.Constraint.Skip
    #return m.dsigmas[s, t] == m.C[s, t, m.x.last()]*(t-m.Trs[s, m.t.last()]/10.0)**2
    return m.dsigmas[s, t] == m.C[s, t, m.x.last()] * (t - m.miu_t[s]) ** 2
m.sigma_integral = pe.Constraint(no_salt_list, m.t, rule=rule_sigmas)

def init_sigmas_rule(m, s):
    return m.sigmas[s, m.t.first()] == 0.0
m.init_sigmas = pe.Constraint(no_salt_list, rule=init_sigmas_rule)


# retention times variable
m.variance_t = pe.Var(no_salt_list, initialize=1.0)

def rule_variance(m,s):
    return m.Cts[s, m.t.last()]*m.variance_t[s] == m.sigmas[s, m.t.last()]

m.variance = pe.Constraint(no_salt_list, rule=rule_variance)

# combination of components into tuples
combined_components = list(combinations(no_salt_list, 2))
n_combinations = len(combined_components)
m.resolution = pe.Var(combined_components)
m.list_resolutions = pe.ConstraintList()

for i, p in enumerate(combined_components):
    s1 = p[0]
    s2 = p[1]
    eps = 1e-5
    w1 = 4*(m.variance_t[s1] ** 2+eps) ** 0.5
    w2 = 4*(m.variance_t[s2] ** 2+eps) ** 0.5
    m.list_resolutions.add(m.resolution[p] == 2.0*(m.miu_t[s1]-m.miu_t[s2])/(w1+w2))

print("done building")
modeler.discretize_space(30)
print("done discretizing space")
modeler.discretize_time(60)
print("done discretizing time")


modeler.initialize_variables()

options = {'constr_viol_tol': 6e-4, 'halt_on_ampl_error': 'yes'}
results = modeler.run_sim(solver_opts=options)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
textos = []

time = results.C.coords['time']
components = results.C.coords['component']

for cname in no_salt_list:
    print(cname, pe.value(m.miu_t[cname]),
          pe.value(m.variance_t[cname]**0.5),
          pe.value(m.Cts[cname, m.t.last()]),
          pe.value(m.Trs[cname, m.t.last()]),
          pe.value(m.sigmas[cname, m.t.last()]))

m.resolution.pprint()

trs = dict()
sigs = dict()
color = {'B':'b', 'C': 'y',  'D':'r'}
for cname in no_salt_list:
    #trs[cname] = pe.value(m.Trs[cname, m.t.last()]/10.0)
    trs[cname] = pe.value(m.miu_t[cname])
    #sigs[cname] = pe.value((m.sigmas[cname, m.t.last()].value / 10.0)**0.5)
    sigs[cname] = pe.value(m.variance_t[cname]**0.5)

for cname in GRM.list_components():
    if cname!= 'A':
        traj = results.C.sel(time=time, col_loc=GRM.column.length, component=cname)
        plt.plot(time, traj, color=color[cname])
        peak_t = trs[cname]
        sd = sigs[cname]
        approx_peak = traj.sel(time=peak_t, method='nearest')
        plt.plot([peak_t, peak_t], [-0.01, approx_peak], color=color[cname])
        plt.plot([peak_t-2*sd, peak_t+2*sd], [-0.01, -0.01], color=color[cname])
plt.show()



