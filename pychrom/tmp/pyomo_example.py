from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.pyomo_modeler import PyomoModeler
from pychrom.modeling.cadet_modeler import CadetModeler
import matplotlib.animation as animation
import sys

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
tspan = [0.0, 5.0, 20.0]

# add discontinuity points to time set
for n, sec in GRM.inlet.sections():
    tspan.append(sec.start_time_sec)

# add discontinuity points to time set
for t in np.linspace(9.0, 11.0, 3):
    tspan.append(t)
tspan.append(1500.0)

for name in GRM.list_components():
    nu = 3.1
    GRM.column.binding_model.set_nu(name, nu)

q_scale = {'A': 1200.0}
c_scale = {'A': 50.0}

modeler.build_model(tspan,
                    #model_type='ConvectionModel',
                    model_type='IdealConvectiveModel2',
                    #model_type='IdealDispersiveModel',
                    q_scale=q_scale,
                    c_scale=c_scale,
                    options={'smooth':False})

#m = modeler.pyomo_column.pyomo_model()
#m.pprint()
#sys.exit()
print("done building")
modeler.discretize_space()
print("done discretizing space")
modeler.discretize_time()
print("done discretizing time")

cadet_modeler = CadetModeler(GRM)
ncol=50
npar=10
cadet_modeler.discretize_column('column', ncol, npar)
tspan = range(1500)
trajectories = cadet_modeler.run_sim(tspan, retrive_c='all')

time = trajectories.C.coords['time']
components = trajectories.C.coords['component']
for cname in components:
    if cname!= 'A':
        traj = trajectories.C.sel(time=time, col_loc=GRM.column.length, component=cname)
        plt.plot(time, traj)
plt.show()

results = modeler.run_sim(solver_opts={'halt_on_ampl_error':'yes'})

plt.figure()

time = results.C.coords['time']
components = results.C.coords['component']
for cname in components:
    if cname != 'A':
        traj = results.C.sel(time=time, col_loc=GRM.column.length, component=cname)
        plt.plot(time, traj)
plt.show()