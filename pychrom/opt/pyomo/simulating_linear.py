from pychrom.core import *
from pychrom.opt.pyomo.build_utils import *
import matplotlib.pyplot as plt
import numpy as np

comps = ['A', 'B']

GRM = GRModel(components=comps)

# create sections
GRM.sec0 = Section(components=comps)
for cname in GRM.list_components():
    GRM.sec0.set_a0(cname, 0.00714)
GRM.sec0.start_time_sec = 0.0

GRM.sec1 = Section(components=comps)
GRM.sec1.start_time_sec = 10.0

# create inlet
GRM.inlet = Inlet(components=comps)
GRM.inlet.add_section('sec0')
GRM.inlet.add_section('sec1')

GRM.column = Column(components=comps)

# defining adsorption
GRM.adsorption = LinearBinding(components=comps)
binding = GRM.adsorption
binding.is_kinetic = True
binding.set_ka('A', 1.14)
binding.set_kd('A', 0.02)
binding.set_ka('B', 0.98)
binding.set_kd('B', 0.01)

# defining column
column = GRM.column
column.dispersion = 5.75e-08
column.length = 0.014
column.column_porosity = 0.37

# component A
column.set_film_diffusion('A', 6.90000000e-06)
column.set_par_diffusion('A', 6.07000000e-11)
# component B
column.set_film_diffusion('B', 6.90000000e-06)
column.set_par_diffusion('B', 6.07000000e-11)

column.particle_porosity = 0.75
column.particle_radius = 4.5e-05
column.velocity = 0.000575
column.binding_model = binding

# create outlet
GRM.outlet = Outlet(components=comps)

# connect units
GRM.connect_unit_operations('inlet', 'column')
GRM.connect_unit_operations('column', 'outlet')

# create a modeler
model = PyomoColumn(GRM.column)
tspan = [t for t in np.linspace(0.0, 11.0, 15)]
tspan.append(4000.0)


model.build_model(tspan)
model.m.pprint()

model.discretize_space(50)
model.discretize_time(60)

model.initialize_variables()

results = model.solve(solver_opts={'halt_on_ampl_error':'yes'})

for cname in results.components:
    to_plot = results.C.sel(component=cname)
    #to_plot.plot()
    plot2d = to_plot.sel(col_loc=GRM.column.length)
    plt.plot(plot2d.time, plot2d)
    #plt.show()
plt.show()

for cname in results.components:
    to_plot = results.Q.sel(component=cname)
    to_plot.plot(cmap=plt.cm.gist_ncar)
    plt.show()
