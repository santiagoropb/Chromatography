from pychrom.modeling.cadet_modeler import CadetModeler
from pychrom.core import *
from pychrom.opt.casadi.build_utils import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

comps = ['A']

GRM = GRModel(components=comps)

# create sections
GRM.sec0 = Section(components=comps)
GRM.sec0.set_a0('A', 0.00714)
GRM.sec0.start_time_sec = 0.0

GRM.sec1 = Section(components=comps)
GRM.sec1.set_a0('A', 0.00714)
GRM.sec1.start_time_sec = 4000.0

# create inlet
GRM.inlet = Inlet(components=comps)
GRM.inlet.add_section('sec0')
GRM.inlet.add_section('sec1')

GRM.column = Column(components=comps)

GRM.adsorption = MCLBinding(components=comps)
binding = GRM.adsorption
binding.is_kinetic = False
binding.set_ka('A', 1.14)
binding.set_kd('A', 0.002)
binding.set_qmax('A', 4.88)

column = GRM.column
column.dispersion = 5.75e-08
column.length = 0.014
column.column_porosity = 0.37
column.set_film_diffusion('A', 6.90000000e-06)
column.set_init_c('A', 0.0)
column.set_init_q('A', 0.0)
column.set_par_diffusion('A', 6.07000000e-11)
column.particle_porosity = 0.75
column.particle_radius = 4.5e-05
column.velocity = 0.000575
column.binding_model = binding

# create outlet
GRM.outlet = Outlet(components=comps)

# connect units
GRM.connect_unit_operations('inlet', 'column')
GRM.connect_unit_operations('column', 'outlet')

cwrapper = CasadiColumn(GRM.column)
lspan = np.linspace(0, column.length, 50)
m = cwrapper.build_model(lspan)

# defines grid of times
tspan = np.linspace(0, 4e3, 1000)
m.grid_t = [t for t in tspan]

# defines dae
dae = {'x': ca.vertcat(*m.states),
       'z': ca.vertcat(*m.algebraics),
       'p': [],
       't': m.t,
       'ode': ca.vertcat(*m.ode),
       'alg': ca.vertcat(*m.alg_eq)}

opts = {'grid': m.grid_t, 'output_t0': True}

integrator = ca.integrator('I', 'idas', dae, opts)

sol = integrator(x0=m.state_ic, z0=m.algebraic_ic)

results = cwrapper.store_values_in_data_set(sol)

for cname in results.components:
    to_plot = results.C.sel(component=cname)
    plot2d = to_plot.sel(col_loc=GRM.column.length)
    plt.plot(plot2d.time, plot2d)
plt.show()