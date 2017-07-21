from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import LinearBinding
from pychrom.modeling.pyomo_modeler import PyomoModeler
from pychrom.modeling.cadet_modeler import CadetModeler
import matplotlib.animation as animation
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

"""
# running a simulation
cadet_modeler = CadetModeler(GRM)
cadet_modeler.discretize_column('column', 50, 5)
tspan =np.linspace(0, 4e3, 1000)
trajectories = cadet_modeler.run_sim(tspan, retrive_c='all')

for cname in trajectories.components:
    to_plot = trajectories.C.sel(component=cname)
    #to_plot.plot()

    plot2d = to_plot.sel(col_loc=GRM.column.length)
    plt.plot(plot2d.time, plot2d)

plt.show()
"""
# create a modeler
modeler = PyomoModeler(GRM)
tspan = [0.0]

# add discontinuity points to time set
for n, sec in GRM.inlet.sections():
    tspan.append(sec.start_time_sec)

# add discontinuity points to time set
for t in np.linspace(9.0, 11.0, 10):
    tspan.append(t)

tspan.append(4000.0)

modeler.build_model(tspan,
                    #model_type='ConvectionModel',
                    #model_type='DispersionModel',
                    #model_type='IdealConvectiveModel',
                    model_type='IdealDispersiveModel',
                    options={'smooth':False})

print("done building")
modeler.discretize_space()
print("done discretizing space")
modeler.discretize_time()
print("done discretizing time")

modeler.initialize_variables()

results = modeler.run_sim(solver_opts={'halt_on_ampl_error':'yes'})

for cname in results.components:
    to_plot = results.C.sel(component=cname)
    #to_plot.plot()

    plot2d = to_plot.sel(col_loc=GRM.column.length)
    plt.plot(plot2d.time, plot2d)

plt.show()
"""
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
textos = []

def animate(l):
    time = results.C.coords['time']
    components = results.C.coords['component']
    loc = results.C.coords['col_loc'][l]
    print(loc)
    ax1.clear()
    for cname in components:
        if cname != 'A':
            traj = results.C.sel(time=time, col_loc=loc, component=cname)
            ax1.plot(time, traj)
    texto = fig.text(0, 0, 'Location {:.3}'.format(float(loc)))
    textos.append(texto)
    if l>1:
        textos[l-1].remove()

n_locations = len(results.C.coords['col_loc'])
ani = animation.FuncAnimation(fig, animate, interval=n_locations)
plt.show()
"""
