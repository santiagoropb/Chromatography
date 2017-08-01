from pychrom.modeling.cadet_modeler import CadetModeler
from pychrom.core import *
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

# create a modeler
modeler = CadetModeler(GRM)
modeler.discretize_column('column', ncol=50, npar=5)

# running a simulation
tspan = np.linspace(0, 1e4, 1000)
retrive_c = 'in_out'
results = modeler.run_sim(tspan, retrive_c=retrive_c)

if retrive_c == 'in_out':

    time = results.C.coords['time']
    components = results.C.coords['component']
    for l in [0.0, 0.014]:
        for cname in components:
            if cname != 'salt':
                traj = results.C.sel(time=time, col_loc=l, component=cname)
                plt.plot(time, traj)
        plt.show()

else:

    for cname in results.components:
        to_plot = results.C.sel(component=cname)
        to_plot.plot()
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
            if cname != 'salt':
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