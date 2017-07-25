from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.cadet_modeler import CadetModeler
import matplotlib.animation as animation
import matplotlib.pyplot as plt

comps = ['salt',
         'lysozyme',
         'cytochrome',
         'ribonuclease']

GRM = GRModel(components=comps)

# create sections
GRM.load = Section(components=comps)
for cname in comps:
    GRM.load.set_a0(cname, 1.0)
GRM.load.set_a0('salt', 50.0)
GRM.load.start_time_sec = 0.0

GRM.wash = Section(components=comps)
GRM.wash.set_a0('salt', 50.0)
GRM.wash.start_time_sec = 10.0

GRM.elute = Section(components=comps)
GRM.elute.set_a0('salt', 100.0)
GRM.elute.set_a1('salt', 0.2)
GRM.elute.start_time_sec = 90.0

# create inlet
GRM.inlet = Inlet(components=comps)
GRM.inlet.add_section('load')
GRM.inlet.add_section('wash')
GRM.inlet.add_section('elute')

# create binding
GRM.salt = 'salt'
GRM.binding = SMABinding(data="sma.yml")
GRM.binding.is_kinetic = True

# create column
GRM.column = Column(data="column.yml")

# create outlet
GRM.outlet = Outlet(components=comps)

# connect units
GRM.connect_unit_operations('inlet', 'column')
GRM.connect_unit_operations('column', 'outlet')

# create a modeler
modeler = CadetModeler(GRM)
ncol=50
npar=10
modeler.discretize_column('column', ncol, npar)

# running a simulation
tspan = range(1500)
retrive_c = 'all'
results = modeler.run_sim(tspan,
                          retrive_c=retrive_c,
                          keep_files=False)


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
        #to_plot = results.C.sel(component=cname)

        to_q = results.Q.sel(component=cname)
        to_plot = to_q.sel(col_loc=0.0)
        for t in results.times:
            for x in results.col_locs:
                for r in results.par_locs:
                    print(cname, float(to_q.sel(time=t, col_loc=x, par_loc=r)))

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
