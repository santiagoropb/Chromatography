from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.cadet_modeler import CadetModeler
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import h5py

comps = ['salt',
         'lysozyme',
         'cytochrome',
         'ribonuclease']

GRM = GRModel(components=comps)
GRM.salt = 'salt'

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


GRM.inlet = Inlet(components=comps)
GRM.inlet.add_section('load')
GRM.inlet.add_section('wash')
GRM.inlet.add_section('elute')
GRM.binding = SMABinding(data="sma.yml")
GRM.column = Column(data="column.yml")
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
                traj = results.C.sel(time=time, location=l, component=cname)
                plt.plot(time, traj)

            #else:
            #    traj = results.C.sel(time=time, location=0.014, component=cname)


        plt.show()

else:

    """
    time = results.C.coords['time']
    components = results.C.coords['component']
    for l in [0.0, 0.014]:
        for cname in components:
            if cname != 'salt':
                traj = results.C.sel(time=time, location=l, component=cname)
                plt.plot(time, traj)

                # else:
                #    traj = results.C.sel(time=time, location=0.014, component=cname)

        plt.show()
    """

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    textos = []

    def animate(l):
        time = results.C.coords['time']
        components = results.C.coords['component']
        loc = results.C.coords['location'][l]
        print(loc)
        ax1.clear()
        for cname in components:
            if cname != 'salt':
                traj = results.C.sel(time=time, location=loc, component=cname)
                ax1.plot(time, traj)
        texto = fig.text(0, 0, 'Location {:.3}'.format(float(loc)))
        textos.append(texto)
        if l>1:
            textos[l-1].remove()

    n_locations = len(results.C.coords['location'])
    ani = animation.FuncAnimation(fig, animate, interval=n_locations)
    plt.show()

