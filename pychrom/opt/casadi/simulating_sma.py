from pychrom.modeling.cadet_modeler import CadetModeler
from pychrom.core import *
from pychrom.opt.casadi.build_utils import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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
GRM.binding.is_kinetic = True

# create column
GRM.column = Column(data="column.yml")

# create outlet
GRM.outlet = Outlet(components=comps)

# connect units
GRM.connect_unit_operations('inlet', 'column')
GRM.connect_unit_operations('column', 'outlet')

for name in GRM.list_components():
    nu = 1.0
    GRM.column.binding_model.set_nu(name, nu)

cwrapper = CasadiColumn(GRM.column)
lspan = np.linspace(0, GRM.column.length, 50)
cwrapper.build_model(lspan, nominal_c={'A': 50}, nominal_q={'A': 1200})

# defines grid of times
tspan = np.linspace(0, 1500, 1500)
results = cwrapper.solve(tspan)

for cname in results.components:
    if cname !='A':
        to_plot = results.C.sel(component=cname)
        plot2d = to_plot.sel(col_loc=GRM.column.length)
        plt.plot(plot2d.time, plot2d)
plt.show()

