from pychrom.core import *
from pychrom.opt.casadi.build_utils import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt

comps = ['A','B','C']

GRM = GRModel(components=comps)

# create sections
GRM.load = Section(components=comps)
GRM.load.set_a0('A', 50.0)
GRM.load.set_a0('B', 0.26)
GRM.load.set_a0('C', 0.3)
GRM.load.start_time_sec = 0.0

GRM.elute = Section(components=comps)
GRM.elute.set_a0('A', 13.46)
GRM.elute.set_a1('A', 0.26538)
GRM.elute.start_time_sec = 100.0

# create inlet
GRM.inlet = Inlet(components=comps)
GRM.inlet.add_section('load')
GRM.inlet.add_section('elute')

# create binding
GRM.salt = 'A'
GRM.bm = EQSMABinding(data="../data/eqsma1.yml")
GRM.bm.is_kinetic = True

# create column
GRM.column = Column(data="../data/column_eq1.yml")

# create outlet
GRM.outlet = Outlet(components=comps)

# connect units
GRM.connect_unit_operations('inlet', 'column')
GRM.connect_unit_operations('column', 'outlet')

cwrapper = CasadiColumn(GRM.column)
lspan = np.linspace(0, GRM.column.length, 50)
cwrapper.build_model(lspan, scale_st=False)

# defines grid of times
tspan = np.linspace(0, 1000, 1000)
results = cwrapper.solve(tspan)

for cname in results.components:
    if cname !='A':
        to_plot = results.C.sel(component=cname)
        plot2d = to_plot.sel(col_loc=GRM.column.length)
        plt.plot(plot2d.time, plot2d)
plt.show()

for cname in results.components:
    to_plot = results.Q.sel(component=cname)
    to_plot.plot(cmap=plt.cm.gist_ncar)
    plt.show()