from pychrom.core import *
from pychrom.opt.pyomo.build_utils import *
from pychrom.opt.casadi.build_utils import *
import matplotlib.pyplot as plt
import numpy as np

from pychrom.opt.pyomo.pyomo_utils import *

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
GRM.elute.set_a1('A', 0.26)
GRM.elute.start_time_sec = 100

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

#casadi modeler
cwrapper = CasadiColumn(GRM.column)
lspan = np.linspace(0, GRM.column.length, 50)
cwrapper.build_model(lspan, scale_st=False)

# defines grid of times
tspanc = np.linspace(0, 1000, 1000)
results_c = cwrapper.solve(tspanc)

# create a modeler
model = PyomoColumn(GRM.column)
tspan = [t for t in np.linspace(0.0, 100, 10)]
tspan.append(1000.0)

model.build_model(tspan, scale_st=False)
#model.m.pprint()

model.discretize_space(40)
model.discretize_time(60, 1)

model.initialize_variables(results_c)
#CheckInstanceFeasibility(model.m, 1e-3)

results = model.solve(solver_opts={'halt_on_ampl_error':'yes'})

for cname in results.components:
    if cname !='A':
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
