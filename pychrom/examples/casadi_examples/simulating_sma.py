from pychrom.core import *
from pychrom.modeling.casadi_modeler import CasadiModeler
import matplotlib.pyplot as plt
import numpy as np
import sys

comps = ['A',
         'B',
         'C',
         'D']

GRM = GRModel(components=comps)

# create sections
GRM.load = Section(components=comps)
for cname in comps:
    GRM.load.set_a0(cname, 10.0)

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


for name in GRM.list_components():
    nu = 1.0
    GRM.column.binding_model.set_nu(name, nu)

# create outlet
GRM.outlet = Outlet(components=comps)

# connect units
GRM.connect_unit_operations('inlet', 'column')
GRM.connect_unit_operations('column', 'outlet')

# create a modeler
modeler = CasadiModeler(GRM)
lspan = np.linspace(0, GRM.column.length, 4)

modeler.build_model(lspan, model_type='IdealConvectiveColumn')


tspan = np.linspace(0, 1500, 3)
results = modeler.run_sim(tspan)

for cname in results.components:
    to_plot = results.C.sel(component=cname)
    to_plot.plot()

    plt.show()
