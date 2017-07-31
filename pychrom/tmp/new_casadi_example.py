from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.casadi.new_models import CasadiColumn
import matplotlib.pyplot as plt
import casadi as ca
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
GRM.binding.is_kinetic = False

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


model = CasadiColumn(GRM.column)
lspan = np.linspace(0, GRM.column.length, 100)
m = model.build_model(lspan)

# defines grid of times
tspan = np.linspace(0, 1500, 1500)
m.grid_t = [t for t in tspan]



# defines dae
dae = {'x': ca.vertcat(*m.states),
       'z': ca.vertcat(*m.algebraics),
       't': m.t,
       'ode': ca.vertcat(*m.ode),
       'alg': ca.vertcat(*m.alg_eq)}

opts = {'grid': m.grid_t, 'output_t0': True}

integrator = ca.integrator('I', 'idas', dae, opts)

sol = integrator(x0=m.state_ic,
                 z0=m.algebraic_ic)
#results = model.store_values_in_data_set(sol)

