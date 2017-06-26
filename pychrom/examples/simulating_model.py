from pychrom.model.chromatograpy_model import GRModel
from pychrom.model.section import Section
from pychrom.model.unit_operation import Inlet, Column, Outlet
from pychrom.model.binding_model import SMABinding
from pychrom.sim.cadet_modeler import CadetModeler
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
modeler.discretize_column('column', 50, 10)

# running a simulation
tspan = range(1500)
modeler.run_sim(tspan)


