from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
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

print("\n\n######################Inlet##############################\n\n")
GRM.inlet = Inlet(components=comps)
GRM.inlet.add_section('load')
GRM.inlet.add_section('wash')
GRM.inlet.add_section('elute')
GRM.inlet.pprint()

print("\n\n######################Sections##############################\n\n")
GRM.load.pprint()
GRM.wash.pprint()
GRM.elute.pprint()
print("\n\n######################Binding##############################\n\n")

GRM.binding = SMABinding(data="sma.yml")
GRM.binding.pprint()

print("\n\n######################Column##############################\n\n")
GRM.column = Column(data="column.yml")
GRM.column.pprint()

print("\n\n######################Outlet##############################\n\n")
GRM.outlet = Outlet(components=comps)
GRM.outlet.pprint()

GRM.connect_unit_operations('inlet', 'column')
GRM.connect_unit_operations('column', 'outlet')

disct_kwargs = dict()
disct_kwargs['ncol'] = 50
disct_kwargs['npar'] = 10

tspan = range(1500)
filename = "first_model.h5"
GRM._write_to_cadet_input_file(filename,
                              tspan,
                              disct_kwargs,
                              dict())

l = []
f = h5py.File(filename, 'r')
f.visit(l.append)