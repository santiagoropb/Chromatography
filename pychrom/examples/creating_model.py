from pychrom.model.chromatograpy_model import GRModel
from pychrom.model.section import Section
from pychrom.model.unit_operation import Inlet, Column
from pychrom.model.binding_model import SMABinding


comps = ['salt',
         'lysozyme',
         'cytochrome',
         'ribonuclease']

GRM = GRModel(components=comps)

GRM.load = Section(components=comps)

for cname in comps:
    GRM.load.set_a0(cname, 1.0)
GRM.load.set_a0('salt', 50.0)

GRM.wash = Section(components=comps)
GRM.wash.set_a0('salt', 50.0)

GRM.elute = Section(components=comps)
GRM.elute.set_a0('salt', 100.0)
GRM.elute.set_a1('salt', 0.2)

GRM.inlet = Inlet(components=comps)

print("\n\n######################Binding##############################\n\n")

GRM.binding = SMABinding(data="sma.yml")
GRM.binding.pprint()

print("\n\n######################Column##############################\n\n")
GRM.column = Column(data="column.yml")
GRM.column.pprint()
