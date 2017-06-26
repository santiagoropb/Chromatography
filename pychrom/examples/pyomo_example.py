from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.binding_model import SMABinding
import pyomo.environ as pe

data_filename = "example.yml"

# pychrom
cmodel = GRModel(data_filename)
cmodel.binding = SMABinding()

# building pyomo model with help from cmodel
components = cmodel.list_components()
m = pe.ConcreteModel()
m.c_var = pe.Var(components)
m.q_var = pe.Var(components)

#m.pprint()

comp_name = components[0]
print(comp_name)
cmodel.salt = 'salt'
dq_dt = cmodel.binding.f_ads(comp_name, m.c_var, m.q_var)
print(dq_dt)


