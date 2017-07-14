
from pychrom.core.binding_model import SMABinding
import xarray as xr

from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.cadet_modeler import CadetModeler

import numpy as np

def create_model():

    comps = ['A', 'B', 'C', 'D']

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
    GRM.binding.is_kinetic = False

    # create column
    GRM.column = Column(data="column.yml")

    # create outlet
    GRM.outlet = Outlet(components=comps)

    # connect units
    GRM.connect_unit_operations('inlet', 'column')
    GRM.connect_unit_operations('column', 'outlet')
    return GRM

def simulate_model(m):
    cadet_modeler = CadetModeler(m)
    ncol = 50
    npar = 10
    cadet_modeler.discretize_column('column', ncol, npar)
    tspan = range(1500)
    trajectories = cadet_modeler.run_sim(tspan, retrive_c='all')
    return trajectories

def check_sma(m, trajectories):

    bm = m.column.binding_model

    x = trajectories.col_locs
    t = trajectories.times
    s = trajectories.components
    r = trajectories.par_locs
    tt = [t[i] for i in range(1,2)]
    xx = [x[i] for i in range(1,2)]
    rr = [r[i] for i in range(1, 2)]
    accum = 0.0
    for i in tt:
        for j in xx:
            for k in rr:
                cvars = dict()
                qvars = dict()
                for w in s:
                    cvars[w] = float(trajectories.Cp.sel(component=w,
                                                   time=i,
                                                   col_loc=j,
                                                   par_loc=k))

                    qvars[w] = float(trajectories.Q.sel(component=w,
                                                  time=i,
                                                  col_loc=j,
                                                  par_loc=k))
                #print(cvars)
                for w in s:
                    accum += bm.f_ads(w,cvars,qvars)
                    #print(accum)
    return accum

m = create_model()
results = simulate_model(m)
#print(check_sma(m, results))

def f(x):
    n_components = m.num_components
    for i, c in enumerate(m.list_components()):
        m.column.binding_model.set_kads(c, x[i])
        m.column.binding_model.set_kdes(c, x[i+n_components])
    accum = check_sma(m,results)
    print(x,accum)
    return accum**2

from scipy.optimize import minimize

x = np.zeros(m.num_components*2)
print(minimize(f, x))