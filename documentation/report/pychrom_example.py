from pychrom.modeling.cadet_modeler import CadetModeler
from pychrom.core import *
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def create_model(noise=False):

    comps = ['salt',
             'lysozyme',
             'cytochrome',
             'ribonuclease']

    GRM = GRModel(components=comps)

    # create sections
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
    
    # create inlet
    GRM.inlet = Inlet(components=comps)
    GRM.inlet.add_section('load')
    GRM.inlet.add_section('wash')
    GRM.inlet.add_section('elute')
    
    # create binding
    GRM.salt = 'salt'
    GRM.binding = SMABinding(data="sma.yml")
    GRM.binding.is_kinetic = True
    
    # create column
    GRM.column = Column(data="column.yml")

    # create outlet
    GRM.outlet = Outlet(components=comps)

    # connect units
    GRM.connect_unit_operations('inlet', 'column')
    GRM.connect_unit_operations('column', 'outlet')
    
    if noise:
        bm = GRM.column.binding_model
        bm.lamda = bm.lamda + np.random.normal(0,200)

    return GRM

def compute_retention_time(model, results):
    
    time = list(results.C.coords['time'])
    components = results.C.coords['component']
    tr = dict()
    for cname in model.list_components():
        if cname != 'salt':
            traj = results.C.sel(time=time, col_loc=model.column.length, component=cname)
            # compute total concentration
            c_integral = 0.0
            ct_integral = 0.0
            for i, t in enumerate(time):
                if i<len(time)-1:
                    t1 = time[i]
                    t2 = time[i+1]
                    dt = t2-t1
                    c_integral += dt*traj.sel(time=t1)
                    ct_integral += dt*traj.sel(time=t1)*t1
            tr[cname] = float(ct_integral/c_integral)
    return tr
