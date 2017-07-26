from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import LinearBinding
from pychrom.modeling.cadet_modeler import CadetModeler
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def check_zeros(dataset, ka, kd):
    accum = 0.0
    for cname in dataset.components:
        cps = dataset.Cp.sel(component=cname)
        qs = dataset.Q.sel(component=cname)
        for t in dataset.times:
            for x in dataset.col_locs:
                for r in dataset.par_locs:
                    cp = float(cps.sel(time=t, col_loc=x, par_loc=r))
                    q = float(qs.sel(time=t, col_loc=x, par_loc=r))
                    value = ka[cname] * cp - kd[cname] * q
                    print(cname, value)
                    accum += value
    return accum


if __name__ == "__main__":

    comps = ['A', 'B']

    GRM = GRModel(components=comps)

    # create sections
    GRM.sec0 = Section(components=comps)
    for cname in GRM.list_components():
        GRM.sec0.set_a0(cname, 0.00714)
    GRM.sec0.start_time_sec = 0.0

    GRM.sec1 = Section(components=comps)
    GRM.sec1.start_time_sec = 10.0

    # create inlet
    GRM.inlet = Inlet(components=comps)
    GRM.inlet.add_section('sec0')
    GRM.inlet.add_section('sec1')

    GRM.column = Column(components=comps)

    # defining adsorption
    GRM.adsorption = LinearBinding(components=comps)
    binding = GRM.adsorption
    binding.is_kinetic = False
    binding.set_ka('A', 1.14)
    binding.set_kd('A', 0.02)
    binding.set_ka('B', 0.98)
    binding.set_kd('B', 0.01)

    # defining column
    column = GRM.column
    column.dispersion = 5.75e-08
    column.length = 0.014
    column.column_porosity = 0.37

    # component A
    column.set_film_diffusion('A', 6.90000000e-06)
    column.set_par_diffusion('A', 6.07000000e-11)
    # component B
    column.set_film_diffusion('B', 6.90000000e-06)
    column.set_par_diffusion('B', 6.07000000e-11)

    column.particle_porosity = 0.75
    column.particle_radius = 4.5e-05
    column.velocity = 0.000575
    column.binding_model = binding

    # create outlet
    GRM.outlet = Outlet(components=comps)

    # connect units
    GRM.connect_unit_operations('inlet', 'column')
    GRM.connect_unit_operations('column', 'outlet')

    # create a modeler
    modeler = CadetModeler(GRM)
    ncol=50
    npar=5
    modeler.discretize_column('column', ncol, npar)

    # running a simulation
    tspan = np.linspace(0, 4e3, 100)
    results = modeler.run_sim(tspan, retrive_c='all')

    ka = {cname: binding.ka(cname) for cname in GRM.list_components()}
    kd = {cname: binding.kd(cname) for cname in GRM.list_components()}

    print(check_zeros(results, ka, kd))

