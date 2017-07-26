from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, Column, Outlet
from pychrom.core.binding_model import SMABinding
from pychrom.modeling.cadet_modeler import CadetModeler
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def check_zeros(dataset, ka, kd, sigma, nu, lamda, salt_name):
    accum = 0.0

    no_salt_list = [cname for cname in dataset.components if cname != salt_name]

    for cname in dataset.components:
        for t in dataset.times:
            for x in dataset.col_locs:
                for r in dataset.par_locs:

                    #check salt residual
                    if cname == salt_name:
                        q_salt = lamda

                        for n in no_salt_list:
                            qj = dataset.Q.sel(component=n, time=t, col_loc=x, par_loc=r)
                            q_salt -= nu[n]*float(qj)

                        value = q_salt - float(dataset.Q.sel(component=cname, time=t, col_loc=x, par_loc=r))

                    else:
                        # check other residuals
                        q_free_sites = lamda
                        q_vals = dict()
                        for n in no_salt_list:
                            qj = dataset.Q.sel(component=n, time=t, col_loc=x, par_loc=r)
                            q_free_sites -= (nu[n] + sigma[n]) * float(qj)
                            q_vals[n] = float(qj)

                        cp = float(dataset.Cp.sel(component=cname, time=t, col_loc=x, par_loc=r))
                        q = float(dataset.Q.sel(component=cname, time=t, col_loc=x, par_loc=r))
                        cp_salt = float(dataset.Cp.sel(component=salt_name, time=t, col_loc=x, par_loc=r))

                        value = ka[cname]*cp*q_free_sites**nu[cname] - kd[cname]*q*cp_salt**nu[cname]

                        if abs(value) > 1e-1:
                            print(cname, value, q_vals, cp, q, cp_salt, q_free_sites)

                    accum += value

                    q = float(dataset.Q.sel(component=cname, time=t, col_loc=x, par_loc=r))


    return accum

def check_zeros2(dataset,bm):
    accum = 0.0

    for cname in dataset.components:
        for t in dataset.times:
            for x in dataset.col_locs:
                for r in dataset.par_locs:
                    c_vars = dict()
                    q_vars = dict()
                    for n in dataset.components:
                        cp = float(dataset.Cp.sel(component=n, time=t, col_loc=x, par_loc=r))
                        q = float(dataset.Q.sel(component=n, time=t, col_loc=x, par_loc=r))
                        c_vars[n] = cp
                        q_vars[n] = q
                    value = bm.new_fads(cname, c_vars, q_vars)
                    print(cname, value)
                    accum += value


if __name__ == "__main__":

    comps = ['salt', 'lysozyme', 'cytochrome', 'ribonuclease']

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
    GRM.binding.is_kinetic = False

    # create column
    GRM.column = Column(data="column.yml")

    # create outlet
    GRM.outlet = Outlet(components=comps)

    # connect units
    GRM.connect_unit_operations('inlet', 'column')
    GRM.connect_unit_operations('column', 'outlet')

    # create a modeler
    modeler = CadetModeler(GRM)
    ncol=3
    npar=2
    modeler.discretize_column('column', ncol, npar)

    # running a simulation
    tspan = np.linspace(0, 1500, 3)
    results = modeler.run_sim(tspan, retrive_c='all', keep_files=True)

    ka = {cname: GRM.binding.ka(cname) for cname in GRM.list_components()}
    kd = {cname: GRM.binding.kd(cname) for cname in GRM.list_components()}
    nu = {cname: GRM.binding.nu(cname) for cname in GRM.list_components()}
    sigma = {cname: GRM.binding.sigma(cname) for cname in GRM.list_components()}
    lamda = GRM.binding.lamda
    print(ka)
    print(kd)
    print(nu)
    print(sigma)
    check_zeros(results, ka, kd, sigma, nu, lamda, GRM.salt)
    #check_zeros2(results, GRM.binding)