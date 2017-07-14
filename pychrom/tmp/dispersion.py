import matplotlib.pyplot as plt
import pyomo.environ as pe
import pyomo.dae as dae
import numpy as np
import xarray as xr

m = pe.ConcreteModel()

# define sets
tao = [0.0, 9.0, 10.0, 1500.0]
z = [0.0, 0.0001, 0.0014]
m.t = dae.ContinuousSet(initialize=tao)
m.x = dae.ContinuousSet(initialize=z)

# define variables
m.C = pe.Var(m.t, m.x, initialize=0.0)
m.dCdx = dae.DerivativeVar(m.C, wrt=m.x)
m.dCdt = dae.DerivativeVar(m.C, wrt=m.t)
m.dCdx2 = dae.DerivativeVar(m.C, wrt=(m.x, m.x))

def rule_pde(m, t, x):

    v = 0.000575
    diff = 6.07e-11
    if x == m.x.bounds()[0] or x == m.x.bounds()[1]:
        return pe.Constraint.Skip

    if t == m.t.bounds()[0]:
        return pe.Constraint.Skip

    lhs = m.dCdt[t, x]
    rhs = -v*m.dCdx[t, x] + diff*m.dCdx2[t, x]
    return lhs == rhs

# differential equation
m.pde = pe.Constraint(m.t, m.x, rule=rule_pde)


def rule_bc1(m, t):
    lin = m.x.bounds()[0]
    if t == m.t.bounds()[0]:
        return pe.Constraint.Skip
    lhs = m.C[t, lin]
    rhs = 1.0 if t < 20.0 else 0.0
    return lhs == rhs

# boundary condition
m.bc1 = pe.Constraint(m.t, rule=rule_bc1)


def rule_bc2(m, t):
    lout = m.x.bounds()[1]
    if t == m.t.bounds()[0]:
        return pe.Constraint.Skip
    lhs = m.dCdx[t, lout]
    rhs = 0.0
    return lhs == rhs

# boundary condition
m.bc2 = pe.Constraint(m.t, rule=rule_bc2)

def rule_ic(m, x):
    t0 = m.t.bounds()[0]
    return m.C[t0, x] == 0.0

# initial condition
m.ic = pe.Constraint(m.x, rule=rule_ic)

m.pprint()

# discretizing space
discretizer1 = pe.TransformationFactory('dae.finite_difference')
discretizer1.apply_to(m, nfe=60, wrt=m.x, scheme='BACKWARD')

# discretizing time
discretizer2 = pe.TransformationFactory('dae.finite_difference')
discretizer2.apply_to(m, nfe=60, ncp=1, wrt=m.t)

#discretizer3 = pe.TransformationFactory('dae.finite_difference')
#discretizer3.apply_to(m, nfe=60, wrt=m.t, scheme='BACKWARD')

# solving
opt = pe.SolverFactory('ipopt')
opt.solve(m, tee=True)

# loading solution to x array
nt = len(m.t)
nx = len(m.x)
sorted_x = sorted(m.x)
sorted_t = sorted(m.t)
conc = np.zeros((nt, nx))
for j, t in enumerate(sorted_t):
    for k, x in enumerate(sorted_x):
        conc[j, k] = pe.value(m.C[t, x])
        if conc[j,k] < -1e-5:
            print(conc[j,k])
results = xr.DataArray(conc,
                       coords=[sorted_t, sorted_x],
                       dims=['time', 'location'])
results.plot()
plt.show()


