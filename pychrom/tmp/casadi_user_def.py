"""
from casadi import *


class test0(Callback):
  def __init__(self, name, opts = {}):
    Callback.__init__(self)
    self.construct(name, opts)

  def get_n_in(self):  return 1
  def get_n_out(self): return 2
  def get_sparsity_in(self,n_in): return Sparsity.dense(2,1)
  def get_sparsity_out(self,n_out):
    if n_out == 0: return Sparsity.dense(1,2)
    if n_out == 1: return Sparsity.dense(1,1)

  def init(self):
    print 'initialising object'

  def eval(self, arg):
    x = arg[0]
    f = sin(x[0])
    jac = horzcat(cos(x[0]),0)
    return [jac, f]

  def has_jacobian(self): return True

  def get_jacobian(self,name,opts):

    return Function('jacobian',[x],[vertcat(DM(2,2),DM(1,2))])


class test(Callback):
  def __init__(self, name, opts = {}):
    Callback.__init__(self)
    self.construct(name, opts)

  def get_n_in(self):  return 1
  def get_n_out(self): return 1
  def get_sparsity_in(self,n_in): return Sparsity.dense(2,1)
  def get_sparsity_out(self,n_out): return Sparsity.dense(1,1)


  def init(self):
    self.low_level = test0('low_level')

  def eval(self, arg):

    [jac,f] = self.low_level(arg[0])

    return [f]

  def has_jacobian(self): return True

  def get_jacobian(self,name,opts):
    x = MX.sym('x',2,1)
    jacSym = self.low_level(x)[0]
    return Function('jacobian',[x],[jacSym])

my_sin = test('my_sin')
x = MX.sym('x',2,1)

nlp = {'x':x,'f':-my_sin(x)}
solver = nlpsol('solver',"ipopt",nlp)

arg = {}
arg['lbx'] = 0
arg['ubx'] = 3.14
sol = solver(**arg)


nlp2 = {'x':x,'f':-sin(x[0])}
solver2 = nlpsol('solver',"ipopt",nlp2)
sol2 = solver2(**arg)

print sol['x']-sol2['x']
"""
from scipy.optimize import minimize_scalar
import pyomo.environ as pe
import matplotlib.pyplot as plt
import numpy as np
import math

b = 100.0
e = 200.0

def f1(x):
    return 400-3*x*3

def f2(x):
    return 50.0+x

def f3(x):
    return 50+2*x+x*3

def jump_f(x):
    if x<=b:
        return f1(x)
    else:
        return f2(x)

def s(x,x0,a):
    return 1.0/(1+math.exp(-a*(x-x0)))


class SmoothFunction(object):

    def __init__(self,
                 left_fun,
                 right_fun,
                 disc_point,
                 start,
                 end):

        self._f1 = left_fun
        self._f2 = right_fun
        self._b = disc_point
        self._n_points = 50
        self._start = start
        self._end = end
        self._band = 0.02
        self._k = 1.0

    def f1(self, x):
        return self._f1(x)

    def f2(self, x):
        return self._f2(x)

    def discontinuous_f(self, x):
        if x <= b:
            return self.f1(x)
        else:
            return self.f2(x)

    def find_k(self, tee=False):

        m = pe.ConcreteModel()
        m.k = pe.Var(initialize=1.0, bounds=(0.0, None))
        m.s = pe.Var(initialize=0.0, bounds=(0.0, None))
        n_points = self._n_points

        start = self._start + 0.6*(self._b-self._start)
        end = start = self._b + 0.4*(self._end-self._b)
        data_p = np.linspace(start,
                             end,
                             n_points)

        sampled = list(map(self.discontinuous_f, data_p))

        def init_y(m, i):
            return sampled[i]

        m.y = pe.Var(range(n_points), initialize=init_y)

        m.c_list = pe.ConstraintList()
        for i, x in enumerate(data_p):
            sigma = 1.0 / (1 + pe.exp(-m.k * (x - b)))
            m.c_list.add(m.y[i] == (1 - sigma) * self.f1(x) + sigma * self.f2(x))

        sigma = 1.0 / (1 + pe.exp(-m.k * self._band * b))
        m.smooth = pe.Constraint(expr=0.05 + m.s == sigma * (1 - sigma))

        m.obj = pe.Objective(expr=sum((m.y[i] - sampled[i]) ** 2 for i in range(n_points)) + m.s**2, sense=pe.minimize)

        opt = pe.SolverFactory('ipopt')
        opt.solve(m, tee=tee)
        self._k = pe.value(m.k)

    def __call__(self, *args, **kwargs):
        x = args[0]
        sigma = 1.0/(1.0+pe.exp(-self._k*(x-self._b)))
        return (1-sigma)*self.f1(x) + sigma*self.f2(x)


def smooth_sections(list_functions, list_points):

    n_functions = len(list_functions)

    fl = list_functions[0]
    tl = list_points[0]
    for i in range(1, n_functions):

        fr = list_functions[i]
        tm = list_points[i]
        tr = list_points[i+1]
        my_fun = SmoothFunction(fl, fr, tm, tl, tr)
        my_fun.find_k(tee=True)
        fl = my_fun
        tl = list_points[i]

    return my_fun


funcs = [f1, f2, f3]
bpoints = [0.0, 22.0, 50.0, 70.0]

smoothed = smooth_sections(funcs, bpoints)
all_x = np.linspace(bpoints[0], bpoints[-1], 1000)
all_y = list(map(smoothed, all_x))
plt.plot(all_x, all_y)

for i, fn in enumerate(funcs):
    l = np.linspace(bpoints[i], bpoints[i+1], 100)
    z = list(map(fn, l))
    plt.plot(l, z, 'r')
    if i>0:
        plt.plot([bpoints[i], bpoints[i]], [fn(bpoints[i]), funcs[i-1](bpoints[i])],'r')

plt.show()




x1 = np.linspace(0.0, b, 1000)
y1 = list(map(f1, x1))

x2 = np.linspace(b, e, 1000)
y2 = list(map(f2, x2))

my_fun = SmoothFunction(f1, f2, b, 0.0, e)
my_fun.find_k(tee=True)

sol_k = my_fun._k

def sm(x):
    return s(x, b, sol_k)

def ds(x):
    return sm(x)*(1-sm(x))

x3 = np.linspace(0.0, e, 1000)
y3 = list(map(my_fun, x3))

smooth = list(map(sm, x3))
dsmooth = list(map(ds, x3))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x1, y1)
ax1.plot(x2, y2)
ax1.plot(x3, y3)
#ax2 = fig.add_subplot(1, 3, 2)
#ax2.plot(x3, smooth)
ax3 = fig.add_subplot(1, 2, 2)
ax3.plot(x3, dsmooth)

plt.show()



