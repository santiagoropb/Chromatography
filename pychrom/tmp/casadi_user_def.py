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