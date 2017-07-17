from pychrom.modeling.pyomo.models import IdealConvectiveColumn, ConvectionModel
from pychrom.modeling.pyomo.models import IdealDispersiveColumn, DispersionModel
from pychrom.core.unit_operation import Column
import pyomo.environ as pe
import warnings
import logging

logger = logging.getLogger(__name__)


class PyomoModeler(object):

    def __init__(self, model):
        """

        :param model: chromatography model
        :type model: ChromatographyModel
        """

        self._model = model
        columns = self._model.list_unit_operations(unit_type=Column)
        n_col = len(columns)
        assert n_col == 1, 'PyomoModeler only supports models with 1 column for now'
        if not self._model.is_fully_specified(print_out=True):
            raise RuntimeError('PyomoModeler requires a fully specified chromatography model')

        # gets helpers to build pyomo model
        self._column = getattr(self._model, columns[0])
        self._inlet = getattr(self._model, self._column.left_connection)
        self._outlet = getattr(self._model, self._column.right_connection)

        # alias for pyomo model
        self.pyomo_column = None

        # determine if concentration variables are being scaled
        self._scale_q = False
        self._scale_c = False

        # tmp flag
        self.wq = False

    def build_model(self,
                    tspan,
                    model_type,
                    lspan=None,
                    rspan=None,
                    q_scale=None,
                    c_scale=None,
                    options=None):

        if options is None:
            options = dict()

        if isinstance(q_scale, dict):
            options['scale_q'] = True
            self._scale_q = True
        if isinstance(c_scale, dict):
            options['scale_c'] = True
            self._scale_c = True

        if model_type == 'ConvectionModel':
            self.pyomo_column = ConvectionModel(self._column)
            self.pyomo_column.build_pyomo_model(tspan, lspan=lspan, rspan=None, **options)
        elif model_type == 'DispersionModel':
            self.pyomo_column = DispersionModel(self._column)
            self.pyomo_column.build_pyomo_model(tspan, lspan=lspan, rspan=None, **options)
        elif model_type == 'IdealConvectiveModel':
            self.pyomo_column = IdealConvectiveColumn(self._column)
            self.pyomo_column.build_pyomo_model(tspan, lspan=lspan, rspan=None, **options)
        elif model_type == 'IdealDispersiveModel':
            self.pyomo_column = IdealDispersiveColumn(self._column)
            self.pyomo_column.build_pyomo_model(tspan, lspan=lspan, rspan=None, **options)
        else:
            raise NotImplementedError("Model not supported yet")

        m = self.pyomo_column.m
        # change scales in pyomo model
        if self._scale_c:
            for k, v in c_scale.items():
                m.sc[k] = v

        if self._scale_q:
            for k, v in q_scale.items():
                m.sq[k] = v

    def discretize_space(self):

        m = self.pyomo_column.m

        # Discretize using Finite Difference
        discretizer = pe.TransformationFactory('dae.finite_difference')
        discretizer.apply_to(m, nfe=50, wrt=m.x, scheme='BACKWARD')

        #discretizer = pe.TransformationFactory('dae.collocation')
        #discretizer.apply_to(m, nfe=40, ncp=3, wrt=m.x)

    def discretize_time(self):
        m = self.pyomo_column.m

        # Discretize using Finite elements and collocation
        discretizer = pe.TransformationFactory('dae.collocation')
        discretizer.apply_to(m, nfe=60, ncp=1, wrt=m.t)

    def initialize_variables(self, trajectories=None):
        self.pyomo_column.initialize_variables(trajectories=trajectories)

    def run_sim(self,
                solver='ipopt',
                solver_opts=None):

        opt = pe.SolverFactory(solver)
        if isinstance(solver_opts, dict):
            for k, v in solver_opts.items():
                opt.options[k] = v

        m = self.pyomo_column.m
        opt.solve(m, tee=True)

        results_set = self.pyomo_column.store_values_in_data_set()
        return results_set

