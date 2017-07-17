from pychrom.modeling.casadi.models import ConvectionModel, DispersionModel
from pychrom.core.unit_operation import Column
import casadi as ca
import warnings
import logging

logger = logging.getLogger(__name__)


class CasadiModeler(object):

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
        self.casadi_column = None

    def build_model(self,
                    lspan,
                    model_type,
                    rspan=None,
                    options=None):

        if options is None:
            options = dict()

        if model_type == 'ConvectionModel':
            self.casadi_column = ConvectionModel(self._column)
            self.casadi_column.build_model(lspan, rspan=None, **options)
            self._parameters = [self._column.velocity, 0.0]
        elif model_type == 'DispersionModel':
            self.casadi_column = DispersionModel(self._column)
            self.casadi_column.build_model(lspan, rspan=None, **options)
            self._parameters = [self._column.velocity, self._column.dispersion]
        else:
            raise NotImplementedError("Model not supported yet")

        """
        m = self.casadi_column.model()
        print(m.states)
        print(m.algebraics)
        print(m.parameters)
        print(m.ode)
        #print(m.alg_eq)
        """

    def run_sim(self, tspan):

        # get model
        m = self.casadi_column.model()

        # defines grid of times
        m.grid_t = [t for t in tspan]

        # defines dae
        dae = {'x': ca.vertcat(*m.states),
               'z': ca.vertcat(*m.algebraics),
               'p': ca.vertcat(*m.parameters),
               't': m.t,
               'ode': ca.vertcat(*m.ode),
               'alg': ca.vertcat(*m.alg_eq)}

        opts = {'grid': m.grid_t, 'output_t0': True}

        integrator = ca.integrator('I', 'idas', dae, opts)

        sol = integrator(x0=m.init_conditions, p=self._parameters)
        return self.casadi_column.store_values_in_data_set(sol)