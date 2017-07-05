import pyomo.environ as pe
import logging
import abc


logger = logging.getLogger(__name__)


class PyomoColumn(abc.ABC):

    def __init__(self, column, dimensionless=True):
        """

        :param model: chromatography model
        :type model: ChromatographyModel
        """

        self._column = column
        self._inlet = getattr(self._column._model(), self._column.left_connection)
        self._outlet = getattr(self._column._model(), self._column.right_connection)

        self.m = pe.ConcreteModel()

        self.dimensionless = dimensionless

    @abc.abstractmethod
    def setup_base(self, tspan, lspan=None, rspan=None):
        """
        Creates base sets and params for modeling a chromatography column with pyomo
        :param tspan:
        :param lspan:
        :param rspan:
        :return: None
        """

    @abc.abstractmethod
    def build_variables(self):
        """
        Create variables for modeling chromatography column with pyomo
        :return: None
        """

    @abc.abstractmethod
    def build_mobile_phase_balance(self):
        """
        Creates PDEs for mobile phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

    @abc.abstractmethod
    def build_stationary_phase_balance(self):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

    @abc.abstractmethod
    def build_adsorption_equations(self):
        """
        Creates PDEs for stationary phase mass balance for modeling chromatography column with pyomo
        :return: boolean
        """

    @abc.abstractmethod
    def build_boundary_conditions(self):
        """
        Creates expressions for boundary conditions
        :return: None
        """

    @abc.abstractmethod
    def build_initial_conditions(self):
        """
        Creates expressions for boundary conditions
        :return: None
        """

    def build_pyomo_model(self, tspan, lspan=None, rspan=None):
        """

        :return:
        """
        # create sets and parameters
        self.setup_base(tspan, lspan, rspan)
        # create variables
        self.build_variables()
        # create constraints
        self.build_mobile_phase_balance()
        self.build_stationary_phase_balance()
        self.build_adsorption_equations()
        self.build_boundary_conditions()
        self.build_initial_conditions()

    def pyomo_model(self):
        """
        Return an instance of the pyomo model for the chromatography column
        :return: pyomo model
        """
        return self.m