from __future__ import print_function
import abc
import pandas as pd
import logging
import yaml
import six

logger = logging.getLogger(__name__)


class BindingModel(abc.ABC):

    def __init__(self):

        # define scalar params container
        self._scalar_params = dict()

        # define components and component indexed parameters
        self._components = set()
        self._indexed_params = dict()


    @property
    def num_components(self):
        """
        Returns number of components in the binding model
        """
        return len(self._components)

    @property
    def num_scalar_parameters(self):
        """
        Returns number of scalar parameters
        """
        return len(self._scalar_params)

    @property
    def num_indexed_parameters(self):
        """
        Returns number of indexed parameters
        """
        return len(self._indexed_params)

    @staticmethod
    def _parse_inputs(inputs):
        """
        Parse inputs
        :param inputs: filename or dictionary with inputs to the binding model
        :return: dictionary with parsed inputs
        """
        if isinstance(inputs, dict):
            agrs = inputs
        elif isinstance(inputs, six.string_types):
            if ".yml" in inputs or ".yaml" in inputs:
                with open(inputs, 'r') as f:
                    args = yaml.load(f)
            else:
                raise RuntimeError('File format not implemented yet. Try .yml or .yaml')
        else:
            raise RuntimeError('inputs must be a dictionary or a file')

        return args

    @abc.abstractmethod
    def add_component_indexed_param(self, comp_id, name, value):
        """
        Add parameter to component
        :param comp_id: id for component
        :param name: name of the parameter
        :param value: real number
        """

    def add_scalar_parameter(self, name, value):
        """
        Add scalar parameter to dict of parameters
        :param name: name of the parameter
        :param value: real number
        """
        self._scalar_params[name] = value

    @abc.abstractmethod
    def add_component(self, comp_id, parameters):
        """
        Add a component to binding model with its corresponding indexed parameters
        :param comp_id: id for the component
        :param parameters: dictionary with parameters
        """

    def list_component_ids(self):
        """
        Returns list of ids for components
        """
        return list(self._components)

    @abc.abstractmethod
    def del_component(self, comp_id):
        """
        Delete component from binding model. All metadata related deleted as well
        :param comp_id: id for component
        """

    @abc.abstractmethod
    def write_to_cadet_input_file(self, filename):
        """
        Append binding model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """

    @abc.abstractmethod
    def q(self, comp_id, c_vars, unfix_params=None, unfix_idx_param=None):
        """
        
        :param comp_id: 
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable 
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """

    @abc.abstractmethod
    def dqdt(self, comp_id, c_vars, q_vars, unfix_params=None, unfix_idx_param=None):
        """

        :param comp_id: 
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """

    @abc.abstractmethod
    def dq_idc_j(self, comp_id_i, comp_id_j, c_vars, q_vars, unfix_params=None, unfix_idx_param=None):
        """

        :param comp_id_i: id for component in the numerator
        :param comp_id_j: id for component in the denominator
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """
        

@BindingModel.register
class SMAModel(BindingModel):

    def __init__(self, inputs):

        # call binding model constructor
        super().__init__()

        self._registered_parameters = set()
        self._registered_parameters.add('kads')
        self._registered_parameters.add('kdes')
        self._registered_parameters.add('upsilon')
        self._registered_parameters.add('sigma')
        self._registered_parameters.add('Lambda')
        self._registered_parameters.add('refq')
        self._registered_parameters.add('refc')

        args = self._parse_inputs(inputs)

        sparams = args.get('scalar parameters')
        if sparams is not None:
            for name, val in sparams.items():
                assert name in self._registered_parameters, "{} is not a SMA parameter".format(name)
                self._scalar_params[name] = val

        dict_comp = args.get('components')
        has_params = False
        if dict_comp is not None:

            if isinstance(dict_comp,list):
                to_loop = dict_comp
            else:
                to_loop = dict_comp.keys()
                has_params = True

            for comp_id in to_loop:
                if comp_id in self._components:
                    msg = "Component {} being overwritten".format(comp_id)
                    logger.warning(msg)
                self._components.add(comp_id)
        else:
            logger.error("SMA no components found")
            raise RuntimeError('SMA model needs to know the components')

        self._indexed_params = pd.DataFrame(index=self._components,
                                            columns=list(self._registered_parameters))

        self._indexed_params.index.name = 'component id'
        self._indexed_params.columns.name = 'parameters'

        if has_params:
            for comp_id, params in dict_comp.items():
                for parameter, value in params.items():
                    assert parameter in self._registered_parameters, "{} is not a SMA parameter".format(parameter)
                    self._indexed_params[parameter][comp_id] = value

    def add_component_indexed_param(self, comp_id, name, value):

        if isinstance(comp_id, list) and isinstance(name, six.string_types) and isinstance(value, list):

            if name not in self._registered_parameters:
                raise RuntimeError("{} is not a SMA parameter".format(name))

            if len(comp_id) != len(value):
                raise RuntimeError("The arrays must be equal size")

            for i,cid in enumerate(comp_id):
                if cid not in self._components:
                    raise RuntimeError("{} not a component".format(cid))
                self._indexed_params[name][cid] = value[i]










    
    





