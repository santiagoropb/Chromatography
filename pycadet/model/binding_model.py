from __future__ import print_function
import abc
import pandas as pd
import logging
import yaml
import six

logger = logging.getLogger(__name__)


class BindingModel(abc.ABC):

    def __init__(self, inputs):

        # parse inputs
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

        # extract scalar params
        self._scalar_params = dict()
        sparams = args.get('scalar parameters')
        if sparams is not None:
            for name, val in sparams.items():
                self._scalar_params[name] = val

        # extract components and component indexed parameters
        self._components = list()

        # dict_comp = args.get('components')

        # component_idx_param_names = []
        # if dict_comp is not None:
        #    for

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

        self._registered_parameters = set()
        self._registered_parameters.add('kads')
        self._registered_parameters.add('kdes')
        self._registered_parameters.add('upsilon')
        self._registered_parameters.add('sigma')
        self._registered_parameters.add('Lambda')




    
    





