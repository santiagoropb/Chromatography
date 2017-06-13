from __future__ import print_function
from enum import Enum
import pandas as pd
import numpy as np
import collections
import warnings
import numbers
import logging
import yaml
import h5py
import copy
import six
import abc
import os

logger = logging.getLogger(__name__)


class ModelType(Enum):
    UNSPECIFIED = 0
    SMA = 1
    

class BindingModel(abc.ABC):

    def __init__(self):

        # define registered params
        self._registered_scalar_parameters = set()
        self._registered_index_parameters = set()

        # define default values for indexed parameters
        self._default_index_params = dict()

        # define scalar params container
        self._scalar_params = dict()

        # define components and component indexed parameters
        self._components = set()
        self._index_params = pd.DataFrame(index=self._components,
                                            columns=self._registered_index_parameters)

        # define type of model
        self._model_type = ModelType.UNSPECIFIED
        self._is_kinetic = True

    @property
    def is_kinetic(self):
        return self._is_kinetic

    @is_kinetic.setter
    def is_kinetic(self, value):
        self._is_kinetic = value

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
        return len(self._index_params)

    @staticmethod
    def _parse_inputs(inputs):
        """
        Parse inputs
        :param inputs: filename or dictionary with inputs to the binding model
        :return: dictionary with parsed inputs
        """
        if isinstance(inputs, dict):
            args = inputs
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
    def write_to_cadet_input_file(self, filename, unitname):
        """
        Append binding model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """

    @abc.abstractmethod
    def f_ads(self, comp_id, c_vars, q_vars, unfix_params=None, unfix_idx_param=None):
        """
        
        :param comp_id: 
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """


    #TODO: define if this is actually required
    #@abc.abstractmethod
    #def dqdt(self, comp_id, c_vars, q_vars, unfix_params=None, unfix_idx_param=None):
    #    """
    #
    #    :param comp_id:
    #    :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
    #    :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
    #    :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
    #    :return: expression if pyomo variable or scalar value
    #    """

    #@abc.abstractmethod
    #def dq_idc_j(self, comp_id_i, comp_id_j, c_vars, q_vars, unfix_params=None, unfix_idx_param=None):
    #    """
    #
    #    :param comp_id_i: id for component in the numerator
    #    :param comp_id_j: id for component in the denominator
    #    :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
    #    :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
    #    :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
    #    :return: expression if pyomo variable or scalar value
    #    """

    def _parse_scalar_params(self, args):
        """
        Parse scalar parameters
        :param args: distionary with parsed inputs
        :return: None
        """

        sparams = args.get('scalar parameters')
        if sparams is not None:
            for name, val in sparams.items():
                msg = """{} is not a scalar parameter 
                of model {}""".format(name, self._model_type)
                assert name in self._registered_scalar_parameters, msg
                self._scalar_params[name] = val
        else:
            msg = """No scalar parameters specified 
            when parsing {}""".format(self._model_type)
            logger.debug(msg)

    def _parse_components(self, args):
        """
        Parse components and indexed parameters
        :param args: dictionary with parsed inputs
        :return: None
        """
        dict_comp = args.get('components')
        has_params = False
        if dict_comp is not None:

            if isinstance(dict_comp, list):
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

        if len(self._components) == 0:
            msg = """No components specified 
            when parsing {}""".format(self._model_type)
            logger.debug(msg)

        self._index_params = pd.DataFrame(index=self._components,
                                            columns=self._registered_index_parameters)

        # set defaults
        for name, default in self._default_index_params:
            self._index_params[name] = default

        self._index_params.index.name = 'component id'
        self._index_params.columns.name = 'parameters'

        if has_params:
            for comp_id, params in dict_comp.items():
                for parameter, value in params.items():
                    msg = """{} is not a parameter
                    of model {}""".format(parameter, self._model_type)
                    assert parameter in self._registered_index_parameters, msg
                    self._index_params[parameter][comp_id] = value
        else:
            msg = """ No indexed parameters 
            specified when parsing {} """.format(self._model_type)
            logger.debug(msg)

    def set_component_indexed_param(self, comp_id, name, value):
        """
        Add parameter to component
        :param comp_id: id for component
        :param name: name of the parameter
        :param value: real number
        """

        if (isinstance(comp_id, list) or isinstance(comp_id, tuple)) and \
                (isinstance(value, list) or isinstance(value, tuple)) and \
                isinstance(name, six.string_types):

            if name not in self._registered_index_parameters:
                msg = """{} is not a parameter 
                of model {}""".format(name, self._model_type)
                raise RuntimeError(msg)

            if len(comp_id) != len(value):
                raise RuntimeError("The arrays must be equal size")

            for i, cid in enumerate(comp_id):
                if cid not in self._components:
                    raise RuntimeError("{} is not a component".format(cid))
                self._index_params[name][cid] = value[i]

        elif (isinstance(value, list) or isinstance(value, tuple)) and \
                (isinstance(name, list) or isinstance(name, tuple)) and \
                isinstance(comp_id, numbers.Integral):

            if comp_id not in self._components:
                raise RuntimeError("{} is not a component".format(comp_id))

            for i, n in enumerate(name):
                if n not in self._registered_index_parameters:
                    msg = """{} is not a parameter 
                    of model {}""".format(name, self._model_type)
                    raise RuntimeError(msg)
                self._index_params[n][comp_id] = value[i]

        elif isinstance(comp_id, numbers.Integral) and \
                isinstance(name, six.string_types) and \
                (isinstance(value, six.string_types) or isinstance(value, numbers.Number)):

            if comp_id not in self._components:
                raise RuntimeError("{} is not a component".format(comp_id))
            self._index_params[name][comp_id] = value

        else:
            raise RuntimeError("input not recognized")

    def add_scalar_parameter(self, name, value):
        """
        Add scalar parameter to dict of parameters
        :param name: name(s) of the parameter
        :param value: real number(s)
        :return: None
        """
        if isinstance(name, six.string_types):
            assert name in self._registered_scalar_parameters
            self._scalar_params[name] = value
        elif (isinstance(name, list) or isinstance(name, tuple)) and \
                (isinstance(value, list) or isinstance(value, tuple)):

            for i, n in enumerate(name):
                assert name in self._registered_scalar_parameters
                self._scalar_params[n] = value[i]
        else:
            raise RuntimeError("input not recognized")

    def set_scalar_parameter(self, name, value):

        # TODO: rethink this
        self.add_scalar_parameter(name, value)

    def add_component(self, comp_id, parameters=None):
        """
        Add a component to binding model with its corresponding indexed parameters
        :param comp_id: id for the component
        :param parameters: dictionary with parameters
        """
        if parameters is None:
            tmp_list = set()
            if isinstance(comp_id, collections.Sequence):
                for cid in comp_id:
                    if cid not in self._components:
                        tmp_list.add(cid)
                    else:
                        msg = """ignoring component {}.
                        The component was already added""".format(cid)
                        logger.warning(msg)
                self._index_params = \
                    self._index_params.reindex(self._index_params.index.union(tmp_list))

                # set defaults
                for name, default in self._default_index_params:
                    self._index_params[name] = default

                self._components.update(tmp_list)
            else:
                if comp_id not in self._components:
                    tmp_list = {comp_id}
                    self._index_params = \
                        self._index_params.reindex(self._index_params.index.union(tmp_list))

                    # set defaults
                    for name, default in self._default_index_params:
                        self._index_params[name] = default

                    self._components.update(tmp_list)
                else:
                    msg = """ignoring component {}.
                    The component was already added""".format(comp_id)
                    logger.warning(msg)
        else:

            if isinstance(comp_id, collections.Sequence) and \
                    isinstance(parameters, collections.Sequence):

                assert len(comp_id) == len(parameters)
                overwritten = set()
                for i, cid in enumerate(comp_id):
                    if not isinstance(parameters[i], dict):
                        msg = """Parameters per component need to
                        be provided in a dictionary"""
                        raise RuntimeError(msg)
                    if cid in self._components:
                        overwritten.add(cid)

                not_overwritten = self._components.difference(overwritten)
                self._index_params = \
                    self._index_params.reindex(self._index_params.index.union(not_overwritten))

                # set defaults
                for name, default in self._default_index_params:
                    self._index_params[name] = default

                self._components.update(not_overwritten)

                for i, cid in enumerate(comp_id):
                    params = parameters[i]
                    for name, value in params.items():
                        if name not in self._registered_index_parameters:
                            msg = """"{} is not a parameter 
                            of model {}""".format(name, self._model_type)
                            raise RuntimeError(msg)
                        self._index_params[name][cid] = value

                if overwritten:
                    for n in overwritten:
                        msg = """Parameters of component {}.
                                were overwritten""".format(n)
                        warnings.warn(msg)

            elif isinstance(comp_id, numbers.Integral) and \
                    isinstance(parameters, dict):

                if comp_id not in self._components:
                    to_add = {comp_id}
                    self._index_params = \
                        self._index_params.reindex(self._index_params.index.union(to_add))

                    # set defaults
                    for name, default in self._default_index_params:
                        self._index_params[name] = default

                    self._components.update(to_add)
                else:
                    msg = """Parameters of component {}.
                            were overwritten""".format(comp_id)
                    warnings.warn(msg)

                for name, value in parameters.items():
                    if name not in self._registered_index_parameters:
                        msg = """"{} is not a parameter 
                                of model {}""".format(name, self._model_type)
                        raise RuntimeError(msg)
                    self._index_params[name][comp_id] = value

            else:
                raise RuntimeError("input not recognized")

    def del_component(self, comp_id):
        """
        Removes component form binding model
        :param comp_id: component id
        :return: None
        """
        self._components.remove(comp_id)
        self._index_params.drop(comp_id)

    def list_component_ids(self):
        """
        Returns list of ids for components
        """
        return list(self._components)

    def get_scalar_parameters(self):
        return copy.deepcopy(self._scalar_params)

    def scalar_parameters(self):
        for n, v in self._scalar_params.items():
            yield n, v

@BindingModel.register
class SMAModel(BindingModel):

    def __init__(self, inputs, comp_id_salt):

        # call parent binding model constructor
        super().__init__()

        # registers scalar paramenters
        self._registered_scalar_parameters.add('Lambda')

        # registers indexed parameters
        self._registered_index_parameters.add('kads')
        self._registered_index_parameters.add('kdes')
        self._registered_index_parameters.add('upsilon')
        self._registered_index_parameters.add('sigma')
        self._registered_index_parameters.add('refq')
        self._registered_index_parameters.add('refc')

        # register defaults
        self._default_index_params['refq'] = 1.0
        self._default_index_params['refc'] = 1.0

        # parse inputs
        args = self._parse_inputs(inputs)
        self._parse_scalar_params(args)
        self._parse_components(args)

        # flags for model specification
        self._model_type = ModelType.SMA
        self._is_kinetic = True

        self._salt_id = comp_id_salt

    def is_salt(self, comp_id):
        assert comp_id in self._components
        return comp_id == self._salt_id

    def f_ads(self, comp_id, c_vars, q_vars, unfix_params=None, unfix_idx_param=None):
        """
        Computes adsorption function for component comp_id
        :param comp_id:
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """
        if unfix_idx_param is not None or unfix_params is not None:
            raise NotImplementedError()

        if not self.is_fully_specified():
            print(self._index_params)
            print(self._scalar_params)
            raise RuntimeError("Missing parameters")

        if self.is_salt(comp_id):
            q_0 = self._scalar_params['Lambda']
            for cj in self._components:
                if not self.is_salt(cj):
                    vj = self._index_params['upsilon'][cj]
                    q_0 -= vj * q_vars[cj]
            return q_0
        else:
            q_0_bar = self._scalar_params['Lambda']
            for cj in self._components:
                if not self.is_salt(cj):
                    vj = self._index_params['upsilon'][cj]
                    sj = self._index_params['sigma'][cj]
                    q_0_bar -= (vj+sj)*q_vars[cj]

            # adsorption term
            kads = self._index_params['kads'][comp_id]
            vi = self._index_params['upsilon'][comp_id]
            q_rf_salt = self._index_params['qref'][self._salt_id]
            adsorption = kads*c_vars[comp_id]*(q_0_bar/q_rf_salt)**vi

            # desorption term
            kdes = self._index_params['kdes'][comp_id]
            vi = self._index_params['upsilon'][comp_id]
            c_rf_salt = self._index_params['cref'][self._salt_id]
            desorption = kdes * q_vars[comp_id] * (c_vars[self._salt_id] / c_rf_salt) ** vi

            return adsorption-desorption

    def is_fully_specified(self):
        has_nan = self._index_params.isnull().values.any()
        return self._scalar_params.has_key('Lambda') and not has_nan

    def write_to_cadet_input_file(self, filename, unitname):

        if not self.is_fully_specified():
            print(self._index_params)
            print(self._scalar_params)
            raise RuntimeError("Missing parameters")

        with h5py.File(filename, 'a') as f:
            subgroup_name = os.path.join(filename, "input", "model", unitname)
            subgroup = f[subgroup_name]
            adsorption = subgroup.create_group('adsorption')

            pointer = np.array(self.is_kinetic, dtype='i')
            adsorption.create_dataset('IS_KINETIC',
                                      data=pointer,
                                      dtype='i')

            # adsorption ks
            params = {'kads': 'SMA_KA',
                      'kdes': 'SMA_KD',
                      'upsilon': 'SMA_NU',
                      'sigma': 'SMA_SIGMA'}

            for k, v in params.items():
                param = list(self._index_params[k])
                if self._salt_id != 0:
                    tmp = param[0]
                    param[0] = param[self._salt_id]
                    param[self._salt_id] = tmp

                pointer = np.array(param, dtype='d')
                adsorption.create_dataset(v,
                                          data=pointer,
                                          dtype='d')

            # scalar params
            pointer = np.array(self._scalar_params['Lambda'], dtype='d')
            adsorption.create_dataset('SMA_LAMBDA',
                                      data=pointer,
                                      dtype='i')







# TODO: check no nans
# TODO: str method
# TODO: get methods for indexed parameters





    
    





