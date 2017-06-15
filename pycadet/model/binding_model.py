from __future__ import print_function
from pycadet.model.registrar import Registrar
import pandas as pd
import numpy as np
import warnings
import logging
import h5py
import abc
import os

logger = logging.getLogger(__name__)


class BindingModel(abc.ABC):

    def __init__(self, *args, **kwargs):

        self._registered_scalar_parameters = set()
        self._registered_sindex_parameters = set()

        self._scalar_params = kwargs.pop('sparams', None)
        self._sindex_params = kwargs.pop('iparamas', None)

        # link to model
        self._model = None

        # define type of model
        self._is_kinetic = True

        # define unit name
        self._unit_name = 'unit_001' # by default is unit1 which is the column

    @property
    def is_kinetic(self):
        return self._is_kinetic

    @is_kinetic.setter
    def is_kinetic(self, value):
        self._is_kinetic = value

    @abc.abstractmethod
    def write_to_cadet_input_file(self, filename, unitname):
        """
        Append binding model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """

    @abc.abstractmethod
    def f_ads(self, comp_id, c_vars, q_vars, scale_vars=False, unfix_params=None, unfix_idx_param=None):
        """
        
        :param comp_id: 
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """

    def _set_params(self):

        if self._scalar_params is not None:
            if not isinstance(self._scalar_params, dict):
                raise RuntimeError('Scalar parameters need to be a dictionary')
            for k in self._scalar_params.keys():
                if k not in self._registered_scalar_parameters:
                    msg = """Scalar parameter {} not recognize in 
                                            binding model {}""".format(k, self.__class__.__name__)
                    raise RuntimeError(msg)
        else:
            self._scalar_params = self._model._scalar_params

        if self._sindex_params is not None:
            if not isinstance(self._sindex_params, pd.DataFrame):
                raise RuntimeError('Index parameters need to be a pandas dataframe')
            # check if names are in model
            for n in self._sindex_params.columns:
                if n not in self._registered_sindex_parameters:
                    msg = """Index parameter {} not recognize in 
                    binding model {}""".format(n, self.__class__.__name__)
                    raise RuntimeError(msg)

            #TODO: verify sindex

            as_list = sorted(self._sindex_params.index.tolist())
            for i in range(len(as_list)):
                idx = as_list.index(i)
                as_list[i] = self._comp_id_to_name[idx]
            self._sindex_params.index = as_list

        else:
            self._scalar_params = self._model._sindex_params
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

    def get_scalar_param(self, name):
        if name not in self._registered_scalar_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        return self._scalar_params[name]

    def get_index_param(self, comp_name, name):
        cid = self._model.get_component_id(comp_name)
        if name not in self._registered_sindex_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        return self._sindex_params.get_value(cid, name)

    def set_scalar_param(self, name, value):
        if name not in self._registered_scalar_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        self._scalar_params[name] = value

    def set_index_param(self, comp_name, name, value):
        cid = self._model.get_component_id(comp_name)
        if name not in self._registered_sindex_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        self._sindex_params.set_value(cid, name, value)

@BindingModel.register
class SMABinding(BindingModel):

    def __init__(self, *args, **kwargs):

        # call parent binding model constructor
        super().__init__(*args, **kwargs)

        self._registered_scalar_parameters = \
            Registrar.adsorption_parameters['sma']['scalar']
        self._registered_sindex_parameters = \
            Registrar.adsorption_parameters['sma']['index']

    def f_ads(self, comp_id, c_vars, q_vars, scale_vars=False, unfix_params=None, unfix_idx_param=None):
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

        if scale_vars:
            raise NotImplementedError()

        if not self.is_fully_specified():
            raise RuntimeError("Missing parameters")

        if self.is_salt(comp_id):
            q_0 = self._scalar_params['lambda']
            for cj in self._components:
                if not self.is_salt(cj):
                    vj = self._index_params.get_value(cj, 'upsilon')
                    q_0 -= vj * q_vars[cj]
            return q_0
        else:
            q_0_bar = self._scalar_params['lambda']
            for cj in self._components:
                if not self.is_salt(cj):
                    vj = self._index_params.get_value(cj, 'upsilon')
                    sj = self._index_params.get_value(cj, 'sigma')
                    q_0_bar -= (vj+sj)*q_vars[cj]

            # adsorption term
            kads = self._index_params.get_value(comp_id, 'kads')
            vi = self._index_params.get_value(comp_id, 'upsilon')
            q_rf_salt = self._scalar_params['sma_qref']
            adsorption = kads * c_vars[comp_id] * (q_0_bar / q_rf_salt) ** vi

            # desorption term
            kdes = self._index_params.get_value(comp_id, 'kdes')
            c_rf_salt = self._scalar_params['sma_cref']
            desorption = kdes * q_vars[comp_id] * (c_vars[self.salt_id] / c_rf_salt) ** vi

            return adsorption-desorption

    def is_fully_specified(self):
        has_nan = self._index_params.isnull().values.any()
        for k in self._registered_scalar_parameters:
            if k not in self._scalar_params:
                return False
        return not has_nan

    def write_to_cadet_input_file(self, filename, unitname):

        if not self.is_fully_specified():
            print(self._index_params)
            print(self._scalar_params)
            raise RuntimeError("Missing parameters")

        with h5py.File(filename, 'a') as f:
            subgroup_name = os.path.join("input", "model", unitname)

            if subgroup_name not in f:
                f.create_group(subgroup_name)
            subgroup = f[subgroup_name]
            if 'adsorption' in subgroup:
                warnings.warn("Overwriting {}/{}".format(subgroup_name,'adsorption'))
                adsorption = subgroup['adsorption']
            else:
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
            pointer = np.array(self._scalar_params['lambda'], dtype='d')
            adsorption.create_dataset('SMA_LAMBDA',
                                      data=pointer,
                                      dtype='i')







# TODO: check no nans
# TODO: str method
# TODO: get methods for indexed parameters





    
    





