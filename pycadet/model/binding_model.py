from __future__ import print_function
from pycadet.model.registrar import Registrar
import pandas as pd
import numpy as np
import warnings
import weakref
import logging
import h5py
import abc
import os

logger = logging.getLogger(__name__)


class BindingModel(abc.ABC):

    def __init__(self, *args, **kwargs):

        self._registered_scalar_parameters = set()
        self._registered_sindex_parameters = set()

        # link to model
        self._model = None

        # define type of model
        self._is_kinetic = True

        # define unit name
        self._unit_name = 'unit_001' # by default is unit1 which is the column

    @property
    def is_kinetic(self):
        self._check_model()
        return self._is_kinetic

    @is_kinetic.setter
    def is_kinetic(self, value):
        self._check_model()
        self._is_kinetic = value

    @property
    def unit(self):
        return self._unit_name

    @unit.setter
    def unit(self, unit_name):
        self._unit_name = unit_name

    @abc.abstractmethod
    def write_to_cadet_input_file(self, filename, unitname):
        """
        Append binding model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """

    @abc.abstractmethod
    def f_ads(self, comp_name, c_vars, q_vars, **kwargs):
        """
        
        :param comp_name: name of component of interest
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

    def get_scalar_parameter(self, name):
        self._check_model()
        if name not in self._registered_scalar_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        return self._model().get_scalar_parameter(name)

    def get_index_parameter(self, comp_name, name):
        self._check_model()
        if name not in self._registered_sindex_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        return self._model().get_index_parameter(comp_name, name)

    def set_scalar_parameter(self, name, value):
        self._check_model()
        if name not in self._registered_scalar_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        self._model().set_scalar_parameter(name, value)

    def set_index_parameter(self, comp_name, name, value):

        self._check_model()
        if name not in self._registered_sindex_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        self._model().set_index_param(comp_name, name, value)

    def get_scalar_parameters(self, with_defaults=False):
        return self._model().get_scalar_parameters(with_defaults=with_defaults)

    def get_index_parameters(self, ids=False):
        return self._model().get_index_parameters(ids=ids)[list(self._registered_sindex_parameters)]

    def _check_model(self):
        if self._model is None:
            msg = """Binding model not attached to a Chromatography model.
                     When a binding model is created it must be attached to a Chromatography
                     model e.g \\n m = GRModel() \\n m.binding = SMABinding(). Alternatively,
                     call binding.set_model(m) to fix the problem"""
            raise RuntimeError(msg)

    def set_model(self, m):
        """
        Attach binding model to Chromatography model
        :param m: model to attach binding
        :return: None
        """
        if self._model is not None:
            warnings.warn("Reseting Chromatography model")
        self._model = weakref.ref(m)

    def is_fully_specified(self):
        self._check_model()
        df = self.get_index_parameters()
        has_nan = df.isnull().values.any()
        for k in self._registered_scalar_parameters:
            if k not in self.get_scalar_parameters(True).keys():
                print("Missing scalar parameter {}".format(k))
                return False

        return not has_nan



@BindingModel.register
class SMABinding(BindingModel):

    def __init__(self, *args, **kwargs):

        # call parent binding model constructor
        super().__init__(*args, **kwargs)

        self._registered_scalar_parameters = \
            Registrar.adsorption_parameters['sma']['scalar']
        self._registered_sindex_parameters = \
            Registrar.adsorption_parameters['sma']['index']

    def f_ads(self, comp_name, c_vars, q_vars, **kwargs):
        """
        Computes adsorption function for component comp_id
        :param comp_id: name of component of interest
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """
        self._check_model()

        unfixed_index_params = kwargs.pop('unfixed_index_params',None)
        unfixed_scalar_params = kwargs.pop('unfixed_scalar_params',None)
        scale_vars = kwargs.pop('scale_vars', None)
        scalar_params = kwargs.pop('scalar_params', None)
        index_params = kwargs.pop('index_params', None)

        if scalar_params is not None or index_params is not None:
            raise NotImplementedError()

        if unfixed_index_params is not None or unfixed_scalar_params is not None:
            raise NotImplementedError()

        if scale_vars is not None:
            raise NotImplementedError()

        if not self.is_fully_specified():
            raise RuntimeError("Missing parameters")

        _index_params = self.get_index_parameters(ids=True)
        _scalar_params = self.get_scalar_parameters(with_defaults=True)

        assert self._model().salt_name is not None, "Salt must be defined in chromatography model"

        if self._model().is_salt(comp_name):
            q_0 = _scalar_params['sma_lambda']
            for cj in self._model().list_components(ids=True):
                if not self._model().is_salt(cj):
                    vj = _index_params.get_value(cj, 'sma_nu')
                    q_0 -= vj * q_vars[cj]
            return q_0
        else:
            q_0_bar = _scalar_params['sma_lambda']
            for cj in self._components:
                if not self.is_salt(cj):
                    vj = self._index_params.get_value(cj, 'sma_nu')
                    sj = self._index_params.get_value(cj, 'sma_sigma')
                    q_0_bar -= (vj+sj)*q_vars[cj]

            # adsorption term
            comp_id = self._model().get_component_id(comp_name)
            kads = _index_params.get_value(comp_id, 'sma_kads')
            vi = _index_params.get_value(comp_id, 'sma_nu')
            q_rf_salt = _scalar_params['sma_qref']
            adsorption = kads * c_vars[comp_id] * (q_0_bar / q_rf_salt) ** vi

            # desorption term
            kdes =  _index_params.get_value(comp_id, 'sma_kdes')
            c_rf_salt = _scalar_params['sma_cref']
            desorption = kdes * q_vars[comp_id] * (c_vars[self.salt_id] / c_rf_salt) ** vi

            return adsorption-desorption

    def write_to_cadet_input_file(self, filename, unitname):

        self._check_model()

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





    
    





