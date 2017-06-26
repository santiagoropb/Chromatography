from __future__ import print_function
from pychrom.core.registrar import Registrar
from pychrom.core.data_manager import DataManager
from enum import Enum
import pandas as pd
import numpy as np
import warnings
import logging
import h5py
import abc
import os

logger = logging.getLogger(__name__)


class BindingType(Enum):

    STERIC_MASS_ACTION = 0
    UNDEFINED = 1

    def __str__(self):
        return "{}".format(self.name)


class BindingModel(DataManager, abc.ABC):

    def __init__(self, data=None, **kwargs):

        # define type of model
        self._is_kinetic = True

        # define binding type
        self._binding_type = BindingType.UNDEFINED
        super().__init__(data=data, **kwargs)

    @property
    def is_kinetic(self):
        self._check_model()
        return self._is_kinetic

    @is_kinetic.setter
    def is_kinetic(self, value):
        self._check_model()
        self._is_kinetic = value

    @property
    def binding_type(self):
        return self._binding_type

    @abc.abstractmethod
    def _write_to_cadet_input_file(self, filename, unitname, **kwargs):
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

    def pprint(self, indent=0):
        t = '\t' * indent
        print(t, self.name, ":\n")
        print(t, "binding type:", self._binding_type)
        super().pprint(indent=indent)


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


@BindingModel.register
class SMABinding(BindingModel):

    def __init__(self, data=None, **kwargs):

        # call parent binding model constructor
        super().__init__(data=data, **kwargs)

        self._registered_scalar_parameters = \
            Registrar.adsorption_parameters['sma']['scalar']

        self._registered_index_parameters = \
            Registrar.adsorption_parameters['sma']['index']

        # set defaults
        self._default_scalar_params = \
            Registrar.adsorption_parameters['sma']['scalar def']

        self._default_index_params = \
            Registrar.adsorption_parameters['sma']['index def']

        # reset index params container
        self._index_params = pd.DataFrame(index=[],
                                          columns=self._registered_index_parameters)

        self._binding_type = BindingType.STERIC_MASS_ACTION

    def f_ads(self, comp_name, c_vars, q_vars, **kwargs):
        """
        Computes adsorption function for component comp_id
        :param comp_name: name of component of interest
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """
        self._check_model()

        unfixed_index_params = kwargs.pop('unfixed_index_params', None)
        unfixed_scalar_params = kwargs.pop('unfixed_scalar_params', None)
        scale_vars = kwargs.pop('scale_vars', None)

        if unfixed_index_params is not None or unfixed_scalar_params is not None:
            raise NotImplementedError()

        if scale_vars is not None:
            raise NotImplementedError()

        if not self.is_fully_specified():
            print(self.get_index_parameters())
            raise RuntimeError("Missing parameters")

        _index_params = self.get_index_parameters()
        _scalar_params = self.get_scalar_parameters(with_defaults=True)

        assert self._model().salt is not None, "Salt must be defined in chromatography model"

        # get list components
        components = self._model().list_components()
        # determine if salt
        is_salt = self._model().is_salt(comp_name)
        salt_name = self._model().salt

        if is_salt:
            q_0 = _scalar_params['sma_lambda']
            for cname in components:
                if self._model().is_salt(comp_name):
                    vj = _index_params.get_value(cname, 'sma_nu')
                    q_0 -= vj * q_vars[cname]
            return q_0
        else:
            q_0_bar = _scalar_params['sma_lambda']
            for cname in components:
                if self._model().is_salt(comp_name):
                    vj = _index_params.get_value(cname, 'sma_nu')
                    sj = _index_params.get_value(cname, 'sma_sigma')
                    q_0_bar -= (vj+sj)*q_vars[cname]

            # adsorption term
            kads = _index_params.get_value(comp_name, 'sma_ka')
            vi = _index_params.get_value(comp_name, 'sma_nu')
            q_rf_salt = _scalar_params['sma_qref']
            adsorption = kads * c_vars[comp_name] * (q_0_bar / q_rf_salt) ** vi

            # desorption term
            kdes = _index_params.get_value(comp_name, 'sma_kd')
            c_rf_salt = _scalar_params['sma_cref']
            desorption = kdes * q_vars[comp_name] * (c_vars[salt_name] / c_rf_salt) ** vi

            return adsorption-desorption

    def _write_to_cadet_input_file(self, filename, unitname, **kwargs):

        self._check_model()

        assert self._model().salt is not None, "Salt must be defined in chromatography model"

        if not self.is_fully_specified():
            print(self.get_index_parameters())
            raise RuntimeError("Missing parameters")

        _index_params = self.get_index_parameters(ids=True)
        _scalar_params = self.get_scalar_parameters(with_defaults=True)

        with h5py.File(filename, 'a') as f:
            subgroup_name = os.path.join("input", "model", unitname)

            if subgroup_name not in f:
                f.create_group(subgroup_name)
            subgroup = f[subgroup_name]
            if 'adsorption' in subgroup:
                warnings.warn("Overwriting {}/{}".format(subgroup_name, 'adsorption'))
                adsorption = subgroup['adsorption']
            else:
                adsorption = subgroup.create_group('adsorption')

            pointer = np.array(self.is_kinetic, dtype='i')
            adsorption.create_dataset('IS_KINETIC',
                                      data=pointer,
                                      dtype='i')

            # adsorption ks
            params = {'sma_ka',
                      'sma_kd',
                      'sma_nu',
                      'sma_sigma'}

            list_ids = self.list_components(ids=True)
            num_components = self.num_components
            for k in params:
                cadet_name = k.upper()
                param = _index_params[k]
                pointer = np.zeros(num_components, dtype='d')
                for i in list_ids:
                    ordered_id = self._model()._ordered_ids_for_cadet.index(i)
                    pointer[ordered_id] = param[i]

                adsorption.create_dataset(cadet_name,
                                          data=pointer,
                                          dtype='d')

            # scalar params
            param_name = 'sma_lambda'
            pointer = np.array(_scalar_params[param_name], dtype='d')
            adsorption.create_dataset(param_name.upper(),
                                      data=pointer,
                                      dtype='i')

            # TODO: this can be moved to based class if some additional
            # TODO: sets are added

    def kads(self, comp_name):
        return self.get_index_parameter(comp_name, 'sma_ka')

    def kdes(self, comp_name):
        return self.get_index_parameter(comp_name, 'sma_kd')

    def nu(self, comp_name):
        return self.get_index_parameter(comp_name, 'sma_nu')

    def sigma(self, comp_name):
        return self.get_index_parameter(comp_name, 'sma_sigma')

    def set_kads(self, comp_name, value):
        self.set_index_parameter(comp_name, 'sma_ka', value)

    def set_kdes(self, comp_name, value):
        self.set_index_parameter(comp_name, 'sma_kd', value)

    def set_nu(self, comp_name, value):
        self.set_index_parameter(comp_name, 'sma_nu', value)

    def set_sigma(self, comp_name, value):
        self.set_index_parameter(comp_name, 'sma_sigma', value)

    @property
    def lamda(self):
        return self.get_scalar_parameter('sma_lambda')

    @lamda.setter
    def lamda(self, value):
        self.set_scalar_parameter('sma_lambda', value)






# TODO: check no nans
# TODO: str method
# TODO: get methods for indexed parameters





    
    





