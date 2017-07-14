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

        self._check_model()

        if not self.is_fully_specified():
            print(self.get_index_parameters())
            raise RuntimeError("Missing parameters")

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
        q_ref = kwargs.pop('q_ref', 1.0)

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

        _scalar_params = self.get_scalar_parameters(with_defaults=True)

        assert self._model().salt is not None, "Salt must be defined in chromatography model"

        # get list components
        components = self._model().list_components()
        # determine if salt
        is_salt = self._model().is_salt(comp_name)
        salt_name = self._model().salt

        loop_nosalt = [n for n in components if n != salt_name]
        q_0 = _scalar_params['sma_lambda']
        for cname in loop_nosalt:
            vj = self.nu(cname)
            q_0 -= vj * q_vars[cname]

        if is_salt:
            return q_0

        q_0_bar = q_0
        for cname in loop_nosalt:
            sj = self.sigma(cname)
            q_0_bar -= sj*q_vars[cname]

        # scale q_0_bar
        #gamma_0_bar = q_0_bar*q_ref
        gamma_0_bar = q_0_bar
        # adsorption term
        kads = self.kads(comp_name)
        vi = self.nu(comp_name)
        adsorption = kads * c_vars[comp_name] * (gamma_0_bar) ** vi

        # desorption term
        kdes = self.kdes(comp_name)
        desorption = kdes * q_vars[comp_name] * (c_vars[salt_name]) ** vi

        return adsorption-desorption

    def f_ads_given_free_sites(self, comp_name, c_vars, q_vars, free_sites_var, **kwargs):
        """
        Computes adsorption function for component comp_id
        :param comp_name: name of component of interest
        :param c_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param q_vars: dictionary from comp_id to variable. Either value or pyomo variable
        :param unfix_params: dictionary from parameter name to variable. Either value or pyomo variable
        :return: expression if pyomo variable or scalar value
        """

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

        _scalar_params = self.get_scalar_parameters(with_defaults=True)

        assert self._model().salt is not None, "Salt must be defined in chromatography model"

        # get list components
        components = self._model().list_components()
        # determine if salt
        is_salt = self._model().is_salt(comp_name)
        salt_name = self._model().salt

        if comp_name == 'free_sites' or is_salt:
            loop_nosalt = [n for n in components if n != salt_name]
            q_0 = _scalar_params['sma_lambda']
            for cname in loop_nosalt:
                vj = self.nu(cname)
                q_0 -= vj * q_vars[cname]

            if is_salt:
                return q_0

            q_0_bar = q_vars[salt_name]
            for cname in loop_nosalt:
                sj = self.sigma(cname)
                q_0_bar -= sj * q_vars[cname]

            if comp_name == 'free_sites':
                return q_0_bar

        # scale q_0_bar
        gamma_0_bar = free_sites_var

        # adsorption term
        kads = self.kads(comp_name)
        vi = self.nu(comp_name)
        adsorption = kads * c_vars[comp_name] * gamma_0_bar ** vi

        # desorption term
        kdes = self.kdes(comp_name)
        desorption = kdes * q_vars[comp_name] * (c_vars[salt_name]) ** vi

        return adsorption - desorption

    def _write_to_cadet_input_file(self, filename, unitname, **kwargs):

        super()._write_to_cadet_input_file(filename, unitname, **kwargs)

        assert self._model().salt is not None, "Salt must be defined in chromatography model"

        with h5py.File(filename, 'a') as f:
            subgroup_name = os.path.join("input", "model", unitname)
            subgroup = f[subgroup_name]
            adsorption = subgroup['adsorption']

            # scalar parameters
            double_scalars = ['sma_cref', 'sma_qref']
            int_scalars = ['sma_lambda']

            # index parameters
            double_index = ['sma_ka', 'sma_kd', 'sma_nu', 'sma_sigma']
            self._cadet_writer_helper(adsorption,
                                      int_scalars=int_scalars,
                                      double_scalars=double_scalars,
                                      double_index=double_index)

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





    
    





