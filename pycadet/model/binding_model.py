from __future__ import print_function
from pycadet.model.registrar import Registrar
from enum import Enum
import numpy as np
import warnings
import weakref
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


class BindingModel(abc.ABC):

    def __init__(self, *args, **kwargs):

        self._registered_scalar_parameters = set()
        self._registered_sindex_parameters = set()

        # link to model
        self._model = None

        # define type of model
        self._is_kinetic = True

        # define binding type
        self._binding_type = BindingType.UNDEFINED

        # set name
        self._name = None

    @property
    def is_kinetic(self):
        self._check_model()
        return self._is_kinetic

    @is_kinetic.setter
    def is_kinetic(self, value):
        self._check_model()
        self._is_kinetic = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @abc.abstractmethod
    def write_to_cadet_input_file(self, filename, unitname, **kwargs):
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
                     call binding.attach_to_model(m, name) to fix the problem"""
            raise RuntimeError(msg)

    def attach_to_model(self, m, name):
        """
        Attach binding model to Chromatography model
        :param m: Chromatography model to attach binding
        :param name: name of binding model
        :return: None
        """
        if self._model is not None:
            warnings.warn("Reseting Chromatography model")
        setattr(m, name, self)

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
        scalar_params = kwargs.pop('scalar_params', None)
        index_params = kwargs.pop('index_params', None)

        if unfixed_index_params is not None or unfixed_scalar_params is not None:
            raise NotImplementedError()

        if scale_vars is not None:
            raise NotImplementedError()

        if not self.is_fully_specified():
            print(self.get_index_parameters())
            raise RuntimeError("Missing parameters")

        if scalar_params is not None or index_params is not None:
            raise NotImplementedError()

        if scalar_params is not None or index_params is not None:
            raise NotImplementedError()

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
            kads = _index_params.get_value(comp_name, 'sma_kads')
            vi = _index_params.get_value(comp_name, 'sma_nu')
            q_rf_salt = _scalar_params['sma_qref']
            adsorption = kads * c_vars[comp_name] * (q_0_bar / q_rf_salt) ** vi

            # desorption term
            kdes = _index_params.get_value(comp_name, 'sma_kdes')
            c_rf_salt = _scalar_params['sma_cref']
            desorption = kdes * q_vars[comp_name] * (c_vars[salt_name] / c_rf_salt) ** vi

            return adsorption-desorption

    def write_to_cadet_input_file(self, filename, unitname, **kwargs):

        self._check_model()

        scalar_params = kwargs.pop('scalar_params', None)
        index_params = kwargs.pop('index_params', None)

        assert self._model().salt is not None, "Salt must be defined in chromatography model"

        if scalar_params is not None or index_params is not None:
            raise NotImplementedError()

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
            params = {'sma_kads',
                      'sma_kdes',
                      'sma_nu',
                      'sma_sigma'}

            salt_id = self._model()._salt_id
            list_ids = self._model().list_components(ids=True)
            num_components = self._model().num_components
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







# TODO: check no nans
# TODO: str method
# TODO: get methods for indexed parameters





    
    





