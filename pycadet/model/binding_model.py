from __future__ import print_function
from pycadet.model.registrar import Registrar
from pycadet.utils import parse_utils
from pycadet.utils import pandas_utils as pd_utils
from enum import Enum
import numpy as np
import warnings
import weakref
import pandas as pd
import logging
import h5py
import copy
import abc
import os
import six

logger = logging.getLogger(__name__)


class BindingType(Enum):

    STERIC_MASS_ACTION = 0
    UNDEFINED = 1

    def __str__(self):
        return "{}".format(self.name)


class BindingModel(abc.ABC):

    def __init__(self, data=None, **kwargs):

        # define parameters
        self._registered_scalar_parameters = set()
        self._registered_index_parameters = set()

        # define default values for indexed parameters
        self._default_scalar_params = dict()
        self._default_index_params = dict()

        self._scalar_params = dict()
        self._index_params = pd.DataFrame()

        # set data
        self._inputs = data
        if data is not None:
            self._inputs = self._parse_inputs(data)

        # link to model
        self._model = None

        # define type of model
        self._is_kinetic = True

        self._components = set()

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
    def binding_type(self):
        return self._binding_type

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

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
        return len(self.get_scalar_parameters(with_defaults=False))

    @property
    def num_index_parameters(self):
        """
        Returns number of indexed parameters (ignore default columns)
        """
        defaults = {k for k in self._default_index_params.keys()}

        df = self._index_params.drop(sorted(defaults), axis=1)
        df.dropna(axis=1, how='all', inplace=True)

        return len(df.columns)

    @staticmethod
    def _parse_inputs(inputs):
        """
        Parse inputs
        :param inputs: filename or dictionary with inputs to the binding model
        :return: dictionary with parsed inputs
        """
        return parse_utils.parse_inputs(inputs)

    def _parse_scalar_parameters(self):

        if self._inputs is not None:
            sparams = self._inputs.get('scalar parameters')
            registered_inputs = self._registered_scalar_parameters
            parsed = parse_utils.parse_scalar_inputs_from_dict(sparams,
                                                               self.__class__.__name__,
                                                               registered_inputs,
                                                               logger)

            for k, v in parsed.items():
                self._scalar_params[k] = v
        else:
            msg = """ No inputs when _parse_scalar_parameters
            was called in {}""".format(self.__class__.__name__)
            logger.debug(msg)

    def _parse_index_parameters(self):

        if self._inputs is not None:
            map_id = self._model()._comp_name_to_id
            registered_inputs = self._registered_index_parameters
            default_inputs = self._default_index_params
            dict_inputs = self._inputs.get('index parameters')
            self._index_params = parse_utils.parse_parameters_indexed_by_components(dict_inputs,
                                                                                    map_id,
                                                                                    registered_inputs,
                                                                                    default_inputs)

            for cid in self._index_params.index:
                self._components.add(cid)

        else:
            msg = """ No inputs when _parse_index_parameters
                    was called in {}""".format(self.__class__.__name__)
            logger.debug(msg)

    def add_component(self, name, parameters=None):

        if not self._model().is_model_component(name):
            msg = """{} is not a component of the 
            chromatography model"""
            raise RuntimeError(msg)

        cid = self._model().get_component_id(name)
        if cid not in self._components:
            self._index_params = pd_utils.add_row_to_df(self._index_params,
                                                        cid,
                                                        parameters=parameters)

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
        """

        :param name: name of the scalar parameter
        :return: value of scalar parameter
        """
        return self._scalar_params[name]

    def get_index_parameter(self, comp_name, name):
        self._check_model()
        """

        :param comp_name: name of component
        :param name:  name of index parameter
        :return: value
        """
        cid = self._model().get_component_id(comp_name)
        return self._index_params.get_value(cid, name)

    def set_scalar_parameter(self, name, value):
        self._check_model()
        """
        set scalar parameter to dict of parameters
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

    def set_index_parameter(self, comp_name, name, value):

        self._check_model()
        if name not in self._registered_index_parameters:
            raise RuntimeError('{} is not a parameter of model {}'.format(name, self.__class__.__name__))
        cid = self._model().get_component_id(comp_name)
        pd_utils.set_value_in_df(self._index_params,
                                 cid,
                                 name,
                                 value)

    def get_scalar_parameters(self, with_defaults=False):
        """

        :param with_defaults: flag indicating if default parameters must be included
        :return: dictionary of scalar parameters
        """
        if with_defaults:
            return copy.deepcopy(self._scalar_params)
        container = dict()
        for n, v in self._scalar_params.items():
            if n not in self._default_scalar_params.keys():
                container[n] = v
        return container

    def get_index_parameters(self, with_defaults=False, ids=False, form='dataframe'):
        """

                :param with_defaults: flag indicating if default parameters must be included
                :return: DataFrame with parameters indexed with components
                """
        if form == 'dataframe':
            if not with_defaults:
                defaults = {k for k in self._default_index_params.keys()}
                df = self._index_params.drop(defaults, axis=1)
                df.dropna(axis=1, how='all', inplace=True)
            else:
                df = self._index_params.dropna(axis=1, how='all')

            if not ids:

                as_list = sorted(df.index.tolist())
                for i in range(len(as_list)):
                    idx = as_list.index(i)
                    as_list[i] = self._model()._comp_id_to_name[idx]
                df.index = as_list

            return df
        elif form == 'dictionary':
            container = dict()
            for cid in self._components:
                if ids:
                    cname = cid
                else:
                    cname = self._model()._comp_id_to_name[cid]
                container[cname] = dict()
                for name in self._registered_index_parameters:
                    if with_defaults:
                        container[cname][name] = self._index_params.get_value(cid, name)
                    else:
                        if name not in self._default_index_params.keys():
                            container[cname][name] = self._index_params.get_value(cid, name)
            return container
        else:
            raise RuntimeError('Form of output not recognized')

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
        self._initialize_containers()

    def is_fully_specified(self):
        self._check_model()
        df = self.get_index_parameters()
        has_nan = df.isnull().values.any()
        for k in self._registered_scalar_parameters:
            if k not in self.get_scalar_parameters(True).keys():
                print("Missing scalar parameter {}".format(k))
                return False

            if not isinstance(self._scalar_params[k], six.string_types):
                if np.isnan(self._scalar_params[k]):
                    print("Parameter {} is nan".format(k))
                    return False
            else:
                if self._scalar_params[k] is None:
                    return False

        return not has_nan

    def _fill_containers(self):

        # initialize containers
        for k in self._registered_scalar_parameters:
            if self._scalar_params.get(k) is None:
                self._scalar_params[k] = np.nan

        if self._inputs is None:
            self._index_params = pd.DataFrame(np.nan,
                                              index=[],
                                              columns=self._registered_index_parameters)

        for k, v in self._default_scalar_params.items():
            if np.isnan(self._scalar_params[k]):
                self._scalar_params[k] = v

    def _initialize_containers(self):

        self._parse_scalar_parameters()
        self._parse_index_parameters()
        self._fill_containers()
        self._inputs = None

    def list_components(self, ids=False):
        if ids:
            return [k for k in self._components]
        else:
            return [self._model()._comp_id_to_name[k] for k in self._components]

    def help(self):
        print("TODO")

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
            params = {'sma_kads',
                      'sma_kdes',
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
        return self.get_index_parameter(comp_name, 'sma_kads')

    def kdes(self, comp_name):
        return self.get_index_parameter(comp_name, 'sma_kdes')

    def nu(self, comp_name):
        return self.get_index_parameter(comp_name, 'sma_nu')

    def sigma(self, comp_name):
        return self.get_index_parameter(comp_name, 'sma_sigma')

    @property
    def lamda(self):
        return self.get_scalar_parameter('sma_lambda')

    @lamda.setter
    def lamda(self, value):
        self.set_scalar_parameter('sma_lambda', value)






# TODO: check no nans
# TODO: str method
# TODO: get methods for indexed parameters





    
    





