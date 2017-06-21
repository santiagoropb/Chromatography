from __future__ import print_function
from pycadet.utils import parse_utils
from pycadet.utils import pandas_utils as pd_utils
import numpy as np
import warnings
import pandas as pd
import logging
import copy
import six

logger = logging.getLogger(__name__)


class DataManager(object):

    def __init__(self, components=None, data=None, **kwargs):

        # define parameters
        self._registered_scalar_parameters = set()
        self._registered_index_parameters = set()

        # define default values for indexed parameters
        self._default_scalar_params = dict()
        self._default_index_params = dict()

        self._scalar_params = dict()
        self._index_params = pd.DataFrame()

        # keep track of components passed
        self._passed_components = components

        # set data
        self._inputs = data
        if data is not None:
            self._inputs = self._parse_inputs(data)

        # link to model
        self._model = None

        # define type of model
        self._is_kinetic = True

        self._components = set()

        # set name
        self._name = None

        #super().__init__()

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

            self._index_params = \
                parse_utils.parse_parameters_indexed_by_components(dict_inputs,
                                                                   map_id,
                                                                   registered_inputs,
                                                                   default_inputs)
            if self._passed_components is not None:
                sub_list = list()
                for name in self._passed_components:
                    if self._model().is_model_component(name):
                        cid = self._model().get_component_id(name)
                        sub_list.append(cid)
                    else:
                        msg = """ {} is not a component of the
                        chromatography model. Ignored""".format(name)
                        raise warnings.warn(msg)
                for cid in sub_list:
                    if cid not in self._index_params.index:
                        self.add_component(self._model()._comp_id_to_name[cid])

            for cid in self._index_params.index:
                self._components.add(cid)

        else:
            if self._passed_components is not None:
                for name in self._passed_components:
                    self.add_component(name)

    def add_component(self, name, parameters=None):

        if not self._model().is_model_component(name):
            msg = """{} is not a component of the 
            chromatography model""".format(name)
            raise RuntimeError(msg)

        cid = self._model().get_component_id(name)
        if cid not in self._components:
            self._index_params = pd_utils.add_row_to_df(self._index_params,
                                                        cid,
                                                        parameters=parameters)
            self._components.add(cid)

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

        for k, v in self._default_scalar_params.items():
            if np.isnan(self._scalar_params[k]):
                self._scalar_params[k] = v

        for k, v in self._default_index_params.items():
            for cid in self._index_params.index:
                if np.isnan(self._index_params.get_value(cid, k)):
                    self._index_params.set_value(cid, k, v)

        if self._index_params.empty:
            self._index_params = pd.DataFrame(index=[],
                                              columns=self._registered_index_parameters)

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