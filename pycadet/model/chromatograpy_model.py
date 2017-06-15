from __future__ import print_function
from pycadet.model.registrar import Registrar
from pycadet.model.binding_model import BindingModel, SMABinding
from pycadet.utils import parse_utils
from collections import OrderedDict
import pandas as pd
import collections
import warnings
import logging
import numbers
import copy
import six
import abc

logger = logging.getLogger(__name__)


class ChromatographyModel(abc.ABC):

    def __init__(self):

        # define registered params
        self._registered_scalar_parameters = Registrar.scalar_parameters
        self._registered_sindex_parameters = Registrar.single_index_parameters

        # define default values for indexed parameters
        self._default_scalar_params = Registrar.default_scalar_parameters
        self._default_sindex_params = Registrar.default_single_index_parameters

        # define scalar params container
        self._scalar_params = dict()

        # define components and component indexed parameters
        self._comp_name_to_id = dict()
        self._comp_id_to_name = dict()
        self._components = set()
        self._sindex_params = pd.DataFrame(index=self._components,
                                           columns=self._registered_sindex_parameters)
        self._salt_id = -1

        self._binding_models = list()

    @property
    def salt_name(self):
        if self._salt_id != -1:
            return self._comp_id_to_name[self._salt_id]
        return None

    @salt_name.setter
    def salt_name(self, name):
        self._salt_id = self._comp_name_to_id[name]

    def is_salt(self, name):

        if name not in self._comp_name_to_id.keys():
            return False
        if self._comp_name_to_id[name] == self._salt_id:
            return True
        return False

    def get_component_id(self, comp_name):
        """
        Returns unique id of component
        :param comp_name: name of component
        :return: integer id
        """
        if comp_name not in self._comp_name_to_id.keys():
            raise RuntimeError('{} is not a model component'.format(comp_name))
        return self._comp_name_to_id[comp_name]

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
    def num_single_index_parameters(self):
        """
        Returns number of indexed parameters (ignore default columns)
        """
        defaults = {k for k in self._default_sindex_params.keys()}

        df = self._sindex_params.drop(sorted(defaults), axis=1)
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

    def _parse_scalar_params(self, dict_inputs):
        """
        Parse scalar parameters
        :param args: distionary with parsed inputs
        :return: None
        """

        sparams = dict_inputs.get('scalar parameters')
        if sparams is not None:
            for name, val in sparams.items():
                msg = """{} is not a scalar parameter 
                    of model {}""".format(name, self.__class__.__name__)
                assert name in self._registered_scalar_parameters, msg
                self._scalar_params[name] = val
        else:
            msg = """No scalar parameters specified 
                when parsing {}""".format(self.__class__.__name__)
            logger.debug(msg)

        for k, val in self._default_scalar_params.items():
            self._scalar_params[k] = val

    def __add_component(self, name):
        new_id = len(self._components)
        self._comp_name_to_id[name] = new_id
        self._comp_id_to_name[new_id] = name
        self._components.add(new_id)

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
                to_loop = sorted(dict_comp)
            else:
                ordered_dict = OrderedDict(dict_comp)
                to_loop = ordered_dict.keys()
                has_params = True

            for comp_name in to_loop:
                if comp_name in self._comp_name_to_id.keys():
                    msg = "Component {} being overwritten".format(comp_name)
                    logger.warning(msg)
                self.__add_component(comp_name)

        else:
            logger.warning("No components found to parsed")

        if len(self._components) == 0:
            logger.warning("No components found to parsed")

        self._sindex_params = pd.DataFrame(index=self._components,
                                           columns=sorted(self._registered_sindex_parameters))

        # set defaults
        for name, default in self._default_sindex_params.items():
            self._sindex_params[name] = default

        self._sindex_params.index.name = 'component id'
        self._sindex_params.columns.name = 'parameters'

        if has_params:
            for comp_name, params in dict_comp.items():
                comp_id = self._comp_name_to_id[comp_name]
                for parameter, value in params.items():
                    msg = """{} is not a parameter
                    of model {}""".format(parameter, self.__class__.__name__)
                    assert parameter in self._registered_sindex_parameters, msg
                    self._sindex_params.set_value(comp_id, parameter, value)
        else:
            msg = """ No indexed parameters 
            specified when parsing {} """.format(self.__class__.__name__)
            logger.warning(msg)

    def add_component(self, comp_name, parameters=None):
        """
        Add a component to binding model with its corresponding indexed parameters
        :param comp_id: id for the component
        :param parameters: dictionary with parameters
        """
        if parameters is None:
            tmp_list = set()
            if (isinstance(comp_name, list) or isinstance(comp_name, tuple)) and \
                    not isinstance(comp_name, six.string_types):

                for cname in comp_name:
                    if cname not in self._comp_name_to_id.keys():
                        self.__add_component(cname)
                        cid = self._comp_name_to_id[cname]
                        tmp_list.add(cid)
                    else:
                        msg = """ignoring component {}.
                        The component was already added""".format(cname)
                        logger.warning(msg)
                self._sindex_params = \
                    self._sindex_params.reindex(self._sindex_params.index.union(tmp_list))

                # set defaults
                for name, default in self._default_sindex_params.items():
                    self._sindex_params[name] = default

            else:
                if comp_name not in self._comp_name_to_id.keys():
                    self.__add_component(comp_name)
                    comp_id = self._comp_name_to_id[comp_name]
                    tmp_list = {comp_id}
                    self._sindex_params = \
                        self._sindex_params.reindex(self._sindex_params.index.union(tmp_list))

                    # set defaults
                    for name, default in self._default_sindex_params.items():
                        self._sindex_params[name] = default

                else:
                    msg = """ignoring component {}.
                    The component was already added""".format(comp_name)
                    logger.warning(msg)
        else:

            if (isinstance(comp_name, list) or isinstance(comp_name, tuple)) and \
                    not isinstance(comp_name, six.string_types) and \
                    isinstance(parameters, collections.Sequence):

                assert len(comp_name) == len(parameters)

                overwritten_cnames = set()
                for i, cname in enumerate(comp_name):
                    if not isinstance(parameters[i], dict):
                        msg = """Parameters per component need to
                        be provided in a dictionary"""
                        raise RuntimeError(msg)
                    if comp_name in self._comp_name_to_id.keys():
                        overwritten_cnames.add(cname)

                not_overwritten_cnames = set(comp_name).difference(overwritten_cnames)
                for name in not_overwritten_cnames:
                    self.__add_component(name)
                not_overwritten = [self._comp_name_to_id[cname] for cname in not_overwritten_cnames]
                self._sindex_params = \
                    self._sindex_params.reindex(self._sindex_params.index.union(not_overwritten))

                # set defaults
                for name, default in self._default_sindex_params.items():
                    self._sindex_params[name] = default

                for i, cname in enumerate(comp_name):
                    params = parameters[i]
                    cid = self._comp_name_to_id[cname]
                    for name, value in params.items():
                        if name not in self._registered_sindex_parameters:
                            msg = """"{} is not a parameter 
                            of model {}""".format(name, self.__class__.__name__)
                            raise RuntimeError(msg)
                        self._sindex_params.set_value(cid, name, value)

                if overwritten_cnames:
                    for n in overwritten_cnames:
                        msg = """Parameters of component {}.
                                were overwritten""".format(n)
                        warnings.warn(msg)

            elif (isinstance(comp_name, six.string_types) or
                      isinstance(comp_name, numbers.Integral)) and \
                    isinstance(parameters, dict):

                if comp_name not in self._comp_name_to_id.keys():
                    self.__add_component(comp_name)
                    comp_id = self._comp_name_to_id[comp_name]
                    to_add = {comp_id}
                    self._sindex_params = \
                        self._sindex_params.reindex(self._sindex_params.index.union(to_add))

                    # set defaults
                    for name, default in self._default_sindex_params.items():
                        self._sindex_params[name] = default

                    self._components.update(to_add)
                else:
                    comp_id = self._comp_name_to_id[comp_name]
                    msg = """Parameters of component {}.
                            were overwritten""".format(comp_name)
                    warnings.warn(msg)

                for name, value in parameters.items():
                    if name not in self._registered_sindex_parameters:
                        msg = """"{} is not a parameter 
                                of model {}""".format(name, self.__class__.__name__)
                        raise RuntimeError(msg)
                    self._sindex_params.set_value(comp_id, name, value)

            else:
                raise RuntimeError("input not recognized")

    def set_index_param(self, comp_name, name, value):
        """
        Add parameter to component
        :param comp_id: id for component
        :param name: name of the parameter
        :param value: real number
        """

        if (isinstance(comp_name, list) or isinstance(comp_name, tuple)) and \
                (isinstance(value, list) or isinstance(value, tuple)) and \
                isinstance(name, six.string_types) and not isinstance(comp_name, six.string_types):

            if name not in self._registered_sindex_parameters:
                msg = """{} is not a parameter 
                of model {}""".format(name, self.__class__.__name__)
                raise RuntimeError(msg)

            if len(comp_name) != len(value):
                raise RuntimeError("The arrays must be equal size")

            for i, cname in enumerate(comp_name):
                cid = self._comp_name_to_id[cname]
                if cid not in self._components:
                    raise RuntimeError("{} is not a component".format(cid))
                self._sindex_params.set_value(cid, name, value[i])

        elif (isinstance(value, list) or isinstance(value, tuple)) and \
                (isinstance(name, list) or isinstance(name, tuple)) and \
                (isinstance(comp_name, six.string_types) or isinstance(comp_name, numbers.Integral)):


            if comp_name not in self._comp_name_to_id.keys():
                raise RuntimeError("{} is not a component".format(comp_name))

            comp_id = self._comp_name_to_id[comp_name]

            for i, n in enumerate(name):
                if n not in self._registered_sindex_parameters:
                    msg = """{} is not a parameter 
                    of model {}""".format(name, self.__class__.__name__)
                    raise RuntimeError(msg)
                self._sindex_params.set_value(comp_id, n, value[i])

        elif (isinstance(comp_name, six.string_types) or isinstance(comp_name, numbers.Integral)) and \
                isinstance(name, six.string_types) and \
                (isinstance(value, six.string_types) or isinstance(value, numbers.Number)):

            if comp_name not in self._comp_name_to_id.keys():
                raise RuntimeError("{} is not a component".format(comp_name))

            comp_id = self._comp_name_to_id[comp_name]
            self._sindex_params.set_value(comp_id, name, value)

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

    def del_component(self, comp_name):
        """
        Removes component form binding model
        :param comp_name: component id
        :return: None
        """
        warnings.warn("Carfull use for cadet simulations. Not sure will work")
        # TODO: think what happends with order of ids that go to cadet
        comp_id = self._comp_name_to_id[comp_name]
        del self._comp_name_to_id[comp_name]
        del self._comp_id_to_name[comp_id]
        self._components.remove(comp_id)
        self._sindex_params.drop(comp_id)

    def list_component_names(self):
        """
        Returns list of names for components
        """
        return list(self._comp_name_to_id.keys())

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

    def scalar_parameters(self):
        for n, v in self._scalar_params.items():
            yield n, v

    def get_index_parameters_dict(self, with_defaults=False):
        """
        Returns index parameters
        :param with_defaults: flag indicating if default parameters must be included
        :return: Nested dictionary with index parameters
        """
        container = dict()
        for cid in self._components:
            cname = self._comp_id_to_name[cid]
            container[cname] = dict()
            for name in self._registered_sindex_parameters:
                if with_defaults:
                    container[cname][name] = self._sindex_params.get_value(cid, name)
                else:
                    if name not in self._default_sindex_params.keys():
                        container[cname][name] = self._sindex_params.get_value(cid, name)
        return container

    def get_sindex_parameters(self, with_defaults=False):
        """

        :param with_defaults: flag indicating if default parameters must be included
        :return: DataFrame with parameters indexed with components
        """
        if not with_defaults:
            defaults = {k for k in self._default_sindex_params.keys()}
            df = self._sindex_params.drop(defaults, axis=1)
            df.dropna(axis=1, how='all', inplace=True)
        else:
            df = self._sindex_params.dropna(axis=1, how='all')

        old_index = df.index
        new_index = [self._comp_id_to_name[cid] for cid in old_index]

        as_list = sorted(df.index.tolist())
        for i in range(len(as_list)):
            idx = as_list.index(i)
            as_list[i] = self._comp_id_to_name[idx]
        df.index = as_list

        return df

    def __setattr__(self, name, value):

        if isinstance(value, BindingModel):
            value._model = self
            value._set_params()
            if len(self._binding_models) >= 1:
                raise NotImplemented("Multiple binging models not supported yet")
            self._binding_models.append(value)
        super(ChromatographyModel, self).__setattr__(name, value)

@ChromatographyModel.register
class GRModel(ChromatographyModel):
    def __init__(self, inputs):
        # call parent binding model constructor
        super().__init__()

        # parse inputs
        args = self._parse_inputs(inputs)
        self._parse_scalar_params(args)
        self._parse_components(args)


