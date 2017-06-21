from __future__ import print_function
from pycadet.model.registrar import Registrar
from pycadet.model.binding_model import BindingModel
from pycadet.model.section import Section
from pycadet.model.unit_operation import UnitOperation, Column
from pycadet.utils import parse_utils
from collections import OrderedDict
import pandas as pd
import collections
import warnings
import logging
import numbers
import weakref
import copy
import six
import abc

logger = logging.getLogger(__name__)


class ChromatographyModel(abc.ABC):

    def __init__(self, components=None):

        # define registered params
        self._registered_scalar_parameters = Registrar.scalar_parameters

        # define default values for indexed parameters
        self._default_scalar_params = Registrar.default_scalar_parameters

        # define scalar params container
        self._scalar_params = dict()

        # define components and component indexed parameters
        self._comp_name_to_id = dict()
        self._comp_id_to_name = dict()
        self._components = set()

        self._salt_id = -1

        self._binding_models =list()

        self._sections = list()

        self._units = list()

        self._ordered_ids_for_cadet = list()

        # add passed components
        if components is not None:
            for cname in components:
                self.add_component(cname)

    @property
    def salt(self):
        if self._salt_id != -1:
            name = self._comp_id_to_name[self._salt_id]
            return name
        return None

    @salt.setter
    def salt(self, name):
        self._salt_id = self._comp_name_to_id[name]
        self._ordered_ids_for_cadet.remove(self._salt_id)
        self._ordered_ids_for_cadet.insert(0, self._salt_id)

    def is_salt(self, name):

        if name not in self._comp_name_to_id.keys():
            return False
        if self._comp_name_to_id[name] == self._salt_id:
            return True
        return False

    def is_model_component(self, name):
        return name in self._comp_name_to_id.keys()

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
        registered_inputs = self._registered_scalar_parameters
        parsed = parse_utils.parse_scalar_inputs_from_dict(sparams,
                                                           self.__class__.__name__,
                                                           registered_inputs,
                                                           logger)

        for k, v in parsed.items():
            self._scalar_params[k] = v

        for k, val in self._default_scalar_params.items():
            self._scalar_params[k] = val

    def add_component(self, name):
        if name not in self._comp_name_to_id.keys():
            new_id = len(self._components)
            self._comp_name_to_id[name] = new_id
            self._comp_id_to_name[new_id] = name
            self._components.add(new_id)
            self._ordered_ids_for_cadet.append(new_id)

    def _parse_components(self, args):
        """
        Parse components and indexed parameters
        :param args: dictionary with parsed inputs
        :return: None
        """
        list_comp = args.get('components', [])
        for cname in list_comp:
            self.add_component(cname)

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
        self._ordered_ids_for_cadet.remove(comp_id)

    def list_components(self, ids=False):
        """
        Returns list of names for components
        """
        if ids:
            return list(self._comp_id_to_name.keys())
        else:
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

    def scalar_parameters(self, with_defaults=False):
        if with_defaults:
            for n, v in self._scalar_params.items():
                yield n, v
        else:
            for n, v in self._scalar_params.items():
                if n not in self._default_scalar_params:
                    yield n,v

    def get_scalar_parameter(self, name):
        """

        :param name: name of the scalar parameter
        :return: value of scalar parameter
        """

        return self._scalar_params[name]

    def list_binding_models(self):
        return [n for n in self._binding_models]

    def binding_models(self):
        for n in self._binding_models:
            yield n, getattr(self,n)

    def __setattr__(self, name, value):
        # TODO: add warning if overwriting name?
        if isinstance(value, BindingModel):
            value._model = weakref.ref(self)
            value.name = name
            value._initialize_containers()
            self._binding_models.append(name)

        if isinstance(value, Section):
            value._model = weakref.ref(self)
            value._section_id = len(self._sections)
            value.name = name
            value._initialize_containers()
            self._sections.append(name)

        if isinstance(value, UnitOperation):
            value._model = weakref.ref(self)
            value._unit_id = len(self._units)
            value._initialize_containers()
            self._units.append(name)

        super(ChromatographyModel, self).__setattr__(name, value)

@ChromatographyModel.register
class GRModel(ChromatographyModel):
    def __init__(self, components=None, data=None):

        # call parent binding model constructor
        super().__init__(components=components)

        # parse inputs
        if data is not None:
            args = self._parse_inputs(data)
            self._parse_scalar_params(args)
            self._parse_components(args)


