from __future__ import print_function
from pycadet.utils import pandas_utils as pd_utils
from pycadet.model.registrar import Registrar
from enum import Enum
from pycadet.utils import parse_utils
import pandas as pd
import numpy as np
import warnings
import logging
import h5py
import copy
import abc
import six
import os

logger = logging.getLogger(__name__)


class UnitOperationType(Enum):
    INLET = 0
    COLUMN = 1
    OUTLET = 2
    UNDEFINED = 3

    def __str__(self):
        return "{}".format(self.name)


class UnitOperation(abc.ABC):

    def __init__(self, data=None, sections=[], **kwargs):

        # Define type of unit operation
        self._unit_type = UnitOperationType.UNDEFINED

        # define parameters
        self._registered_scalar_parameters = set()
        self._registered_index_parameters = set()

        # define default values for indexed parameters
        self._default_scalar_params = dict()
        self._default_index_params = dict()

        self._scalar_params = dict()
        self._index_params = pd.DataFrame()

        # define link to model
        self._model = None

        self._inputs = data
        if data is not None:
            self._inputs = self._parse_inputs(data)

        # define unit internal id
        self._unit_id = None

        self._sections = []
        # append sections
        if len(sections):
            self.add_section(sections)

        # defines components in unit operation
        self._components = set()

        # set name
        self._name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def num_components(self):
        """
        Returns number of components in unit operation
        """
        return len(self._components)

    @property
    def num_sections(self):
        return len(self._sections)

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

    @property
    def num_index_parameters(self):
        """
        Returns number of indexed parameters (ignore default columns)
        """
        defaults = {k for k in self._default_index_params.keys()}

        df = self._index_params.drop(sorted(defaults), axis=1)
        df.dropna(axis=1, how='all', inplace=True)

        return len(df.columns)

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
            msg = """UnitOperation not attached to a Chromatography model.
                     When a Unit operation is created it must be attached to a Chromatography
                     model e.g \\n m = GRModel() \\n m.inlet = Inlet(). Alternatively,
                     call inlet.attach_to_model(m, name) to fix the problem"""
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

    @abc.abstractmethod
    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """

    def add_section(self, section):
        """

        :param section: Chromatography model defined section
        :return: None
        """
        self.num_components
        print("TODO")


@UnitOperation.register
class Inlet(UnitOperation):

    def __init__(self, data=None, sections=[], **kwargs):

        super().__init__(data=data,
                         sections=sections,
                         **kwargs)

        # Define type of unit operation
        self._unit_type = UnitOperationType.INLET

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """
        print("TODO")


@UnitOperation.register
class Column(UnitOperation):

    def __init__(self, data=None, sections=[], **kwargs):

        super().__init__(data=data, sections=sections, **kwargs)

        self._registered_scalar_parameters = \
            Registrar.column_parameters['scalar']

        self._registered_index_parameters = \
            Registrar.column_parameters['index']

        # set defaults
        self._default_scalar_params = \
            Registrar.column_parameters['scalar def']

        self._default_index_params = \
            Registrar.column_parameters['index def']

        # Define type of unit operation
        self._unit_type = UnitOperationType.INLET

        if self.num_sections > 0:
            raise RuntimeError('Multiple sections per column not supported yet')

        # binding model
        self._binding = None


    @property
    def dispersion(self):
        return self.get_scalar_parameter('col_dispersion')

    @dispersion.setter
    def dispersion(self, value):
        self.set_scalar_parameter('col_dispersion', value)

    @property
    def length(self):
        return self.get_scalar_parameter('col_length')

    @length.setter
    def length(self, value):
        self.set_scalar_parameter('col_length', value)

    @property
    def particle_porosity(self):
        return self.get_scalar_parameter('par_porosity')

    @particle_porosity.setter
    def particle_porosity(self, value):
        self.set_scalar_parameter('par_porosity', value)

    @property
    def column_porosity(self):
        return self.get_scalar_parameter('col_porosity')

    @particle_porosity.setter
    def column_porosity(self, value):
        self.set_scalar_parameter('par_porosity', value)

    @property
    def particle_radius(self):
        return self.get_scalar_parameter('par_radius')

    @particle_porosity.setter
    def particle_radius(self, value):
        self.set_scalar_parameter('par_radius', value)

    @property
    def velocity(self):
        return self.get_scalar_parameter('velocity')

    @velocity.setter
    def velocity(self, value):
        self.set_scalar_parameter('velocity', value)

    @property
    def binding_model(self):
        if self._binding is not None:
            return getattr(self._model(), self._binding)
        else:
            msg = """ Binding model not set yet
            """
            raise RuntimeError(msg)

    @binding_model.setter
    def binding_model(self, name):
        if isinstance(name, six.string_types):
            if hasattr(self._model(), name):
                self._binding = name
            else:
                msg = """"{} is not a binding model of 
                the chromatogrphy model""".format(name)
                raise RuntimeError(msg)
        else:
            bm = name.name
            if hasattr(self._model(), bm):
                self._binding = bm
            else:
                msg = """"{} is not a binding model of 
                the chromatogrphy model""".format(bm)
                raise RuntimeError(msg)

    def init_c(self, comp_name):
        return self.get_index_parameter(comp_name, 'init_c')

    #def init_cp(self, comp_name):
    #    return self.get_index_parameter(comp_name, 'init_cp')

    def init_q(self, comp_name):
        return self.get_index_parameter(comp_name, 'init_q')

    def film_diffusion(self, comp_name):
        return self.get_index_parameter(comp_name, 'film_diffusion')

    def par_diffusion(self, comp_name):
        return self.get_index_parameter(comp_name, 'par_diffusion')

    def set_init_c(self, comp_name, value):
        self.set_index_parameter(comp_name, 'init_c', value)

    #def set_init_cp(self, comp_name, value):
    #    self.set_index_parameter(comp_name, 'init_cp', value)

    def set_init_q(self, comp_name, value):
        self.set_index_parameter(comp_name, 'init_q', value)

    def set_film_diffusion(self, comp_name, value):
        self.set_index_parameter(comp_name, 'film_diffusion', value)

    def set_par_diffusion(self, comp_name, value):
        self.set_index_parameter(comp_name, 'par_diffusion', value)

    def is_fully_specified(self):
        self._check_model()
        df = self.get_index_parameters()
        has_nan = df.isnull().values.any()
        for k in self._registered_scalar_parameters:
            if k not in self.get_scalar_parameters(True).keys():
                print("Missing scalar parameter {}".format(k))
                return False
            if k != 'binding':
                if np.isnan(self._scalar_params[k]):
                    print("Parameter {} is nan".format(k))
                    return False
            else:
                if self._scalar_params[k] is None:
                    print("Binding model needs to be specified".format(k))
                    return False

        return not has_nan

    def _fill_containers(self):

        # initialize containers
        for k in self._registered_scalar_parameters:
            if k != 'binding':
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

        # link binding model
        if 'binding' in self._scalar_params.keys():
            bm = self._scalar_params['binding']
            if hasattr(self._model(), bm):
                self._binding = bm
            else:
                msg = """"{} is not a binding model of 
                the chromatogrphy model""".format(bm)
                raise RuntimeError(msg)

        self._parse_index_parameters()
        self._fill_containers()
        self._inputs = None

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """

        self._check_model()
        if not self.is_fully_specified():
            print(self.get_index_parameters())
            raise RuntimeError("Missing parameters")

        unitname = str(self._unit_id).zfill(3)
        with h5py.File(filename, 'a') as f:

            subgroup_name = os.path.join("input", "model", unitname)
            if subgroup_name not in f:
                f.create_group(subgroup_name)
            column = f[subgroup_name]

            # scalar parameters
            # strings
            # unit type
            s = 'GENERAL_RATE_MODEL'
            dtype = 'S{}'.format(len(s)+1)
            pointer = np.array(s, dtype=dtype)
            column.create_dataset('UNIT_TYPE',
                                  data=pointer,
                                  dtype=dtype)
            # adsorption model
            s = str(self.binding_model.binding_type)
            dtype = 'S{}'.format(len(s) + 1)
            pointer = np.array(s, dtype=dtype)
            column.create_dataset('ADSORPTION_MODEL',
                                  data=pointer,
                                  dtype=dtype)

            # integers
            value = self.num_components
            pointer = np.array(value, dtype='i')
            column.create_dataset('NCOMP',
                                  data=pointer,
                                  dtype='i')

            # doubles
            list_params = ['col_length',
                           'col_porosity',
                           'par_porosity',
                           'par_radius',
                           'col_dispersion',
                           'velocity']

            for p in list_params:
                value = self.get_scalar_parameter(p)
                name = p.upper()
                pointer = np.array(value, dtype='d')
                column.create_dataset(name,
                                      data=pointer,
                                      dtype='d')

            # index parameters
            # doubles
            list_params=['init_c',
                         'init_q',
                         'film_diffusion',
                         'par_diffusion',
                         'par_surfdiffusion']

            list_ids = self.list_components(ids=True)
            _index_params = self.get_index_parameters(ids=True, with_defaults=True)

            num_components = self.num_components
            for k in list_params:
                cadet_name = k.upper()
                param = _index_params[k]
                pointer = np.zeros(num_components, dtype='d')
                for i in list_ids:
                    ordered_id = self._model()._ordered_ids_for_cadet.index(i)
                    pointer[ordered_id] = param[i]

                column.create_dataset(cadet_name,
                                      data=pointer,
                                      dtype='d')

        self.binding_model.write_to_cadet_input_file(filename, unitname)

@UnitOperation.register
class Outlet(UnitOperation):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Define type of unit operation
        self._unit_type = UnitOperationType.OUTLET

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """
        print("TODO")


