from __future__ import print_function
from pycadet.model.data_manager import DataManager
from pycadet.model.binding_model import BindingModel
from pycadet.model.registrar import Registrar
from pycadet.model.section import Section
from enum import Enum
import pandas as pd
import numpy as np
import logging
import h5py
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


class InletType(Enum):
    PIECEWISE_CUBIC_POLY = 0

    def __str__(self):
        return "{}".format(self.name)


class UnitOperation(DataManager, abc.ABC):

    def __init__(self, data=None, sections=None, **kwargs):

        # Define type of unit operation
        self._unit_type = UnitOperationType.UNDEFINED

        # define unit internal id
        self._unit_id = None

        self._sections = []
        # append sections
        if sections is not None:
            for s in sections:
                self.add_section(s)

        super().__init__(data=data, **kwargs)

    @property
    def num_sections(self):
        return len(self._sections)

    @property
    def num_scalar_parameters(self):
        """
        Returns number of scalar parameters
        """
        return len(self.get_scalar_parameters(with_defaults=False))

    def _check_model(self):
        if self._model is None:
            msg = """UnitOperation not attached to a Chromatography model.
                     When a Unit operation is created it must be attached to a Chromatography
                     model e.g \\n m = GRModel() \\n m.inlet = Inlet(). Alternatively,
                     call inlet.attach_to_model(m, name) to fix the problem"""
            raise RuntimeError(msg)

    @abc.abstractmethod
    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """

    def add_section(self, name):
        """

        :param section: Chromatography model defined section
        :return: None
        """

        if isinstance(name, six.string_types):
            if hasattr(self._model(), name):
                sec = getattr(self._model(), name)
                sec_components = sec.list_components(ids=True)
                if sec.num_components != self.num_components:
                    msg = """ The section does not
                    have the same number of components
                    as the {}
                    """.format(self.__class__.__name__)
                    raise RuntimeError(msg)
                for i in sec_components:
                    if i not in self.list_components(ids=True):
                        msg = """ Component {} in section is not a 
                        component of
                        {} """.format(self._model()._comp_id_to_name[i],
                                      self.__class__.__name__)
                        raise RuntimeError(msg)
                self._sections.append(name)
            else:
                msg = """"{} is not a section of 
                        the chromatogrphy model""".format(name)
                raise RuntimeError(msg)
        elif isinstance(name, Section):
            sec = name
            sec_components = sec.list_components(ids=True)
            if sec.num_components != self.num_components:
                msg = """ The section does not
                                have the same number of components
                                as the {}
                                """.format(self.__class__.__name__)
                raise RuntimeError(msg)
            for i in sec_components:
                if i not in self.list_components(ids=True):
                    msg = """ Component {} in section is not a 
                                    component of
                                    {} """.format(self._model()._comp_id_to_name[i],
                                                  self.__class__.__name__)
                    raise RuntimeError(msg)

            section = name.name
            if hasattr(self._model(), section):
                self._sections.append(section)
            else:
                msg = """"{} is not a section of 
                            the chromatogrphy model""".format(section)
                raise RuntimeError(msg)
        else:
            raise RuntimeError("input not recognized")


@UnitOperation.register
class Inlet(UnitOperation):

    def __init__(self, data=None, sections=None, **kwargs):

        super().__init__(data=data,
                         sections=sections,
                         **kwargs)

        # Define type of unit operation
        self._unit_type = UnitOperationType.INLET

        # Define type inlet
        self._inlet_type = InletType.PIECEWISE_CUBIC_POLY

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append inlet to cadet hdf5 input file
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
            inlet = f[subgroup_name]

            s = str(self._unit_type)
            dtype = 'S{}'.format(len(s) + 1)
            pointer = np.array(s, dtype=dtype)
            inlet.create_dataset('UNIT_TYPE',
                                  data=pointer,
                                  dtype=dtype)

            s = str(self._inlet_type)
            dtype = 'S{}'.format(len(s) + 1)
            pointer = np.array(s, dtype=dtype)
            inlet.create_dataset('INLET_TYPE',
                                 data=pointer,
                                 dtype=dtype)

            # integers
            value = self.num_components
            pointer = np.array(value, dtype='i')
            inlet.create_dataset('NCOMP',
                                  data=pointer,
                                  dtype='i')

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

    @column_porosity.setter
    def column_porosity(self, value):
        self.set_scalar_parameter('par_porosity', value)

    @property
    def particle_radius(self):
        return self.get_scalar_parameter('par_radius')

    @particle_radius.setter
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
        elif isinstance(name, BindingModel):
            bm = name.name
            if hasattr(self._model(), bm):
                self._binding = bm
            else:
                msg = """"{} is not a binding model of 
                the chromatogrphy model""".format(bm)
                raise RuntimeError(msg)
        else:
            raise RuntimeError("input not recognized")

    def init_c(self, comp_name):
        return self.get_index_parameter(comp_name, 'init_c')

    def init_q(self, comp_name):
        return self.get_index_parameter(comp_name, 'init_q')

    def film_diffusion(self, comp_name):
        return self.get_index_parameter(comp_name, 'film_diffusion')

    def par_diffusion(self, comp_name):
        return self.get_index_parameter(comp_name, 'par_diffusion')

    def set_init_c(self, comp_name, value):
        self.set_index_parameter(comp_name, 'init_c', value)

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


