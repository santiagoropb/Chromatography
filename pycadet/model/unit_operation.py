from __future__ import print_function
from enum import Enum
from pycadet.utils.parse_utils import parse_inputs
import pandas as pd
import numpy as np
import logging
import weakref
import abc

logger = logging.getLogger(__name__)


class UnitOperationType(Enum):
    INLET = 0
    COLUMN = 1
    OUTLET = 2
    UNDEFINED = 3

    def __str__(self):
        return "{}".format(self.name)


class UnitOperation(abc.ABC):

    def __init__(self, sections=[], **kwargs):

        sublist_components = kwargs.pop('components', None)
        # Define type of unit operation
        self._unit_type = UnitOperationType.UNDEFINED

        if sublist_components is not None:
            raise RuntimeError("sublist components feature not supported")

        # define link to model
        self._model = None

        # TODO: manage this internally
        # define unit internal id
        self._unit_id = None

        self._sections = []
        # append sections
        if len(sections):
            self.add_section(sections)

    @property
    def num_components(self):
        return self._model().num_components

    @property
    def num_sections(self):
        return len(self._sections)

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
        self.num_components()
        print("TODO")


@UnitOperation.register
class Inlet(UnitOperation):

    def __init__(self, sections=[], **kwargs):

        self.super().__init__(sections=sections,**kwargs)

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

    def __init__(self, sections=[], **kwargs):

        self.super().__init__(sections=sections, **kwargs)

        # Define type of unit operation
        self._unit_type = UnitOperationType.INLET

        if self.num_sections > 0:
            raise RuntimeError('Multiple sections per column not supported yet')


    @property
    def dispersion(self):
        return self._model().get_scalar_parameter('col_dispersion')

    @dispersion.setter
    def dispersion(self, value):
        self._model().set_scalar_parameter('col_dispersion', value)

    @property
    def length(self):
        return self._model().get_scalar_parameter('col_length')

    @length.setter
    def length(self, value):
        self._model().set_scalar_parameter('col_length', value)

    @property
    def particle_porosity(self):
        return self._model().get_scalar_parameter('par_porosity')

    @particle_porosity.setter
    def particle_porosity(self, value):
        self._model().set_scalar_parameter('par_porosity', value)

    @property
    def column_porosity(self):
        return self._model().get_scalar_parameter('col_porosity')

    @particle_porosity.setter
    def column_porosity(self, value):
        self._model().set_scalar_parameter('par_porosity', value)

    @property
    def particle_radius(self):
        return self._model().get_scalar_parameter('par_radius')

    @particle_porosity.setter
    def particle_radius(self, value):
        self._par_radius = value





    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperationn to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """
        print("TODO")


@UnitOperation.register
class Outlet(UnitOperation):
    def __init__(self):
        print("TODO")

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """
        print("TODO")


@UnitOperation.register
class Outlet(UnitOperation):

    def __init__(self, **kwargs):

        self.super().__init__(**kwargs)

        # Define type of unit operation
        self._unit_type = UnitOperationType.OUTLET

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """
        print("TODO")


