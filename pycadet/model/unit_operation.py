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

    def __init__(self, **kwargs):

        sublist_components = kwargs.pop('components',None)
        # Define type of unit operation
        self._unit_type = UnitOperationType.UNDEFINED

        if sublist_components is not None:
            raise RuntimeError("sublist components feature not supported")

        # define link to model
        self._model = None

        # TODO: manage this internally
        # define unit internal id
        self._unit_id = None

    @property
    def num_components(self):
        return self._model().num_components

    @abc.abstractmethod
    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """


@UnitOperation.register
class Inlet(UnitOperation):

    def __init__(self):
        print("TODO")

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """
        print("TODO")


@UnitOperation.register
class Column(UnitOperation):

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
    def __init__(self):
        print("TODO")

    def write_to_cadet_input_file(self, filename, **kwargs):
        """
        Append UnitOperation model to cadet hdf5 input file
        :param filename: name of cadet hdf5 input file
        """
        print("TODO")



