from __future__ import print_function
from pycadet.utils.parse_utils import parse_inputs
import pandas as pd
import numpy as np
import logging
import weakref
import warnings

logger = logging.getLogger(__name__)


class Section(object):

    __slots__ = ('_name',
                 '_start_time_sec',
                 '_end_time_sec',
                 '_coefficients',
                 '_model')

    def __init__(self,
                 coefficients,
                 start_time_sec=0,
                 end_time_sec=-1):

        # name section
        self._name = None

        # set start time
        self._start_time_sec = start_time_sec

        # set end time
        self._end_time_sec = end_time_sec

        # set coefficients
        if not isinstance(coefficients, dict):
            msg = """Section needs a nested dictionary 
                                comp_name -> coeff_name ->value"""
            raise RuntimeError(msg)

        if len(coefficients):
            for k, v in coefficients.items():
                if not isinstance(v, dict):
                    msg = """Section needs a nested dictionary 
                    comp_name -> coeff_name ->value"""
                    raise RuntimeError(msg)

        self._coefficients = coefficients

        # link to Chromatography model
        self._model = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def start_time_sec(self):
        """
        Return start time in seconds. default 0
        """
        return self._start_time_sec

    @start_time_sec.setter
    def start_time_sec(self, value):
        self._start_time_sec = value

    @property
    def end_time_sec(self):
        """
        Return end time in seconds. default -1
        """
        return self._end_time_sec

    @end_time_sec.setter
    def end_time_sec(self, value):
        """

        :param value: time in seconds
        :return: None
        """
        self._end_time_sec = value

    def add_coefficient(self, comp_name, coeff_name, value):
        """

        :param comp_name: name of component
        :param coeff_name: name of coefficient (a0,a1,a2,a3)
        :param value: value of coefficient
        :return: None
        """
        # TODO: expand to allow multiple adds
        if comp_name not in self._coefficients.keys():
            self._coefficients[comp_name] = dict()
        self._coefficients[comp_name][coeff_name] = value

    # TODO create set method for coefficients

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

    def _check_model(self):
        if self._model is None:
            msg = """Section not attached to a Chromatography model.
                     When a section is created it must be attached to a Chromatography
                     model e.g \\n m = GRModel() \\n m.section1 = Section(coeff). Alternatively,
                     call section.attach_to_model(m, name) to fix the problem"""
            raise RuntimeError(msg)

    def write_to_cadet_input_file(self, filename, unitname, **kwargs):
        """
        Write section to hdf5 file
        :param filename:
        :param unitname:
        :param kwargs:
        :return:
        """
        print("TODO")

