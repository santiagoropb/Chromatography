from __future__ import print_function
import pandas as pd
import numpy as np
import logging
import yaml
import six

logger = logging.getLogger(__name__)


class section(object):

    def __init__(self, inputs):

        # define registered parameters
        self._coefficients = {'a0', 'a1', 'a2', 'a3'}

        # define container components
        self._components = set()
        self._index_params = pd.DataFrame(index=self._components,
                                          columns=self._coefficients)

    @staticmethod
    def _parse_inputs(inputs):
        """
        Parse inputs
        :param inputs: filename or dictionary with inputs to the binding model
        :return: dictionary with parsed inputs
        """
        if isinstance(inputs, dict):
            args = inputs
        elif isinstance(inputs, six.string_types):
            if ".yml" in inputs or ".yaml" in inputs:
                with open(inputs, 'r') as f:
                    args = yaml.load(f)
            else:
                raise RuntimeError('File format not implemented yet. Try .yml or .yaml')
        else:
            raise RuntimeError('inputs must be a dictionary or a file')

        return args

    @property
    def num_components(self):
        """
        Returns number of components in the binding model
        """
        return len(self._components)

    def _parse_components(self, args):
        """
        Parse components and indexed parameters
        :param args: dictionary with parsed inputs
        :return: None
        """
        dict_comp = args
        has_params = False
        if dict_comp is not None:

            if isinstance(dict_comp, list):
                to_loop = dict_comp
            else:
                to_loop = dict_comp.keys()
                has_params = True

            for comp_id in to_loop:
                if comp_id in self._components:
                    msg = "Component {} being overwritten".format(comp_id)
                    logger.warning(msg)
                self._components.add(comp_id)
        else:
            logger.error(" No components found")
            raise RuntimeError('The section needs to know the components')

        if self.num_components == 0:
            msg = """No components specified 
                    when parsing section"""
            logger.debug(msg)

        n_coef = len(self._coefficients)
        data = np.zeros((self.num_components, n_coef), dtype="d")

        self._index_params = pd.DataFrame(data=data,
                                          index=self._components,
                                          columns=sorted(self._coefficients))

        self._index_params.index.name = 'component id'
        self._index_params.columns.name = 'coefficient'

        if has_params:
            for comp_id, params in dict_comp.items():
                for parameter, value in params.items():
                    msg = """{} is not a valid coefficient""".format(parameter)
                    assert parameter in self._coefficients, msg
                    self._index_params.set_value(comp_id, parameter, value)
        else:
            msg = """ No coefficients specified when parsing section """
            logger.debug(msg)