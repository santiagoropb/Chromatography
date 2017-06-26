from __future__ import print_function
from pychrom.core.data_manager import DataManager
from pychrom.core.registrar import Registrar
from tabulate import tabulate
import pandas as pd
import numpy as np
import warnings
import logging
import h5py
import os


logger = logging.getLogger(__name__)


class Section(DataManager):

    def __init__(self, components=None, data=None, **kwargs):

        super().__init__(components=components,
                         data=data,
                         **kwargs)

        self._registered_scalar_parameters = \
            Registrar.section_parameters['scalar']

        self._registered_index_parameters = \
            Registrar.section_parameters['index']

        # set defaults
        self._default_scalar_params = \
            Registrar.section_parameters['scalar def']

        self._default_index_params = \
            Registrar.section_parameters['index def']

        # reset index params container
        self._index_params = pd.DataFrame(index=[],
                                          columns=self._registered_index_parameters)
        # define unit internal id
        self._section_id = None


    @property
    def start_time_sec(self):
        """
        Return start time in seconds. default 0
        """
        return self.get_scalar_parameter('start_time_sec')

    @property
    def num_index_parameters(self):
        """
        Returns number of indexed parameters (ignore default columns)
        """

        df = self._index_params
        df.dropna(axis=1, how='all', inplace=True)

        return len(df.columns)

    @start_time_sec.setter
    def start_time_sec(self, value):
        self.set_scalar_parameter('start_time_sec', value)

    def a0(self, comp_name):
        return self.get_index_parameter(comp_name, 'const_coeff')

    def a1(self, comp_name):
        return self.get_index_parameter(comp_name, 'lin_coeff')

    def a2(self, comp_name):
        return self.get_index_parameter(comp_name, 'quad_coeff')

    def a3(self, comp_name):
        return self.get_index_parameter(comp_name, 'cube_coeff')

    def set_a0(self, comp_name, value):
        return self.set_index_parameter(comp_name, 'const_coeff', value)

    def set_a1(self, comp_name, value):
        return self.set_index_parameter(comp_name, 'lin_coeff', value)

    def set_a2(self, comp_name, value):
        return self.set_index_parameter(comp_name, 'quad_coeff', value)

    def set_a3(self, comp_name, value):
        return self.set_index_parameter(comp_name, 'cube_coeff', value)

    def f(self, comp_name, c_var):

        df = self.get_index_parameters(with_defaults=True)
        ordered_coeff = ['const_coeff', 'lin_coeff', 'quad_coeff', 'cube_coeff']
        accum = 0.0
        for j, coeff in enumerate(ordered_coeff):
            accum += df.get_value(comp_name, coeff)*c_var[comp_name]**j
        return accum

    def f_str(self,comp_name):
        df = self.get_index_parameters(with_defaults=True)
        ordered_coeff = ['const_coeff', 'lin_coeff', 'quad_coeff', 'cube_coeff']
        accum = ''
        for j, coeff in enumerate(ordered_coeff):
            if df.get_value(comp_name, coeff)>0.0:
                aj = df.get_value(comp_name, coeff)
                if j==0:
                    accum += str(aj)
                else:
                    accum += ' +'+str(aj)+'t^{}'.format(j)
        if accum=='':
            accum='0.0'
        return accum


    def _check_model(self):
        if self._model is None:
            msg = """Section not attached to a Chromatography model.
                     When a section is created it must be attached to a Chromatography
                     model e.g \\n m = GRModel() \\n m.section1 = Section(coeff). Alternatively,
                     call section.attach_to_model(m, name) to fix the problem"""
            raise RuntimeError(msg)

    def get_index_parameters(self, with_defaults=True, ids=False, form='dataframe'):
        return super().get_index_parameters(with_defaults=with_defaults, ids=ids, form=form)

    def write_to_cadet_input_file(self, filename, unitname, **kwargs):
        """
        Write section to hdf5 file
        :param filename:
        :param unitname:
        :param kwargs:
        :return:
        """

        self._check_model()

        if not self.is_fully_specified():
            print(self.get_index_parameters())
            raise RuntimeError("Missing parameters")

        with h5py.File(filename, 'a') as f:
            subgroup_name = os.path.join("input", "model", unitname)
            if subgroup_name not in f:
                f.create_group(subgroup_name)
            subgroup = f[subgroup_name]
            section_name = 'sec_'+str(self._section_id).zfill(3)
            if section_name in subgroup:
                warnings.warn("Overwriting {}/{}".format(subgroup_name, section_name))
                section = subgroup[section_name]
            else:
                section = subgroup.create_group(section_name)

            # index parameters
            # doubles
            list_params = ['const_coeff', 'lin_coeff', 'quad_coeff', 'cube_coeff']
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

                section.create_dataset(cadet_name,
                                       data=pointer,
                                       dtype='d')

    def pprint(self, indent=0):
        t = '\t' * indent
        print(t, self.name, ":\n")

        if self.num_scalar_parameters != 0:
            print(t, "scalar parameters")
            sp = self.get_scalar_parameters(with_defaults=True)
            data = sorted([(k, v) for k, v in sp.items()])
            headers = ['parameter', 'value']
            print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

        comps = self.list_components()
        data = [(k, self.f_str(k)) for k in comps]
        headers = ['component', 'f(t)']
        table =tabulate(data,
                        headers=headers,
                        tablefmt="fancy_grid",
                        floatfmt=".1f")
        print(table)

        #super().pprint(indent=indent)



