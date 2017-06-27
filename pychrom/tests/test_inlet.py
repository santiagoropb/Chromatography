from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.section import Section
from pychrom.core.unit_operation import Inlet, UnitOperationType, InletType
from collections import OrderedDict
import numpy as np
import unittest
import tempfile
import h5py
import os


class TestInlet(unittest.TestCase):

    def setUp(self):
        self.test_data = dict()
        self.test_data['index parameters'] = OrderedDict()
        self.test_data['scalar parameters'] = dict()
        self.test_sections = list()

        self.base_model_data = dict()
        self.model_components = ['salt',
                                'lysozyme',
                                'cytochrome',
                                'ribonuclease']

        self.m = GRModel(components=self.model_components)

        GRM = self.m
        GRM.load = Section(components=self.model_components)
        for cname in self.model_components:
            GRM.load.set_a0(cname, 1.0)
        GRM.load.set_a0('salt', 50.0)

        GRM.wash = Section(components=self.model_components)
        GRM.wash.set_a0('salt', 50.0)

        GRM.elute = Section(components=self.model_components)
        GRM.elute.set_a0('salt', 100.0)
        GRM.elute.set_a1('salt', 0.2)

        GRM.inlet = Inlet(components=self.model_components)

    def test_unit_type(self):
        inlet = self.m.inlet
        self.assertEqual(inlet._unit_type, UnitOperationType.INLET)

    def test_inlet_type(self):
        inlet = self.m.inlet
        self.assertEqual(inlet._inlet_type, InletType.PIECEWISE_CUBIC_POLY)

    def test_num_sections(self):

        GRM = self.m
        self.assertEqual(GRM.inlet.num_sections, 0)
        GRM.inlet.add_section('load')
        self.assertEqual(GRM.inlet.num_sections, 1)
        GRM.inlet.add_section('wash')
        self.assertEqual(GRM.inlet.num_sections, 2)
        GRM.inlet.add_section('elute')
        self.assertEqual(GRM.inlet.num_sections, 3)

    def test_write_to_cadet(self):

        inlet = self.m.inlet

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "inlet_tmp.hdf5")

        inlet._write_to_cadet_input_file(filename)

        # read back and verify output
        with h5py.File(filename, 'r') as f:
            unitname = 'unit_'+str(inlet._unit_id).zfill(3)

            path = os.path.join("input", "model", unitname)

            strings = dict()

            s = str(inlet._unit_type)
            dtype = 'S{}'.format(len(s) + 1)
            pointer = np.array(s, dtype=dtype)
            strings['UNIT_TYPE'] = pointer

            s = str(inlet._inlet_type)
            dtype = 'S{}'.format(len(s) + 1)
            pointer = np.array(s, dtype=dtype)
            strings['INLET_TYPE'] = pointer

            for name, v in strings.items():
                dataset = os.path.join(path, name)
                # check unit type
                read = f[dataset].value
                self.assertEqual(read, v)

            # integers
            name = 'NCOMP'
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, inlet.num_components)

            for s in inlet.list_sections():
                section = inlet.get_section(s)
                section_name = 'sec_'+str(section._section_id).zfill(3)
                path = os.path.join("input", "model", unitname, section_name)

                list_params = list(section._registered_index_parameters)
                for p in list_params:
                    name = p.upper()
                    dataset = os.path.join(path, name)
                    read = f[dataset]
                    for i, e in enumerate(read):
                        comp_id = self.m._ordered_ids_for_cadet[i]
                        comp_name = self.m._comp_id_to_name[comp_id]
                        value = section.get_index_parameter(comp_name, p)
                        self.assertEqual(value, e)
