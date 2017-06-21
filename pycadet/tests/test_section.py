from pycadet.model.chromatograpy_model import GRModel
from pycadet.model.section import Section
from pycadet.utils.compare import equal_dictionaries, pprint_dict
from collections import OrderedDict
import numpy as np
import unittest
import tempfile
import yaml
import h5py
import shutil
import os


class TestSection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_model_data = dict()
        cls.base_model_data['components'] = ['salt',
                                             'lysozyme',
                                             'cytochrome',
                                             'ribonuclease']

        cls.base_model_data['scalar parameters'] = dict()
        cls.m = GRModel(data=cls.base_model_data)

    def setUp(self):
        self.test_data = dict()
        self.test_data['index parameters'] = OrderedDict()
        self.test_data['scalar parameters'] = dict()

        comps = self.test_data['index parameters']
        sparams = self.test_data['scalar parameters']

        sparams['start_time_sec'] = 0.0

        # index parameters
        # components and index params
        self.comp_names = ['salt',
                           'lysozyme',
                           'cytochrome',
                           'ribonuclease']

        for cname in self.comp_names:
            comps[cname] = dict()

        cid = 'salt'
        comps[cid]['const_coeff'] = 50.0
        comps[cid]['lin_coeff'] = 0.0
        comps[cid]['quad_coeff'] = 0.0
        comps[cid]['cube_coeff'] = 0.0

        cid = 'lysozyme'
        comps[cid]['const_coeff'] = 1.0
        comps[cid]['lin_coeff'] = 0.0
        comps[cid]['quad_coeff'] = 0.0
        comps[cid]['cube_coeff'] = 0.0

        cid = 'cytochrome'
        comps[cid]['const_coeff'] = 1.0
        comps[cid]['lin_coeff'] = 0.0
        comps[cid]['quad_coeff'] = 0.0
        comps[cid]['cube_coeff'] = 0.0

        cid = 'ribonuclease'
        comps[cid]['const_coeff'] = 1.0
        comps[cid]['lin_coeff'] = 0.0
        comps[cid]['quad_coeff'] = 0.0
        comps[cid]['cube_coeff'] = 0.0


    def test_is_fully_specified(self):

        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        self.assertTrue(GRM.sec.is_fully_specified())

    def test_parsing_from_dict(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        parsed = sec._parse_inputs(self.test_data)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_from_yaml(self):
        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "test_data.yml")

        with open(filename, 'w') as outfile:
            yaml.dump(self.test_data, outfile, default_flow_style=False)

        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        parsed = sec._parse_inputs(filename)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

        shutil.rmtree(test_dir)

    def test_parsing_scalar_params(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        self.assertEqual(sec.num_scalar_parameters, 1)
        parsed = sec.get_scalar_parameters()
        unparsed = self.test_data['scalar parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_components(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        parsed = sec.get_index_parameters(form='dictionary')
        unparsed = self.test_data['index parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_set_index_param(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        sec.set_index_parameter('lysozyme', 'lin_coeff', 777)
        parsed = sec.get_index_parameters(form='dictionary')
        self.test_data['index parameters']['lysozyme']['lin_coeff'] = 777
        unparsed = self.test_data['index parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_num_components(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        self.assertEqual(4, sec.num_components)

    def test_num_index_params(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        self.assertEqual(4, sec.num_index_parameters)

    def test_a0(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'salt'
        val = self.test_data['index parameters'][cname]['const_coeff']
        self.assertEqual(sec.a0(cname), val)

    def test_a1(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'salt'
        val = self.test_data['index parameters'][cname]['lin_coeff']
        self.assertEqual(sec.a1(cname), val)

    def test_a2(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'salt'
        val = self.test_data['index parameters'][cname]['quad_coeff']
        self.assertEqual(sec.a2(cname), val)

    def test_a3(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'salt'
        val = self.test_data['index parameters'][cname]['cube_coeff']
        self.assertEqual(sec.a1(cname), val)

    def test_set_a0(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'cytochrome'
        val = 7.0
        sec.set_a0(cname, val)
        self.assertEqual(sec.a0(cname), val)

    def test_set_a1(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'cytochrome'
        val = 7.0
        sec.set_a1(cname, val)
        self.assertEqual(sec.a1(cname), val)

    def test_set_a2(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'cytochrome'
        val = 7.0
        sec.set_a2(cname, val)
        self.assertEqual(sec.a2(cname), val)

    def test_set_a3(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec
        cname = 'cytochrome'
        val = 7.0
        sec.set_a3(cname, val)
        self.assertEqual(sec.a3(cname), val)

    def test_f(self):
        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec

        components = sec.list_components()
        vals = [1.0]*len(components)
        c_vars = dict(zip(components,vals))
        for cname in components:
            a0 = self.test_data['index parameters'][cname]['const_coeff']
            a1 = self.test_data['index parameters'][cname]['lin_coeff']
            a2 = self.test_data['index parameters'][cname]['quad_coeff']
            a3 = self.test_data['index parameters'][cname]['cube_coeff']
            f = a0 + a1*c_vars[cname] + a2*c_vars[cname]**2 +a3*c_vars[cname]**3
            self.assertEqual(f, sec.f(cname, c_vars))

    def test_write_to_cadet_input_file(self):

        GRM = self.m
        GRM.sec = Section(data=self.test_data)
        sec = GRM.sec

        test_dir = tempfile.mkdtemp()

        filename = os.path.join(test_dir, "col_tmp.hdf5")

        unitname = 'unit_000'
        sec.write_to_cadet_input_file(filename, unitname)

        section_name = str(sec._section_id).zfill(3)
        # read back and verify output
        with h5py.File(filename, 'r') as f:
            path = os.path.join("input", "model", unitname, section_name)

            list_params = list(sec._registered_index_parameters)

            for p in list_params:
                name = p.upper()
                dataset = os.path.join(path, name)
                read = f[dataset]
                for i, e in enumerate(read):
                    comp_id = GRM._ordered_ids_for_cadet[i]
                    comp_name = GRM._comp_id_to_name[comp_id]
                    value = self.test_data['index parameters'][comp_name][p]
                    self.assertEqual(value, e)




