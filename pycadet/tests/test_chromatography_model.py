from pycadet.model.chromatograpy_model import ChromatographyModel, GRModel
from pycadet.utils.compare import equal_dictionaries, pprint_dict
from collections import OrderedDict
import unittest
import tempfile
import yaml
import h5py
import shutil
import os


class TestChromatographyModel(unittest.TestCase):

    def setUp(self):

        self.test_data = dict()
        self.test_data['components'] = ['salt',
                                        'lysozyme',
                                        'cytochrome',
                                        'ribonuclease']

        self.test_data['scalar parameters'] = dict()

        sparams = self.test_data['scalar parameters']

        # set scalar params
        sparams['sma_lambda'] = 1200

    def test_parsing_from_dict(self):
        parsed = ChromatographyModel._parse_inputs(self.test_data)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_from_yaml(self):

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "test_data.yml")

        with open(filename, 'w') as outfile:
            yaml.dump(self.test_data, outfile, default_flow_style=False)

        parsed = ChromatographyModel._parse_inputs(filename)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

        shutil.rmtree(test_dir)

    def test_parsing_scalar_params(self):
        m = GRModel(self.test_data)
        self.assertEqual(m.num_scalar_parameters, 1)
        parsed = m.get_scalar_parameters()
        unparsed = self.test_data['scalar parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_components(self):
        m = GRModel(self.test_data)
        inner = m.list_components()
        outer = set(self.test_data['components'])
        self.assertEqual(m.num_components, len(outer))
        for c in inner:
            self.assertTrue(c in outer)

    @unittest.skip("ignored for now")
    def test_del_component(self):
        m = GRModel(self.test_data)
        m.del_component('lysozyme')
        self.assertEqual(m.num_components, 3)
        self.assertFalse('lisozome' in m.list_components())

    def test_add_component(self):
        m = GRModel(self.test_data)
        m.add_component('chlorine')
        self.assertEqual(m.num_components, 5)
        self.assertTrue('chlorine' in m.list_components())
        self.assertTrue('chlorine' in m._comp_name_to_id.keys())

    def test_num_components(self):
        m = GRModel(self.test_data)
        self.assertEqual(4, m.num_components)

    def test_is_salt(self):
        m = GRModel(self.test_data)
        m.salt = 'salt'
        self.assertTrue(m.is_salt('salt'))
        self.assertFalse(m.is_salt('blah'))
