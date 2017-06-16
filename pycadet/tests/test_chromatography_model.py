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
        self.test_data['components'] = OrderedDict()
        self.test_data['scalar parameters'] = dict()

        comps = self.test_data['components']
        sparams = self.test_data['scalar parameters']

        # set scalar params
        sparams['sma_lambda'] = 1200

        # components and index params
        comp_names = ['salt',
                      'lysozyme',
                      'cytochrome',
                      'ribonuclease']

        for cname in comp_names:
            comps[cname] = dict()

        # salt
        cid = 'salt'
        comps[cid]['sma_kads'] = 0.0
        comps[cid]['sma_kdes'] = 0.0
        comps[cid]['sma_nu'] = 0.0
        comps[cid]['sma_sigma'] = 0.0


        # lysozyme
        cid = 'lysozyme'
        comps[cid]['sma_kads'] = 35.5
        comps[cid]['sma_kdes'] = 1000.0
        comps[cid]['sma_nu'] = 4.7
        comps[cid]['sma_sigma'] = 11.83

        # cytochrome
        cid = 'cytochrome'
        comps[cid]['sma_kads'] = 1.59
        comps[cid]['sma_kdes'] = 1000.0
        comps[cid]['sma_nu'] = 5.29
        comps[cid]['sma_sigma'] = 10.6

        # ribonuclease
        cid = 'ribonuclease'
        comps[cid]['sma_kads'] = 7.7
        comps[cid]['sma_kdes'] = 1000.0
        comps[cid]['sma_nu'] = 3.7
        comps[cid]['sma_sigma'] = 10.0


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
        parsed = m.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    @unittest.skip("ignored for now")
    def test_del_component(self):
        m = GRModel(self.test_data)
        m.del_component('lysozyme')
        parsed = m.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        del self.test_data['components']['lysozyme']
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_set_sindex_param(self):

        m = GRModel(self.test_data)
        m.set_index_parameter('lysozyme', 'sma_kads', 777)
        parsed = m.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        self.test_data['components']['lysozyme']['sma_kads'] = 777
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

        # two parameters
        m.set_index_parameter('lysozyme', ['sma_kads', 'sma_kdes'], [2, 3])
        parsed = m.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        self.test_data['components']['lysozyme']['sma_kads'] = 2
        self.test_data['components']['lysozyme']['sma_kdes'] = 3
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

        # two components
        m.set_index_parameter(['salt','lysozyme'], 'sma_kads', [2, 3])
        parsed = m.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        self.test_data['components']['salt']['sma_kads'] = 2
        self.test_data['components']['lysozyme']['sma_kads'] = 3
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_add_component(self):
        # TODO: improve this test to check lists
        m = GRModel(self.test_data)
        m.add_component('water')
        n_c = m.num_components
        df = len(m.get_index_parameters().index)
        self.assertEqual(n_c, df)

        new_component = dict()
        new_component['sma_kads'] = 7.7
        new_component['sma_kdes'] = 1000.0
        new_component['sma_nu'] = 3.7
        new_component['sma_sigma'] = 10.0

        m = GRModel(self.test_data)
        m.add_component('buffer', new_component)
        self.test_data['components']['buffer'] = new_component

        parsed = m.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_num_components(self):
        m = GRModel(self.test_data)
        self.assertEqual(4, m.num_components)

    def test_num_index_params(self):
        m = GRModel(self.test_data)
        self.assertEqual(4, m.num_single_index_parameters)

    def test_is_salt(self):
        m = GRModel(self.test_data)
        m.salt = 'salt'
        self.assertTrue(m.is_salt('salt'))
        self.assertFalse(m.is_salt('blah'))

    """
    def test_get_index_parameters(self):
        m = GRModel(self.test_data)
        df = m.get_sindex_parameters()
        print(m._sindex_params)
        print(m._comp_name_to_id)
        print(df)
    """