from pycadet.model.binding_model import SMAModel, ModelType
from pycadet.utils.compare import equal_dictionaries
import unittest
import tempfile
import yaml
import h5py
import shutil
import os


class TestSMAModel(unittest.TestCase):

    def setUp(self):

        self.test_data = dict()
        self.test_data['components'] = dict()
        self.test_data['scalar parameters'] = dict()

        comps = self.test_data['components']
        sparams = self.test_data['scalar parameters']

        # set scalar params
        sparams['lambda'] = 1200

        # components and index params
        self.cname_to_id = dict()
        self.cname_to_id['salt'] = 0
        self.cname_to_id['lysozyme'] = 1
        self.cname_to_id['cytochrome'] = 2
        self.cname_to_id['ribonuclease'] = 3

        for cid in self.cname_to_id.values():
            comps[cid] = dict()

        # salt
        cid = 0
        comps[cid]['kads'] = 0.0
        comps[cid]['kdes'] = 0.0
        comps[cid]['upsilon'] = 0.0
        comps[cid]['sigma'] = 0.0
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        # lysozyme
        cid = 1
        comps[cid]['kads'] = 35.5
        comps[cid]['kdes'] = 1000.0
        comps[cid]['upsilon'] = 4.7
        comps[cid]['sigma'] = 11.83
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        # cytochrome
        cid = 2
        comps[cid]['kads'] = 1.59
        comps[cid]['kdes'] = 1000.0
        comps[cid]['upsilon'] = 5.29
        comps[cid]['sigma'] = 10.6
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        # ribonuclease
        cid = 3
        comps[cid]['kads'] = 7.7
        comps[cid]['kdes'] = 1000.0
        comps[cid]['upsilon'] = 3.7
        comps[cid]['sigma'] = 10.0
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        filename = 'sma_data.yml'
        with open(filename, 'w') as outfile:
            yaml.dump(self.test_data, outfile, default_flow_style=False)

    def test_parsing_from_dict(self):
        parsed = SMAModel._parse_inputs(self.test_data)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_from_yaml(self):
        parsed = SMAModel._parse_inputs("sma_data.yml")
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_scalar_params(self):
        m = SMAModel(self.test_data)
        self.assertEqual(m.num_scalar_parameters, 3)
        parsed = m.get_scalar_parameters()
        unparsed = self.test_data['scalar parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_components(self):
        m = SMAModel(self.test_data)
        parsed = m.get_index_parameters_dict(with_defaults=True)
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_del_component(self):
        m = SMAModel(self.test_data)
        m.del_component(2)
        parsed = m.get_index_parameters_dict(with_defaults=True)
        del self.test_data['components'][2]
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_set_component_indexed_param(self):
        # TODO: improve this test
        m = SMAModel(self.test_data)
        m.set_component_indexed_param(1, 'kads', 777)
        parsed = m.get_index_parameters_dict(with_defaults=True)
        self.test_data['components'][1]['kads'] = 777
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

        # two parameters
        m.set_component_indexed_param(1, ['kads', 'kdes'], [2, 3])
        parsed = m.get_index_parameters_dict(with_defaults=True)
        self.test_data['components'][1]['kads'] = 2
        self.test_data['components'][1]['kdes'] = 3
        unparsed = self.test_data['components']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_add_component(self):
        # TODO: improve this test
        m = SMAModel(self.test_data)
        m.add_component(4)
        n_p = len(m._registered_index_parameters)
        n_c = len(m._components)
        self.assertEqual(n_p*n_c, m._index_params.size)

    def test_is_kinetic(self):
        m = SMAModel(self.test_data)
        self.assertTrue(m.is_kinetic)
        m.is_kinetic = 0
        self.assertFalse(m.is_kinetic)

    def test_num_components(self):
        m = SMAModel(self.test_data)
        self.assertEqual(4, m.num_components)

    def test_num_index_params(self):
        m = SMAModel(self.test_data)
        self.assertEqual(6, m.num_index_parameters)

    def test_is_salt(self):
        m = SMAModel(self.test_data)
        self.assertTrue(m.is_salt(0))
        m = SMAModel(self.test_data, 1)
        self.assertTrue(m.is_salt(1))
        m = SMAModel(self.test_data, 2)
        self.assertFalse(m.is_salt(1))

    def test_is_fully_specified(self):
        m = SMAModel(self.test_data)
        m.add_component([4, 5])
        self.assertFalse(m.is_fully_specified())
        m = SMAModel(self.test_data)
        self.assertTrue(m.is_fully_specified())

    def test_f_ads(self):

        m = SMAModel(self.test_data)

        c_vars = dict()
        c_vars[0] = 1.0
        c_vars[1] = 0.5
        c_vars[2] = 0.5
        c_vars[3] = 0.5

        q_vars = dict()
        q_vars[0] = 1.0
        q_vars[1] = 0.0
        q_vars[2] = 0.0
        q_vars[3] = 0.0

        q0 = self.test_data['scalar parameters']['lambda']

        for cid in range(1,4):
            vi = self.test_data['components'][cid]['upsilon']
            q0 -= vi*q_vars[cid]

        q0_bar = q0
        for cid in range(1,4):
            si = self.test_data['components'][cid]['sigma']
            q0_bar -= si * q_vars[cid]

        dqidt = dict()
        mdqidt = dict()
        for cid in range(4):
            mdqidt[cid] = m.f_ads(cid, c_vars, q_vars)
            if cid == 0:
                dqidt[cid] = q0
            else:

                vi = self.test_data['components'][cid]['upsilon']
                kads = self.test_data['components'][cid]['kads']
                ads = kads*c_vars[cid] * (q0_bar) ** vi

                kdes = self.test_data['components'][cid]['kdes']
                des = kdes * q_vars[cid] * (c_vars[0]) ** vi
                dqidt[cid] = ads - des

        for cid in range(4):
            self.assertAlmostEqual(dqidt[cid],mdqidt[cid])

    def test_write_to_cadet(self):

        m = SMAModel(self.test_data)

        test_dir = tempfile.mkdtemp()

        filename = os.path.join(test_dir, "sma_tmp.hdf5")
        m.write_to_cadet_input_file(filename, 'unit_001')

        # read back and verify output
        with h5py.File(filename, 'r') as f:
            params = {'kads': 'SMA_KA',
                      'kdes': 'SMA_KD',
                      'upsilon': 'SMA_NU',
                      'sigma': 'SMA_SIGMA'}
            # assumes salt is component 0
            for p, name in params.items():
                path = 'input/model/unit_001/adsorption/{}'.format(name)
                for i,e in enumerate(f[path]):
                    value = self.test_data['components'][i][p]
                    self.assertEqual(value, e)

            is_k = f['input/model/unit_001/adsorption/IS_KINETIC'].value
            self.assertEqual(is_k, m.is_kinetic)

        shutil.rmtree(test_dir)




if __name__ == '__main__':
    unittest.main()

# most of the tests here are actually testing the
# base class. Should move this to a separate testclass




