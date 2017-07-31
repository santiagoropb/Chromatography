from pychrom.core.binding_model import SMABinding
from pychrom.core.chromatograpy_model import GRModel
from pychrom.utils.compare import equal_dictionaries
from collections import OrderedDict
import unittest
import tempfile
import yaml
import h5py
import shutil
import os


class TestBindingModel(unittest.TestCase):

    def setUp(self):

        self.base_model_data = dict()
        self.base_model_data['components'] = ['salt',
                                             'lysozyme',
                                             'cytochrome',
                                             'ribonuclease']

        self.base_model_data['scalar parameters'] = dict()
        self.m = GRModel(data=self.base_model_data)

        self.test_data = dict()
        self.test_data['index parameters'] = OrderedDict()
        self.test_data['scalar parameters'] = dict()

        comps = self.test_data['index parameters']
        sparams = self.test_data['scalar parameters']

        # set scalar params
        sparams['sma_lambda'] = 1200
        sparams['sma_cref'] = 1.0
        sparams['sma_qref'] = 1.0

        # components and index params
        self.comp_names = ['salt',
                      'lysozyme',
                      'cytochrome',
                      'ribonuclease']

        for cname in self.comp_names:
            comps[cname] = dict()

        # salt
        cid = 'salt'
        comps[cid]['sma_ka'] = 0.0
        comps[cid]['sma_kd'] = 0.0
        comps[cid]['sma_nu'] = 0.0
        comps[cid]['sma_sigma'] = 0.0

        # lysozyme
        cid = 'lysozyme'
        comps[cid]['sma_ka'] = 35.5
        comps[cid]['sma_kd'] = 1000.0
        comps[cid]['sma_nu'] = 4.7
        comps[cid]['sma_sigma'] = 11.83

        # cytochrome
        cid = 'cytochrome'
        comps[cid]['sma_ka'] = 1.59
        comps[cid]['sma_kd'] = 1000.0
        comps[cid]['sma_nu'] = 5.29
        comps[cid]['sma_sigma'] = 10.6

        # ribonuclease
        cid = 'ribonuclease'
        comps[cid]['sma_ka'] = 7.7
        comps[cid]['sma_kd'] = 1000.0
        comps[cid]['sma_nu'] = 3.7
        comps[cid]['sma_sigma'] = 10.0

    def test_is_kinetic(self):
        GRM = self.m
        GRM.binding = SMABinding()
        self.assertTrue(GRM.binding.is_kinetic)
        GRM.binding.is_kinetic = 0
        self.assertFalse(GRM.binding.is_kinetic)

    def test_is_fully_specified(self):

        GRM = GRModel(data=self.base_model_data)
        GRM.binding = SMABinding(data=self.test_data)
        self.assertTrue(GRM.binding.is_fully_specified())

    def test_parsing_from_dict(self):

        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        parsed = bm._parse_inputs(self.test_data)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_from_yaml(self):

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "test_data.yml")

        with open(filename, 'w') as outfile:
            yaml.dump(self.test_data, outfile, default_flow_style=False)

        #with open("sma.yml", 'w') as outfile:
        #    yaml.dump(self.test_data, outfile, default_flow_style=False)

        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        parsed = bm._parse_inputs(filename)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

        shutil.rmtree(test_dir)

    def test_parsing_scalar_params(self):
        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        self.assertEqual(bm.num_scalar_parameters, 1)
        parsed = bm.get_scalar_parameters(with_defaults=True)
        unparsed = self.test_data['scalar parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_components(self):
        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        parsed = bm.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        unparsed = self.test_data['index parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_set_index_param(self):

        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        bm.set_index_parameter('lysozyme', 'sma_ka', 777)
        parsed = bm.get_index_parameters(with_defaults=False,
                                        form='dictionary')
        self.test_data['index parameters']['lysozyme']['sma_ka'] = 777
        unparsed = self.test_data['index parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_num_components(self):
        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        self.assertEqual(4, bm.num_components)

    def test_num_index_params(self):
        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        self.assertEqual(4, bm.num_index_parameters)


class TestSMABindingModel(unittest.TestCase):

    def setUp(self):

        self.base_model_data = dict()
        self.base_model_data['components'] = ['salt',
                                              'lysozyme',
                                              'cytochrome',
                                              'ribonuclease']

        self.base_model_data['scalar parameters'] = dict()

        self.test_data = dict()
        self.test_data['index parameters'] = OrderedDict()
        self.test_data['scalar parameters'] = dict()

        comps = self.test_data['index parameters']
        sparams = self.test_data['scalar parameters']

        # set scalar params
        sparams['sma_lambda'] = 1200
        sparams['sma_cref'] = 1.0
        sparams['sma_qref'] = 1200.0

        # components and index params
        self.comp_names = ['salt',
                           'lysozyme',
                           'cytochrome',
                           'ribonuclease']

        for cname in self.comp_names:
            comps[cname] = dict()

        # salt
        cid = 'salt'
        comps[cid]['sma_ka'] = 0.0
        comps[cid]['sma_kd'] = 0.0
        comps[cid]['sma_nu'] = 0.0
        comps[cid]['sma_sigma'] = 0.0

        # lysozyme
        cid = 'lysozyme'
        comps[cid]['sma_ka'] = 35.5
        comps[cid]['sma_kd'] = 1000.0
        comps[cid]['sma_nu'] = 4.7
        comps[cid]['sma_sigma'] = 11.83

        # cytochrome
        cid = 'cytochrome'
        comps[cid]['sma_ka'] = 1.59
        comps[cid]['sma_kd'] = 1000.0
        comps[cid]['sma_nu'] = 5.29
        comps[cid]['sma_sigma'] = 10.6

        # ribonuclease
        cid = 'ribonuclease'
        comps[cid]['sma_ka'] = 7.7
        comps[cid]['sma_kd'] = 1000.0
        comps[cid]['sma_nu'] = 3.7
        comps[cid]['sma_sigma'] = 10.0

        self.m = GRModel(data=self.base_model_data)

    def test_ka(self):
        GRM = GRModel(data=self.base_model_data)
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        cname = 'ribonuclease'
        val = self.test_data['index parameters'][cname]['sma_ka']
        self.assertEqual(bm.ka(cname), val)

    def test_kd(self):
        GRM = GRModel(data=self.base_model_data)
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        cname = 'salt'
        val = self.test_data['index parameters'][cname]['sma_kd']
        self.assertEqual(bm.kd(cname), val)

    def test_nu(self):
        GRM = GRModel(data=self.base_model_data)
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        cname = 'salt'
        val = self.test_data['index parameters'][cname]['sma_nu']
        self.assertEqual(bm.nu(cname), val)

    def test_sigma(self):
        GRM = GRModel(data=self.base_model_data)
        GRM.binding = SMABinding(data=self.test_data)
        bm = GRM.binding
        cname = 'lysozyme'
        val = self.test_data['index parameters'][cname]['sma_sigma']
        self.assertEqual(bm.sigma(cname), val)

    @unittest.skip("")
    def test_f_ads(self):

        GRM = self.m
        GRM.binding = SMABinding(data=self.test_data)
        m = GRM.binding


        GRM.salt = 'salt'

        c_vars = dict()
        c_vars['salt'] = 1.0
        c_vars['lysozyme'] = 0.5
        c_vars['cytochrome'] = 0.5
        c_vars['ribonuclease'] = 0.5

        q_vars = dict()
        q_vars['salt'] = 1.0
        q_vars['lysozyme'] = 0.0
        q_vars['cytochrome'] = 0.0
        q_vars['ribonuclease'] = 0.0

        q0 = self.test_data['scalar parameters']['sma_lambda']

        for cname in self.comp_names:
            if not GRM.is_salt(cname):
                vi = self.test_data['index parameters'][cname]['sma_nu']
                q0 -= vi * q_vars[cname]

        q0_bar = q0
        for cname in self.comp_names:
            if not GRM.is_salt(cname):
                si = self.test_data['index parameters'][cname]['sma_sigma']
                q0_bar -= si * q_vars[cname]

        dqidt = dict()
        mdqidt = dict()
        for cname in self.comp_names:
            mdqidt[cname] = m.f_ads(cname, c_vars, q_vars)
            if GRM.is_salt(cname):
                dqidt[cname] = q0
            else:

                vi = self.test_data['index parameters'][cname]['sma_nu']
                ka = self.test_data['index parameters'][cname]['sma_ka']
                q_ref = self.test_data['scalar parameters']['sma_qref']
                ads = ka * c_vars[cname] * (q0_bar) ** vi

                kd = self.test_data['index parameters'][cname]['sma_kd']
                c_ref = self.test_data['scalar parameters']['sma_cref']
                des = kd * q_vars[cname] * (c_vars[GRM.salt]) ** vi
                dqidt[cname] = ads - des

        for cname in self.comp_names:
            #print(dqidt[cname],mdqidt[cname])
            self.assertAlmostEqual(dqidt[cname], mdqidt[cname])

    def test_write_to_cadet(self):
        GRM = self.m
        GRM.salt = 'salt'
        GRM.binding = SMABinding(data=self.test_data)
        m = GRM.binding

        test_dir = tempfile.mkdtemp()

        filename = os.path.join(test_dir, "sma_tmp.hdf5")
        unit = 'unit_001'
        m._write_to_cadet_input_file(filename, unit)

        path = os.path.join("input", "model", unit, "adsorption")
        # read back and verify output
        with h5py.File(filename, 'r') as f:
            params = {'sma_ka',
                      'sma_kd',
                      'sma_nu',
                      'sma_sigma'}
            # assumes salt is component 0
            for p in params:
                name = p.upper()
                dataset = os.path.join(path, name)
                read = f[dataset]
                for i, e in enumerate(read):
                    comp_id = GRM._ordered_ids_for_cadet[i]
                    comp_name = GRM._comp_id_to_name[comp_id]
                    value = self.test_data['index parameters'][comp_name][p]
                    self.assertEqual(value, e)

            name = 'IS_KINETIC'
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, m.is_kinetic)

        shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main()

# most of the tests here are actually testing the
# base class. Should move this to a separate testclass




