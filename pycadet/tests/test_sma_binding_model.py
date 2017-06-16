from pycadet.model.binding_model import SMABinding,BindingModel
from pycadet.model.chromatograpy_model import GRModel
from pycadet.utils.compare import equal_dictionaries
from collections import OrderedDict
import unittest
import tempfile
import yaml
import h5py
import shutil
import os


class TestBindingModel(unittest.TestCase):

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
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        # lysozyme
        cid = 'lysozyme'
        comps[cid]['sma_kads'] = 35.5
        comps[cid]['sma_kdes'] = 1000.0
        comps[cid]['sma_nu'] = 4.7
        comps[cid]['sma_sigma'] = 11.83
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        # cytochrome
        cid = 'cytochrome'
        comps[cid]['sma_kads'] = 1.59
        comps[cid]['sma_kdes'] = 1000.0
        comps[cid]['sma_nu'] = 5.29
        comps[cid]['sma_sigma'] = 10.6
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        # ribonuclease
        cid = 'ribonuclease'
        comps[cid]['sma_kads'] = 7.7
        comps[cid]['sma_kdes'] = 1000.0
        comps[cid]['sma_nu'] = 3.7
        comps[cid]['sma_sigma'] = 10.0
        comps[cid]['cref'] = 1.0
        comps[cid]['qref'] = 1.0

        self.m = GRModel(self.test_data)

    def test_is_kinetic(self):
        GRM = self.m
        GRM.binding = SMABinding(self.test_data)
        self.assertTrue(GRM.binding.is_kinetic)
        GRM.binding.is_kinetic = 0
        self.assertFalse(GRM.binding.is_kinetic)


    def test_is_fully_specified(self):
        GRM = self.m
        GRM.binding = SMABinding(self.test_data)
        GRM.add_component('chlorine')
        self.assertFalse(GRM.binding.is_fully_specified())

        GRM = GRModel(self.test_data)
        GRM.binding = SMABinding(self.test_data)
        print(GRM.binding.get_index_parameters())
        self.assertTrue(GRM.binding.is_fully_specified())


    """
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

        m = SMABinding(self.test_data)

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
    """



if __name__ == '__main__':
    unittest.main()

# most of the tests here are actually testing the
# base class. Should move this to a separate testclass




