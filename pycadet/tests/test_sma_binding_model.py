from pycadet.model.binding_model import SMABinding
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
        self.comp_names = ['salt',
                      'lysozyme',
                      'cytochrome',
                      'ribonuclease']

        for cname in self.comp_names:
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
        GRM.binding = SMABinding()
        self.assertTrue(GRM.binding.is_kinetic)
        GRM.binding.is_kinetic = 0
        self.assertFalse(GRM.binding.is_kinetic)

    def test_is_fully_specified(self):
        GRM = self.m
        GRM.binding = SMABinding()
        GRM.add_component('chlorine')
        self.assertFalse(GRM.binding.is_fully_specified())

        GRM = GRModel(self.test_data)
        GRM.binding = SMABinding()
        self.assertTrue(GRM.binding.is_fully_specified())

    def test_f_ads(self):
        GRM = self.m
        GRM.binding = SMABinding()
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
                vi = self.test_data['components'][cname]['sma_nu']
                q0 -= vi*q_vars[cname]

        q0_bar = q0
        for cname in self.comp_names:
            if not GRM.is_salt(cname):
                si = self.test_data['components'][cname]['sma_sigma']
                q0_bar -= si * q_vars[cname]

        dqidt = dict()
        mdqidt = dict()
        for cname in self.comp_names:
            mdqidt[cname] = m.f_ads(cname, c_vars, q_vars)
            if GRM.is_salt(cname):
                dqidt[cname] = q0
            else:

                vi = self.test_data['components'][cname]['sma_nu']
                kads = self.test_data['components'][cname]['sma_kads']
                ads = kads*c_vars[cname] * (q0_bar) ** vi

                kdes = self.test_data['components'][cname]['sma_kdes']
                des = kdes * q_vars[cname] * (c_vars[GRM.salt]) ** vi
                dqidt[cname] = ads - des

        for cname in self.comp_names:
            self.assertAlmostEqual(dqidt[cname], mdqidt[cname])

    def test_write_to_cadet(self):
        GRM = self.m
        GRM.salt = 'salt'
        GRM.binding = SMABinding()
        m = GRM.binding

        test_dir = tempfile.mkdtemp()

        filename = os.path.join(test_dir, "sma_tmp.hdf5")
        unit = 'unit_001'
        m.write_to_cadet_input_file(filename,unit)

        # read back and verify output
        with h5py.File(filename, 'r') as f:
            params = {'sma_kads',
                      'sma_kdes',
                      'sma_nu',
                      'sma_sigma'}
            # assumes salt is component 0
            for p in params:
                name = p.upper()
                path = 'input/model/unit_001/adsorption/{}'.format(name)
                for i, e in enumerate(f[path]):
                    comp_id = GRM._ordered_ids_for_cadet[i]
                    comp_name = GRM._comp_id_to_name[comp_id]
                    value = self.test_data['components'][comp_name][p]
                    self.assertEqual(value, e)

            is_k = f['input/model/unit_001/adsorption/IS_KINETIC'].value
            self.assertEqual(is_k, m.is_kinetic)

        shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()

# most of the tests here are actually testing the
# base class. Should move this to a separate testclass




