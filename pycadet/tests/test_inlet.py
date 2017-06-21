from pycadet.model.chromatograpy_model import GRModel
from pycadet.model.section import Section
from pycadet.model.unit_operation import Inlet
from pycadet.utils.compare import equal_dictionaries, pprint_dict
from collections import OrderedDict
import unittest
import tempfile
import yaml
import h5py
import shutil
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


    def test_num_sections(self):

        GRM = self.m
        self.assertEqual(GRM.inlet.num_sections, 0)
        GRM.inlet.add_section('load')
        self.assertEqual(GRM.inlet.num_sections, 1)
        GRM.inlet.add_section('wash')
        self.assertEqual(GRM.inlet.num_sections, 2)
        GRM.inlet.add_section('elute')
        self.assertEqual(GRM.inlet.num_sections, 3)

        #print(GRM.load.get_index_parameters())
        #print(GRM.wash.get_index_parameters())
        #print(GRM.elute.get_index_parameters())
