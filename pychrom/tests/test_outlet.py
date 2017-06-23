from pychrom.model.chromatograpy_model import GRModel
from pychrom.model.unit_operation import Outlet, UnitOperationType
from pychrom.utils.compare import equal_dictionaries, pprint_dict
from collections import OrderedDict
import numpy as np
import unittest
import tempfile
import yaml
import h5py
import shutil
import os


class Testoutlet(unittest.TestCase):

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
        GRM.outlet = Outlet(components=self.model_components)

    def test_unit_type(self):
        outlet = self.m.outlet
        self.assertEqual(outlet._unit_type, UnitOperationType.OUTLET)

    def test_write_to_cadet(self):

        outlet = self.m.outlet

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "outlet_tmp.hdf5")

        outlet.write_to_cadet_input_file(filename)

        # read back and verify output
        with h5py.File(filename, 'r') as f:
            unitname = 'unit_'+str(outlet._unit_id).zfill(3)

            path = os.path.join("input", "model", unitname)

            strings = dict()

            s = str(outlet._unit_type)
            dtype = 'S{}'.format(len(s) + 1)
            pointer = np.array(s, dtype=dtype)
            strings['UNIT_TYPE'] = pointer

            for name, v in strings.items():
                dataset = os.path.join(path, name)
                # check unit type
                print(dataset)
                read = f[dataset].value
                self.assertEqual(read, v)

            # integers
            name = 'NCOMP'
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, outlet.num_components)
