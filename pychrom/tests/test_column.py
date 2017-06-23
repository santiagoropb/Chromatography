from pychrom.model.unit_operation import Column, UnitOperationType
from pychrom.model.chromatograpy_model import GRModel
from pychrom.model.binding_model import SMABinding
from pychrom.model.registrar import Registrar
from pychrom.utils.compare import equal_dictionaries, pprint_dict
from collections import OrderedDict
import numpy as np
import unittest
import tempfile
import yaml
import h5py
import shutil
import os


class TestColumn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_model_data = dict()
        cls.base_model_data['components'] = ['salt',
                                             'lysozyme',
                                             'cytochrome',
                                             'ribonuclease']

        cls.base_model_data['scalar parameters'] = dict()
        cls.m = GRModel(data=cls.base_model_data)

        cls.sma_data = dict()
        cls.sma_data['index parameters'] = OrderedDict()
        cls.sma_data['scalar parameters'] = dict()

        comps = cls.sma_data['index parameters']
        sparams = cls.sma_data['scalar parameters']

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

        GRM = cls.m
        GRM.binding = SMABinding(data=cls.sma_data)

    def test_unit_type(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        self.assertEqual(col._unit_type, UnitOperationType.COLUMN)

    def setUp(self):

        self.test_data = dict()
        self.test_data['index parameters'] = OrderedDict()
        self.test_data['scalar parameters'] = dict()

        comps = self.test_data['index parameters']
        sparams = self.test_data['scalar parameters']

        # set scalar params
        sparams['col_length'] = 0.014
        sparams['col_porosity'] = 0.37
        sparams['par_porosity'] = 0.75
        sparams['par_radius'] = 4.5e-5
        sparams['col_dispersion'] = 5.75e-8
        sparams['velocity'] = 5.75e-4
        sparams['binding'] = 'binding'  # name of the based model binding

        # index parameters
        # components and index params
        self.comp_names = ['salt',
                           'lysozyme',
                           'cytochrome',
                           'ribonuclease']

        for cname in self.comp_names:
            comps[cname] = dict()

        cid = 'salt'
        comps[cid]['init_c'] = 50.0
        comps[cid]['init_q'] = 1200.0
        comps[cid]['film_diffusion'] = 6.9e-6
        comps[cid]['par_diffusion'] = 7e-10
        comps[cid]['par_surfdiffusion'] = 0.0

        cid = 'lysozyme'
        comps[cid]['init_c'] = 0.0
        comps[cid]['init_q'] = 0.0
        comps[cid]['film_diffusion'] = 6.9e-6
        comps[cid]['par_diffusion'] = 6.07e-11
        comps[cid]['par_surfdiffusion'] = 0.0

        cid = 'cytochrome'
        comps[cid]['init_c'] = 0.0
        comps[cid]['init_q'] = 0.0
        comps[cid]['film_diffusion'] = 6.9e-6
        comps[cid]['par_diffusion'] = 6.07e-11
        comps[cid]['par_surfdiffusion'] = 0.0

        cid = 'ribonuclease'
        comps[cid]['init_c'] = 0.0
        comps[cid]['init_q'] = 0.0
        comps[cid]['film_diffusion'] = 6.9e-6
        comps[cid]['par_diffusion'] = 6.07e-11
        comps[cid]['par_surfdiffusion'] = 0.0

    """
    def test_is_fully_specified(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        self.assertTrue(GRM.column.is_fully_specified())
    """

    def test_parsing_from_dict(self):

        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        parsed = col._parse_inputs(self.test_data)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_from_yaml(self):

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "test_column_data.yml")

        with open(filename, 'w') as outfile:
            yaml.dump(self.test_data, outfile, default_flow_style=False)

        #with open("column.yml", 'w') as outfile:
        #    yaml.dump(self.test_data, outfile, default_flow_style=False)

        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        parsed = col._parse_inputs(self.test_data)
        unparsed = self.test_data
        self.assertTrue(equal_dictionaries(parsed, unparsed))

        shutil.rmtree(test_dir)

    def test_parsing_scalar_params(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        self.assertEqual(col.num_scalar_parameters, 7)
        parsed = col.get_scalar_parameters()
        unparsed = self.test_data['scalar parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_components(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        parsed = col.get_index_parameters(with_defaults=True,
                                          form='dictionary')
        unparsed = self.test_data['index parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_set_index_param(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        col.set_index_parameter('lysozyme', 'par_diffusion', 777)
        parsed = col.get_index_parameters(with_defaults=True,
                                          form='dictionary')
        self.test_data['index parameters']['lysozyme']['par_diffusion'] = 777
        unparsed = self.test_data['index parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_num_components(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        self.assertEqual(4, col.num_components)

    def test_init_c(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for cname in col.list_components():
            val = self.test_data['index parameters'][cname]['init_c']
            self.assertEqual(col.init_c(cname), val)

    def test_init_q(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for cname in col.list_components():
            val = self.test_data['index parameters'][cname]['init_q']
            self.assertEqual(col.init_q(cname), val)

    def test_film_diffusion(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for cname in col.list_components():
            val = self.test_data['index parameters'][cname]['film_diffusion']
            self.assertEqual(col.film_diffusion(cname), val)

    def test_par_diffusion(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for cname in col.list_components():
            val = self.test_data['index parameters'][cname]['par_diffusion']
            self.assertEqual(col.par_diffusion(cname), val)

    def test_set_init_c(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for i, cname in enumerate(col.list_components()):
            val = i
            col.set_init_c(cname, val)
            self.assertEqual(GRM.column.init_c(cname), val)

    def test_set_init_q(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for i, cname in enumerate(col.list_components()):
            val = i
            col.set_init_q(cname, val)
            self.assertEqual(GRM.column.init_q(cname), val)

    def test_set_film_diffusion(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for i, cname in enumerate(col.list_components()):
            val = i
            col.set_film_diffusion(cname, val)
            self.assertEqual(GRM.column.film_diffusion(cname), val)

    def test_set_par_diffusion(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        for i, cname in enumerate(col.list_components()):
            val = i
            col.set_par_diffusion(cname, val)
            self.assertEqual(GRM.column.par_diffusion(cname), val)

    def test_dispersion(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        val = self.test_data['scalar parameters']['col_dispersion']
        self.assertEqual(col.dispersion, val)
        col.dispersion =2.0
        self.assertEqual(col.dispersion, 2.0)

    def test_length(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        val = self.test_data['scalar parameters']['col_length']
        self.assertEqual(col.length, val)
        col.length = 2.0
        self.assertEqual(col.length, 2.0)

    def test_particle_porosity(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        val = self.test_data['scalar parameters']['par_porosity']
        self.assertEqual(col.particle_porosity, val)
        col.particle_porosity = 2.0
        self.assertEqual(col.particle_porosity, 2.0)

    def test_velocity(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        val = self.test_data['scalar parameters']['velocity']
        self.assertEqual(col.velocity, val)
        col.velocity = 2.0
        self.assertEqual(col.velocity, 2.0)

    def test_binding_model(self):
        GRM = self.m
        GRM.column = Column(data=self.test_data)
        col = GRM.column
        val = self.test_data['scalar parameters']['binding']
        self.assertEqual(col.binding_model.name, val)

    def test_is_fully_specified(self):
        GRM = self.m
        GRM.salt = 'salt'
        GRM.column = Column(data=self.test_data)
        self.assertTrue(GRM.column.is_fully_specified())

    def test_write_to_cadet_file(self):

        GRM = self.m
        GRM.salt = 'salt'
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        test_dir = tempfile.mkdtemp()

        filename = os.path.join(test_dir, "col_tmp.hdf5")

        col.write_to_cadet_input_file(filename)

        # read back and verify output
        with h5py.File(filename, 'r') as f:
            unitname = 'unit_'+str(col._unit_id).zfill(3)
            path = os.path.join("input", "model", unitname)

            strings = dict()
            strings['UNIT_TYPE'] = b'GENERAL_RATE_MODEL'

            s = str(GRM.binding.binding_type)
            dtype = 'S{}'.format(len(s) + 1)
            pointer = np.array(s, dtype=dtype)
            strings['ADSORPTION_MODEL'] = pointer

            for name, v in strings.items():
                dataset = os.path.join(path, name)
                # check unit type
                read = f[dataset].value
                self.assertEqual(read, v)

            # integers
            name = 'NCOMP'
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, col.num_components)

            # doubles
            list_params = ['col_length',
                           'col_porosity',
                           'par_porosity',
                           'par_radius',
                           'col_dispersion',
                           'velocity']

            for p in list_params:
                name = p.upper()
                dataset = os.path.join(path, name)
                read = f[dataset].value
                self.assertEqual(read, self.test_data['scalar parameters'][p])

            # index parameters
            # doubles
            list_params = ['init_c',
                           'init_q',
                           'film_diffusion',
                           'par_diffusion',
                           'par_surfdiffusion']

            for p in list_params:
                name = p.upper()
                dataset = os.path.join(path, name)
                read = f[dataset]
                for i, e in enumerate(read):
                    comp_id = GRM._ordered_ids_for_cadet[i]
                    comp_name = GRM._comp_id_to_name[comp_id]
                    value = self.test_data['index parameters'][comp_name][p]
                    self.assertEqual(value, e)

    def test_write_return_to_cadet_file(self):

        GRM = self.m
        GRM.salt = 'salt'
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        test_dir = tempfile.mkdtemp()

        all_datasets = ['WRITE_SOLUTION_COLUMN_INLET',
                        'WRITE_SOLUTION_COLUMN_OUTLET',
                        'WRITE_SOLUTION_COLUMN',
                        'WRITE_SOLUTION_PARTICLE',
                        'WRITE_SOLUTION_FLUX',
                        'WRITE_SOLDOT_COLUMN_INLET',
                        'WRITE_SOLDOT_COLUMN_OUTLET',
                        'WRITE_SOLDOT_COLUMN',
                        'WRITE_SOLDOT_PARTICLE',
                        'WRITE_SOLDOT_FLUX',
                        'WRITE_SENS_COLUMN_INLET',
                        'WRITE_SENS_COLUMN_OUTLET',
                        'WRITE_SENS_COLUMN',
                        'WRITE_SENS_PARTICLE',
                        'WRITE_SENS_FLUX',
                        'WRITE_SENSDOT_COLUMN_INLET',
                        'WRITE_SENSDOT_COLUMN_OUTLET',
                        'WRITE_SENSDOT_COLUMN',
                        'WRITE_SENSDOT_PARTICLE',
                        'WRITE_SENSDOT_FLUX']

        filename = os.path.join(test_dir, "col_return_tmp1.hdf5")
        col.write_return_to_cadet_input_file(filename)

        # test with defaults
        # read back and verify output
        with h5py.File(filename, 'r') as f:
            unitname = 'unit_' + str(col._unit_id).zfill(3)
            path = os.path.join("input", "return", unitname)

            data_sets_dict = dict()
            for n in all_datasets:
                data_sets_dict[n] = 0

            list_inputs_c = ['WRITE_SOLUTION_COLUMN_INLET',
                             'WRITE_SOLUTION_COLUMN_OUTLET']

            list_inputs_s = ['WRITE_SENS_COLUMN_INLET',
                             'WRITE_SENS_COLUMN_OUTLET']

            for n in list_inputs_c:
                data_sets_dict[n] = 1

            for n in list_inputs_s:
                data_sets_dict[n] = 1

            for p, v in data_sets_dict.items():
                name = p.upper()
                dataset = os.path.join(path, name)
                read = f[dataset].value
                self.assertEqual(read, v)

        # test with defaults
        filename = os.path.join(test_dir, "col_return_tmp2.hdf5")
        col.write_return_to_cadet_input_file(filename,
                                             concentrations='all',
                                             sensitivities='all')

        with h5py.File(filename, 'r') as f:
            unitname = 'unit_' + str(col._unit_id).zfill(3)
            path = os.path.join("input", "return", unitname)

            data_sets_dict = dict()
            for n in all_datasets:
                data_sets_dict[n] = 0

            list_inputs_c = ['WRITE_SOLUTION_COLUMN',
                             'WRITE_SOLUTION_PARTICLE',
                             'WRITE_SOLUTION_FLUX']

            list_inputs_s = ['WRITE_SENS_COLUMN',
                             'WRITE_SENS_PARTICLE',
                             'WRITE_SENS_FLUX']

            for n in list_inputs_c:
                data_sets_dict[n] = 1

            for n in list_inputs_s:
                data_sets_dict[n] = 1

            for p, v in data_sets_dict.items():
                name = p.upper()
                dataset = os.path.join(path, name)
                read = f[dataset].value
                self.assertEqual(read, v)

    def test_write_discretization_to_cadet_input_file(self):

        GRM = self.m
        GRM.salt = 'salt'
        GRM.column = Column(data=self.test_data)
        col = GRM.column

        test_dir = tempfile.mkdtemp()

        filename = os.path.join(test_dir, "col_disc_tmp.hdf5")

        kwargs = dict()
        reg_disc = Registrar.discretization_defaults
        reg_weno = Registrar.weno_defaults


        int_params = ['use_analytic_jacobian',
                      'gs_type',
                      'max_krylov',
                      'max_restarts']

        int_weno = ['boundary_model',
                    'weno_order']

        double_params = ['schur_safety']

        double_weno = ['weno_eps']

        string_params = ['par_disc_type']

        for n in int_params+double_params+string_params:
            kwargs[n] = reg_disc[n]

        for n in int_weno+double_weno:
            kwargs[n] = reg_weno[n]

        ncol = 50
        npar = 10
        col.write_discretization_to_cadet_input_file(filename,
                                                     ncol,
                                                     npar,
                                                     **kwargs)

        with h5py.File(filename, 'r') as f:
            unitname = 'unit_' + str(col._unit_id).zfill(3)
            path = os.path.join("input", "model", unitname, "discretization")

            # check integer parameters
            for n in int_params:
                name = n.upper()
                v = kwargs[n]
                dataset = os.path.join(path, name)
                read = f[dataset].value
                self.assertEqual(read, v)

            name = 'NCOL'
            v = ncol
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, v)

            name = 'NPAR'
            v = npar
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, v)

            # check double parameters
            for n in double_params:
                name = n.upper()
                v = kwargs[n]
                dataset = os.path.join(path, name)
                read = f[dataset].value
                self.assertEqual(read, v)

            # check integer parameters
            for n in string_params:
                name = n.upper()
                v = kwargs[n].encode()
                dataset = os.path.join(path, name)
                read = f[dataset].value
                self.assertEqual(read, v)

            #check weno parameters
            for n in int_weno:
                name = n.upper()
                v = kwargs[n]
                dataset = os.path.join(path, "weno", name)
                read = f[dataset].value
                self.assertEqual(read, v)

            # check weno parameters
            for n in double_weno:
                name = n.upper()
                v = kwargs[n]
                dataset = os.path.join(path, "weno", name)
                read = f[dataset].value
                self.assertEqual(read, v)

















