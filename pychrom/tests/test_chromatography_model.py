from pychrom.model.chromatograpy_model import ChromatographyModel, GRModel
from pychrom.model.unit_operation import Inlet, Column, Outlet
from pychrom.model.binding_model import SMABinding
from pychrom.model.section import Section
from pychrom.utils.compare import equal_dictionaries, pprint_dict
from pychrom.model.registrar import Registrar
import numpy as np
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
        m = GRModel(data=self.test_data)
        self.assertEqual(m.num_scalar_parameters, 1)
        parsed = m.get_scalar_parameters()
        unparsed = self.test_data['scalar parameters']
        self.assertTrue(equal_dictionaries(parsed, unparsed))

    def test_parsing_components(self):
        m = GRModel(data=self.test_data)
        inner = m.list_components()
        outer = set(self.test_data['components'])
        self.assertEqual(m.num_components, len(outer))
        for c in inner:
            self.assertTrue(c in outer)

    @unittest.skip("ignored for now")
    def test_del_component(self):
        m = GRModel(data=self.test_data)
        m.del_component('lysozyme')
        self.assertEqual(m.num_components, 3)
        self.assertFalse('lisozome' in m.list_components())

    def test_add_component(self):
        m = GRModel(data=self.test_data)
        m.add_component('chlorine')
        self.assertEqual(m.num_components, 5)
        self.assertTrue('chlorine' in m.list_components())
        self.assertTrue('chlorine' in m._comp_name_to_id.keys())

    def test_num_components(self):
        m = GRModel(data=self.test_data)
        self.assertEqual(4, m.num_components)

    def test_is_salt(self):
        m = GRModel(data=self.test_data)
        m.salt = 'salt'
        self.assertTrue(m.is_salt('salt'))
        self.assertFalse(m.is_salt('blah'))

    def test_solver_info_to_cadet_input(self):
        m = GRModel(data=self.test_data)

        comps = self.test_data['components']
        m.load = Section(components=comps)
        for cname in comps:
            m.load.set_a0(cname, 1.0)
        m.load.set_a0('salt', 50.0)
        m.load.start_time_sec = 0.0

        m.wash = Section(components=comps)
        m.wash.set_a0('salt', 50.0)
        m.wash.start_time_sec = 1.0

        m.elute = Section(components=comps)
        m.elute.set_a0('salt', 100.0)
        m.elute.set_a1('salt', 0.2)
        m.elute.start_time_sec = 3.0

        reg_solver = Registrar.solver_defaults

        n_threads = reg_solver['nthreads']
        kwargs = dict()
        kwargs['nthreads'] = n_threads
        user_times = range(10)

        double_params = ['abstol',
                         'algtol',
                         'init_step_size',
                         'reltol']

        for n in double_params:
            kwargs[n] = reg_solver[n]

        int_params = ['max_steps']
        for n in int_params:
            kwargs[n] = reg_solver[n]

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "solver_tmp.hdf5")
        m._write_solver_info_to_cadet_input_file(filename, user_times, **kwargs)

        # read back and verify output
        with h5py.File(filename, 'r') as f:
            path = os.path.join("input", "solver")

            # integers
            name = 'NTHREADS'
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, kwargs['nthreads'])

            # user times
            name = 'USER_SOLUTION_TIMES'
            dataset = os.path.join(path, name)
            read = f[dataset]
            for i, t in enumerate(read):
                self.assertEqual(user_times[i], t)

            # sections
            name = 'NSEC'
            dataset = os.path.join(path, "sections", name)
            read = f[dataset].value
            self.assertEqual(read, m.num_sections)

            name = 'SECTION_CONTINUITY'
            pointer = np.zeros(m.num_sections - 1, 'i')
            dataset = os.path.join(path, "sections", name)
            read = f[dataset]
            for i, t in enumerate(read):
                self.assertEqual(pointer[i], t)

            sec_times = np.zeros(m.num_sections+1, dtype='d')
            for n, sec in m.sections():
                sec_id = sec._section_id
                sec_times[sec_id] = sec.start_time_sec
            sec_times[-1] = user_times[-1]

            name = 'SECTION_TIMES'
            dataset = os.path.join(path, "sections", name)
            read = f[dataset]
            for i, t in enumerate(read):
                self.assertEqual(sec_times[i], t)

            # time integrator
            for n in int_params:
                name = n.upper()
                dataset = os.path.join(path, "time_integrator", name)
                read = f[dataset].value
                self.assertEqual(read, kwargs[n])

            for n in double_params:
                name = n.upper()
                dataset = os.path.join(path, "time_integrator", name)
                read = f[dataset].value
                self.assertEqual(read, kwargs[n])


    def test_write_connections_to_cadet_input(self):

        m = GRModel(data=self.test_data)

        comps = self.test_data['components']
        m.load = Section(components=comps)
        for cname in comps:
            m.load.set_a0(cname, 1.0)
        m.load.set_a0('salt', 50.0)
        m.load.start_time_sec = 0.0

        m.wash = Section(components=comps)
        m.wash.set_a0('salt', 50.0)
        m.wash.start_time_sec = 1.0

        m.elute = Section(components=comps)
        m.elute.set_a0('salt', 100.0)
        m.elute.set_a1('salt', 0.2)
        m.elute.start_time_sec = 3.0

        m.inlet = Inlet(components=comps)
        m.column = Column(components=comps)
        m.connect_unit_operations('inlet', 'column')

        # TODO: add check number of connections is at least the minimum
        if not m.num_units:
            msg = "Cannot write connections. There is no Units"
            raise RuntimeError(msg)

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "connections_tmp.hdf5")

        m._write_connections_to_cadet_input_file(filename, 'load')
        # read back and verify output
        with h5py.File(filename, 'r') as f:
            path = os.path.join("input", "model", "connections")

            # switches
            # TODO: verify how is this computed
            name = 'NSWITCHES'
            dataset = os.path.join(path, name)
            read = f[dataset].value
            self.assertEqual(read, 1)

            name = 'CONNECTIONS'
            dataset = os.path.join(path, "switch_000", name)
            read = f[dataset]
            for i, c in enumerate(read):
                self.assertEqual(c, m._connections[i])

            name = 'SECTION'
            sec_id = m.load._section_id
            dataset = os.path.join(path, "switch_000", name)
            read = f[dataset].value
            self.assertEqual(read, sec_id)

    def test_write_to_cadet_input_file(self):

        comps = self.test_data['components']

        GRM = GRModel(components=comps)
        GRM.salt = 'salt'

        GRM.load = Section(components=comps)

        for cname in comps:
            GRM.load.set_a0(cname, 1.0)
        GRM.load.set_a0('salt', 50.0)
        GRM.load.start_time_sec = 0.0

        GRM.wash = Section(components=comps)
        GRM.wash.set_a0('salt', 50.0)
        GRM.wash.start_time_sec = 10.0

        GRM.elute = Section(components=comps)
        GRM.elute.set_a0('salt', 100.0)
        GRM.elute.set_a1('salt', 0.2)
        GRM.elute.start_time_sec = 90.0

        GRM.inlet = Inlet(components=comps)
        GRM.inlet.add_section('load')
        GRM.inlet.add_section('wash')
        GRM.inlet.add_section('elute')

        sma_data = dict()
        sma_data['index parameters'] = dict()
        sma_data['scalar parameters'] = dict()

        # set scalar params
        sma_data['scalar parameters']['sma_lambda'] = 1200

        # set components and index params

        for cname in comps:
            sma_data['index parameters'][cname] = dict()

        # salt
        cid = 'salt'
        sma_data['index parameters'][cid]['sma_ka'] = 0.0
        sma_data['index parameters'][cid]['sma_kd'] = 0.0
        sma_data['index parameters'][cid]['sma_nu'] = 0.0
        sma_data['index parameters'][cid]['sma_sigma'] = 0.0

        # lysozyme
        cid = 'lysozyme'
        sma_data['index parameters'][cid]['sma_ka'] = 35.5
        sma_data['index parameters'][cid]['sma_kd'] = 1000.0
        sma_data['index parameters'][cid]['sma_nu'] = 4.7
        sma_data['index parameters'][cid]['sma_sigma'] = 11.83

        # cytochrome
        cid = 'cytochrome'
        sma_data['index parameters'][cid]['sma_ka'] = 1.59
        sma_data['index parameters'][cid]['sma_kd'] = 1000.0
        sma_data['index parameters'][cid]['sma_nu'] = 5.29
        sma_data['index parameters'][cid]['sma_sigma'] = 10.6

        # ribonuclease
        cid = 'ribonuclease'
        sma_data['index parameters'][cid]['sma_ka'] = 7.7
        sma_data['index parameters'][cid]['sma_kd'] = 1000.0
        sma_data['index parameters'][cid]['sma_nu'] = 3.7
        sma_data['index parameters'][cid]['sma_sigma'] = 10.0

        GRM.binding = SMABinding(data=sma_data)

        column_data = dict()
        column_data['index parameters'] = dict()
        column_data['scalar parameters'] = dict()

        # set scalar params
        column_data['scalar parameters']['col_length'] = 0.014
        column_data['scalar parameters']['col_porosity'] = 0.37
        column_data['scalar parameters']['par_porosity'] = 0.75
        column_data['scalar parameters']['par_radius'] = 4.5e-5
        column_data['scalar parameters']['col_dispersion'] = 5.75e-8
        column_data['scalar parameters']['velocity'] = 5.75e-4
        column_data['scalar parameters']['binding'] = 'binding'  # name of the based model binding

        # index parameters
        for cname in comps:
            column_data['index parameters'][cname] = dict()

        # components and index params
        cid = 'salt'
        column_data['index parameters'][cid]['init_c'] = 50.0
        column_data['index parameters'][cid]['init_q'] = 1200.0
        column_data['index parameters'][cid]['film_diffusion'] = 6.9e-6
        column_data['index parameters'][cid]['par_diffusion'] = 7e-10
        column_data['index parameters'][cid]['par_surfdiffusion'] = 0.0

        cid = 'lysozyme'
        column_data['index parameters'][cid]['init_c'] = 0.0
        column_data['index parameters'][cid]['init_q'] = 0.0
        column_data['index parameters'][cid]['film_diffusion'] = 6.9e-6
        column_data['index parameters'][cid]['par_diffusion'] = 6.07e-11
        column_data['index parameters'][cid]['par_surfdiffusion'] = 0.0

        cid = 'cytochrome'
        column_data['index parameters'][cid]['init_c'] = 0.0
        column_data['index parameters'][cid]['init_q'] = 0.0
        column_data['index parameters'][cid]['film_diffusion'] = 6.9e-6
        column_data['index parameters'][cid]['par_diffusion'] = 6.07e-11
        column_data['index parameters'][cid]['par_surfdiffusion'] = 0.0

        cid = 'ribonuclease'
        column_data['index parameters'][cid]['init_c'] = 0.0
        column_data['index parameters'][cid]['init_q'] = 0.0
        column_data['index parameters'][cid]['film_diffusion'] = 6.9e-6
        column_data['index parameters'][cid]['par_diffusion'] = 6.07e-11
        column_data['index parameters'][cid]['par_surfdiffusion'] = 0.0

        GRM.column = Column(data=column_data)

        GRM.outlet = Outlet(components=comps)

        GRM.connect_unit_operations('inlet', 'column')
        GRM.connect_unit_operations('column', 'outlet')

        disct_kwargs = dict()
        disct_kwargs['ncol'] = 50
        disct_kwargs['npar'] = 10

        tspan = range(150)

        test_dir = tempfile.mkdtemp()
        filename = os.path.join(test_dir, "first_model.h5")

        GRM.write_to_cadet_input_file(filename,
                                      tspan,
                                      disct_kwargs,
                                      dict())

        datasets = ['input',
                    'input/model',
                    'input/model/GS_TYPE',
                    'input/model/MAX_KRYLOV',
                    'input/model/MAX_RESTARTS',
                    'input/model/NUNITS',
                    'input/model/SCHUR_SAFETY',
                    'input/model/connections',
                    'input/model/connections/NSWITCHES',
                    'input/model/connections/switch_000',
                    'input/model/connections/switch_000/CONNECTIONS',
                    'input/model/connections/switch_000/SECTION',
                    'input/model/unit_000',
                    'input/model/unit_000/INLET_TYPE',
                    'input/model/unit_000/NCOMP',
                    'input/model/unit_000/UNIT_TYPE',
                    'input/model/unit_000/sec_000',
                    'input/model/unit_000/sec_000/CONST_COEFF',
                    'input/model/unit_000/sec_000/CUBE_COEFF',
                    'input/model/unit_000/sec_000/LIN_COEFF',
                    'input/model/unit_000/sec_000/QUAD_COEFF',
                    'input/model/unit_000/sec_001',
                    'input/model/unit_000/sec_001/CONST_COEFF',
                    'input/model/unit_000/sec_001/CUBE_COEFF',
                    'input/model/unit_000/sec_001/LIN_COEFF',
                    'input/model/unit_000/sec_001/QUAD_COEFF',
                    'input/model/unit_000/sec_002',
                    'input/model/unit_000/sec_002/CONST_COEFF',
                    'input/model/unit_000/sec_002/CUBE_COEFF',
                    'input/model/unit_000/sec_002/LIN_COEFF',
                    'input/model/unit_000/sec_002/QUAD_COEFF',
                    'input/model/unit_001',
                    'input/model/unit_001/ADSORPTION_MODEL',
                    'input/model/unit_001/COL_DISPERSION',
                    'input/model/unit_001/COL_LENGTH',
                    'input/model/unit_001/COL_POROSITY',
                    'input/model/unit_001/FILM_DIFFUSION',
                    'input/model/unit_001/INIT_C',
                    'input/model/unit_001/INIT_Q',
                    'input/model/unit_001/NCOMP',
                    'input/model/unit_001/PAR_DIFFUSION',
                    'input/model/unit_001/PAR_POROSITY',
                    'input/model/unit_001/PAR_RADIUS',
                    'input/model/unit_001/PAR_SURFDIFFUSION',
                    'input/model/unit_001/UNIT_TYPE',
                    'input/model/unit_001/VELOCITY',
                    'input/model/unit_001/adsorption',
                    'input/model/unit_001/adsorption/IS_KINETIC',
                    'input/model/unit_001/adsorption/SMA_KA',
                    'input/model/unit_001/adsorption/SMA_KD',
                    'input/model/unit_001/adsorption/SMA_LAMBDA',
                    'input/model/unit_001/adsorption/SMA_NU',
                    'input/model/unit_001/adsorption/SMA_SIGMA',
                    'input/model/unit_001/discretization',
                    'input/model/unit_001/discretization/GS_TYPE',
                    'input/model/unit_001/discretization/MAX_KRYLOV',
                    'input/model/unit_001/discretization/MAX_RESTARTS',
                    'input/model/unit_001/discretization/NBOUND',
                    'input/model/unit_001/discretization/NCOL',
                    'input/model/unit_001/discretization/NPAR',
                    'input/model/unit_001/discretization/PAR_DISC_TYPE',
                    'input/model/unit_001/discretization/SCHUR_SAFETY',
                    'input/model/unit_001/discretization/USE_ANALYTIC_JACOBIAN',
                    'input/model/unit_001/discretization/weno',
                    'input/model/unit_001/discretization/weno/BOUNDARY_MODEL',
                    'input/model/unit_001/discretization/weno/WENO_EPS',
                    'input/model/unit_001/discretization/weno/WENO_ORDER',
                    'input/model/unit_002',
                    'input/model/unit_002/NCOMP',
                    'input/model/unit_002/UNIT_TYPE',
                    'input/return',
                    'input/return/WRITE_SOLUTION_TIMES',
                    'input/return/unit_001',
                    'input/return/unit_001/WRITE_SENS_COLUMN',
                    'input/return/unit_001/WRITE_SENS_COLUMN_INLET',
                    'input/return/unit_001/WRITE_SENS_COLUMN_OUTLET',
                    'input/return/unit_001/WRITE_SENS_FLUX',
                    'input/return/unit_001/WRITE_SENS_PARTICLE',
                    'input/return/unit_001/WRITE_SOLUTION_COLUMN',
                    'input/return/unit_001/WRITE_SOLUTION_COLUMN_INLET',
                    'input/return/unit_001/WRITE_SOLUTION_COLUMN_OUTLET',
                    'input/return/unit_001/WRITE_SOLUTION_FLUX',
                    'input/return/unit_001/WRITE_SOLUTION_PARTICLE',
                    'input/solver',
                    'input/solver/NTHREADS',
                    'input/solver/USER_SOLUTION_TIMES',
                    'input/solver/sections',
                    'input/solver/sections/NSEC',
                    'input/solver/sections/SECTION_CONTINUITY',
                    'input/solver/sections/SECTION_TIMES',
                    'input/solver/time_integrator',
                    'input/solver/time_integrator/ABSTOL',
                    'input/solver/time_integrator/ALGTOL',
                    'input/solver/time_integrator/INIT_STEP_SIZE',
                    'input/solver/time_integrator/MAX_STEPS',
                    'input/solver/time_integrator/RELTOL']

        with h5py.File(filename, 'r') as f:
            for ds in datasets:
                self.assertTrue(ds in f)



