from pychrom.model.chromatograpy_model import ChromatographyModel, GRModel
from pychrom.model.unit_operation import Inlet, Column
from pychrom.model.section import Section
from pychrom.utils.compare import equal_dictionaries, pprint_dict
from pychrom.model.registrar import Registrar
from collections import OrderedDict
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
        m.write_solver_info_to_cadet_input_file(filename, user_times, **kwargs)

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

            sec_times = np.zeros(m.num_sections, dtype='d')
            for n, sec in m.sections():
                sec_id = sec._section_id
                sec_times[sec_id] = sec.start_time_sec

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

        m.write_connections_to_cadet_input_file(filename, 'load')
        # read back and verify output
        with h5py.File(filename, 'r') as f:
            path = os.path.join("input", "connections")

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



