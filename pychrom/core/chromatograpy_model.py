from __future__ import print_function
from pychrom.core.unit_operation import UnitOperation, Column, Inlet, Outlet
from pychrom.core.registrar import Registrar
from pychrom.core.binding_model import BindingModel
from pychrom.core.section import Section
from pychrom.utils import parse_utils
import numpy as np
import warnings
import numbers
import logging
import weakref
import h5py
import copy
import six
import abc
import os

logger = logging.getLogger(__name__)


class ChromatographyModel(abc.ABC):

    def __init__(self, components=None):

        # elements containers
        self._binding_models = list()
        self._sections = list()
        self._units = list()
        self._num_columns = 0

        # define registered params
        self._registered_scalar_parameters = Registrar.scalar_parameters

        # define default values for indexed parameters
        self._default_scalar_params = Registrar.default_scalar_parameters

        # define scalar params container
        self._scalar_params = dict()

        # define components and component indexed parameters
        self._comp_name_to_id = dict()
        self._comp_id_to_name = dict()
        self._components = set()

        self._salt_id = -1

        self._ordered_ids_for_cadet = list()

        self._connections = list()

        # add passed components
        if components is not None:
            for cname in components:
                self.add_component(cname)

    @property
    def salt(self):
        if self._salt_id != -1:
            name = self._comp_id_to_name[self._salt_id]
            return name
        return None

    @salt.setter
    def salt(self, name):
        self._salt_id = self._comp_name_to_id[name]
        self._ordered_ids_for_cadet.remove(self._salt_id)
        self._ordered_ids_for_cadet.insert(0, self._salt_id)

    def is_salt(self, name):

        if name not in self._comp_name_to_id.keys():
            return False
        if self._comp_name_to_id[name] == self._salt_id:
            return True
        return False

    def is_model_component(self, name):
        return name in self._comp_name_to_id.keys()

    def get_component_id(self, comp_name):
        """
        Returns unique id of component
        :param comp_name: name of component
        :return: integer id
        """
        if comp_name not in self._comp_name_to_id.keys():
            raise RuntimeError('{} is not a model component'.format(comp_name))
        return self._comp_name_to_id[comp_name]

    @property
    def num_components(self):
        """
        Returns number of components in the binding model
        """
        return len(self._components)

    @property
    def num_scalar_parameters(self):
        """
        Returns number of scalar parameters
        """
        return len(self.get_scalar_parameters(with_defaults=False))

    @property
    def num_sections(self):
        """
        Returns number of sections
        """
        return len(self._sections)

    @property
    def num_units(self):
        """
        Returns number of units
        """
        return len(self._units)

    @property
    def num_columns(self):
        return self._num_columns

    @property
    def num_binding_models(self):
        """
        Returns number of binding models
        """
        return len(self._binding_models)

    @staticmethod
    def _parse_inputs(inputs):
        """
        Parse inputs
        :param inputs: filename or dictionary with inputs to the binding model
        :return: dictionary with parsed inputs
        """
        return parse_utils.parse_inputs(inputs)

    def _parse_scalar_params(self, dict_inputs):
        """
        Parse scalar parameters
        :param args: distionary with parsed inputs
        :return: None
        """

        sparams = dict_inputs.get('scalar parameters')
        registered_inputs = self._registered_scalar_parameters
        parsed = parse_utils.parse_scalar_inputs_from_dict(sparams,
                                                           self.__class__.__name__,
                                                           registered_inputs)

        for k, v in parsed.items():
            self._scalar_params[k] = v

        for k, val in self._default_scalar_params.items():
            self._scalar_params[k] = val

    def add_component(self, name):
        if name not in self._comp_name_to_id.keys():
            new_id = len(self._components)
            self._comp_name_to_id[name] = new_id
            self._comp_id_to_name[new_id] = name
            self._components.add(new_id)
            self._ordered_ids_for_cadet.append(new_id)

    def _parse_components(self, args):
        """
        Parse components and indexed parameters
        :param args: dictionary with parsed inputs
        :return: None
        """
        list_comp = args.get('components', [])
        for cname in list_comp:
            self.add_component(cname)

    def add_scalar_parameter(self, name, value):
        """
        Add scalar parameter to dict of parameters
        :param name: name(s) of the parameter
        :param value: real number(s)
        :return: None
        """
        if isinstance(name, six.string_types):
            assert name in self._registered_scalar_parameters
            self._scalar_params[name] = value
        elif (isinstance(name, list) or isinstance(name, tuple)) and \
                (isinstance(value, list) or isinstance(value, tuple)):

            for i, n in enumerate(name):
                assert name in self._registered_scalar_parameters
                self._scalar_params[n] = value[i]
        else:
            raise RuntimeError("input not recognized")

    def set_scalar_parameter(self, name, value):

        # TODO: rethink this
        self.add_scalar_parameter(name, value)

    def del_component(self, comp_name):
        """
        Removes component form binding model
        :param comp_name: component id
        :return: None
        """
        warnings.warn("Carfull use for cadet simulations. Not sure will work")
        # TODO: think what happends with order of ids that go to cadet
        comp_id = self._comp_name_to_id[comp_name]
        del self._comp_name_to_id[comp_name]
        del self._comp_id_to_name[comp_id]
        self._components.remove(comp_id)
        self._ordered_ids_for_cadet.remove(comp_id)

    def list_components(self, ids=False):
        """
        Returns list of names for components
        """
        if ids:
            return list(self._comp_id_to_name.keys())
        else:
            return list(self._comp_name_to_id.keys())

    def get_scalar_parameters(self, with_defaults=False):
        """

        :param with_defaults: flag indicating if default parameters must be included
        :return: dictionary of scalar parameters
        """
        if with_defaults:
            return copy.deepcopy(self._scalar_params)
        container = dict()
        for n, v in self._scalar_params.items():
            if n not in self._default_scalar_params.keys():
                container[n] = v
        return container

    def scalar_parameters(self, with_defaults=False):
        if with_defaults:
            for n, v in self._scalar_params.items():
                yield n, v
        else:
            for n, v in self._scalar_params.items():
                if n not in self._default_scalar_params:
                    yield n,v

    def get_scalar_parameter(self, name):
        """

        :param name: name of the scalar parameter
        :return: value of scalar parameter
        """

        return self._scalar_params[name]

    def list_binding_models(self):
        return [n for n in self._binding_models]

    def binding_models(self):
        for n in self._binding_models:
            yield n, getattr(self, n)

    def list_sections(self):
        return [n for n in self._sections]

    def sections(self):
        for n in self._sections:
            yield n, getattr(self, n)

    def list_unit_operations(self, unit_type=None):

        if unit_type is None:
            return [n for n in self._units]
        units = list()
        valid_types = [Column, Inlet, Outlet]
        if unit_type not in valid_types:
            raise RuntimeError("unitType not recognized")
        for n in self._units:
            u = getattr(self, n)
            if isinstance(u, unit_type):
                units.append(n)
        return units

    def unit_operations(self, unit_type=None):
        if unit_type is None:
            for n in self._units:
                yield n, getattr(self, n)
        else:
            valid_types = [Column, Inlet, Outlet]
            if unit_type not in valid_types:
                raise RuntimeError("unitType not recognized")
            for n in self._units:
                u = getattr(self, n)
                if isinstance(u, unit_type):
                    yield n, u

    def _write_connections_to_cadet_input_file(self, filename, active_sec):

        # checking all units connected
        n_connections = len(self._connections)/4
        minimum_con = n_connections*(n_connections-1)/2
        set_connected = set()
        for uid in self._connections:
            if uid >= 0:
                set_connected.add(uid)

        for n, u in self.unit_operations():
            if u._unit_id not in set_connected:
                msg = "Unit {} is not connected".format(n)
                raise RuntimeError(msg)

        with h5py.File(filename, 'a') as f:
            subgroup_name = os.path.join("input", "model", "connections")
            if subgroup_name not in f:
                f.create_group(subgroup_name)
            connections = f[subgroup_name]

            # TODO: ask about this number of switches
            name = 'NSWITCHES'
            pointer = np.array(1, dtype='i')
            connections.create_dataset(name,
                                       data=pointer,
                                       dtype='i')

            switch = connections.create_group("switch_000")

            name = 'CONNECTIONS'
            pointer = np.array(self._connections, dtype='i')
            switch.create_dataset(name,
                                  data=pointer,
                                  dtype='i')

            name = 'SECTION'
            sec = getattr(self, active_sec)
            pointer = np.array(sec._section_id, dtype='i')
            switch.create_dataset(name,
                                  data=pointer,
                                  dtype='i')

    def _write_solver_info_to_cadet_input_file(self, filename, tspan, **kwargs):

        for n, sec in self.sections():
            if np.isnan(sec.start_time_sec):
                msg = """Cannot write file. 
                Section {} does
                not have a start time.
                """.format(n)
                raise RuntimeError(msg)

        n_threads = kwargs.pop('nthreads', Registrar.solver_defaults['nthreads'])

        time_integrator_double_params = dict()
        list_params = ['abstol',
                       'algtol',
                       'init_step_size',
                       'reltol']
        for n in list_params:
            time_integrator_double_params[n] = kwargs.pop(n, Registrar.solver_defaults[n])

        n = 'max_steps'
        time_integrator_int_params = {n: kwargs.pop(n, Registrar.solver_defaults[n])}

        with h5py.File(filename, 'a') as f:
            subgroup_name = os.path.join("input", "solver")

            if subgroup_name not in f:
                f.create_group(subgroup_name)
            solver = f[subgroup_name]

            name = 'NTHREADS'
            pointer = np.array(n_threads, dtype='i')
            solver.create_dataset(name,
                                  data=pointer,
                                  dtype='i')

            name = 'USER_SOLUTION_TIMES'
            if isinstance(tspan, np.ndarray):
                pointer = tspan
            else:
                pointer = np.array(tspan, dtype='d')
            solver.create_dataset(name,
                                  data=pointer,
                                  dtype='d')

            sections = solver.create_group("sections")
            name = 'NSEC'
            v = self.num_sections
            pointer = np.array(v, dtype='i')
            sections.create_dataset(name,
                                    data=pointer,
                                    dtype='i')

            name = 'SECTION_CONTINUITY'
            pointer = np.zeros(self.num_sections-1, 'i')
            sections.create_dataset(name,
                                    data=pointer,
                                    dtype='i')

            # TODO: ask about this
            sec_times = np.zeros(self.num_sections+1, dtype='d')
            for n, sec in self.sections():
                sec_id = sec._section_id
                sec_times[sec_id] = sec.start_time_sec
            sec_times[-1] = tspan[-1]

            name = 'SECTION_TIMES'
            pointer = sec_times
            sections.create_dataset(name,
                                    data=pointer,
                                    dtype='d')

            time_integrator = solver.create_group("time_integrator")

            # write integers
            for n, v in time_integrator_int_params.items():
                name = n.upper()
                pointer = np.array(v, dtype='i')
                time_integrator.create_dataset(name,
                                               data=pointer,
                                               dtype='i')

            # write double
            for n, v in time_integrator_double_params.items():
                name = n.upper()
                pointer = np.array(v, dtype='d')
                time_integrator.create_dataset(name,
                                               data=pointer,
                                               dtype='d')

    def connect_unit_operations(self, name_from, name_to):

        ufrom = getattr(self, name_from)
        uto = getattr(self, name_to)

        ufrom._right_connection = name_to
        uto._left_connection = name_from

        connection = [ufrom.unit_id, uto.unit_id, -1, -1]
        for i in connection:
            self._connections.append(i)

    def pprint(self):

        print("{} component declarations:".format(self.num_components))
        for k, v in self._comp_name_to_id.items():
            print('\t', v, k)

        print("{} binding model declarations:".format(self.num_binding_models))
        for k, v in self.binding_models():
            v.pprint(indent=1)

        print("{} section declarations:".format(self.num_sections))
        for k, v in self.sections():
            v.pprint(indent=1)

        print("{} unit operation declarations:".format(self.num_units))
        print("\n Inlets")
        for k, v in self.unit_operations(unit_type=Inlet):
            v.pprint(indent=2)
        print("\n Columns")
        for k, v in self.unit_operations(unit_type=Column):
            v.pprint(indent=2)
        print("\n Outlets")
        for k, v in self.unit_operations(unit_type=Outlet):
            v.pprint(indent=2)

    def is_fully_specified(self, print_out=False, with_connections=True):

        fully_specified = True
        # check binding models
        for n, e in self.binding_models():
            fully_specified *= e.is_fully_specified(print_out=print_out)

        # check sections
        for n, e in self.sections():
            fully_specified *= e.is_fully_specified(print_out=print_out)

        # check unit operations
        for n, e in self.unit_operations():
            fully_specified *= e.is_fully_specified(print_out=print_out,
                                                    with_connections=with_connections)

        return fully_specified

    def _sort_sections_by_time(self):
        # check sections
        fully_specified = True
        for n, e in self.sections():
            fully_specified *= e.is_fully_specified()
        if fully_specified:
            sec_times = []
            for n, sec in self.sections():
                sec_times.append(sec.start_time_sec)

            # sort the section ids according to time
            sorted_times = sorted(sec_times)
            for n, sec in self.sections():
                t = sec.start_time_sec
                new_id = sorted_times.index(t)
                sec._section_id = new_id
        else:
            msg = "cant sort section unless they are "
            msg += "fully specified"
            raise RuntimeError(msg)

    def _write_to_cadet_input_file(self,
                                  filename,
                                  tspan,
                                  disct_kwargs,
                                  solver_kwargs,
                                  **kwargs):

        # TODO: extend to allow different discretizations for different columns

        active_sec = kwargs.pop('active_set', 'default')
        with_solver = kwargs.pop('with_solver', True)
        with_discretization = kwargs.pop('with_discretization', True)
        concentrations = kwargs.pop('retrive_c', 'in_out')
        sensitivities = kwargs.pop('retrive_sens', 'in_out')
        sol_t = kwargs.pop('sol_times', 'all')

        # check num columns
        if self._num_columns == 0:
            raise RuntimeError("At least one column is required in the model")

        # check num sections
        if self.num_sections == 0:
            raise RuntimeError("At least one section is required in the model")

        # check everything is specified
        self.is_fully_specified(print_out=True)

        # before anything is written make reset sections ids based on time
        self._sort_sections_by_time()

        if with_discretization:
            # check required entries in discretization
            req_entries = ['ncol', 'npar']
            for k in req_entries:
                if k not in disct_kwargs.keys():
                    msg = "{} is required to write cadet file".format(k)
                    msg+= " Please provide {} in disct_kwargs".format(k)
                    raise RuntimeError(msg)
                else:
                    if not isinstance(disct_kwargs[k], numbers.Integral):
                        msg = "{} is not an integer".format(k)
                        raise RuntimeError(msg)

        if active_sec == 'default':
            earliest_time = np.inf
            active_sec = self._sections[0]
            for n, s in self.sections():
                if s.start_time_sec <= earliest_time:
                    earliest_time = s.start_time_sec
                    active_sec = s.name

        # check if file already exists
        if os.path.isfile(filename):
            warnings.warn("Overwriting cadet input file")
            os.remove(filename)

        all_time_opts = ['WRITE_SOLUTION_TIMES',
                         'WRITE_SOLUTION_LAST',
                         'WRITE_SENS_LAST',
                         'SPLIT_COMPONENTS_DATA']

        time_kwargs = dict()
        for o in all_time_opts:
            time_kwargs[o] = 0

        if sol_t == 'all':
            time_kwargs['WRITE_SOLUTION_TIMES'] = 1
        elif sol_t == 'last':
            time_kwargs['WRITE_SOLUTION_LAST'] = 1
            time_kwargs['WRITE_SENS_LAST'] = 1
        else:
            msg = "solution times option not recognizes"
            raise RuntimeError(msg)

        with h5py.File(filename, 'w') as f:
            inp = f.create_group("input")

            ret = inp.create_group("return")

            # write integers
            for n, v in time_kwargs.items():
                name = n.upper()
                pointer = np.array(v, dtype='i')
                ret.create_dataset(name,
                                   data=pointer,
                                   dtype='i')

            model = inp.create_group("model")

            reg_disc = Registrar.discretization_defaults

            inkwargs = dict()
            int_params = ['gs_type',
                          'max_krylov',
                          'max_restarts']

            double_params = ['schur_safety']

            for n in int_params+double_params:
                inkwargs[n] = disct_kwargs.pop(n, reg_disc[n])

            inkwargs['nunits'] = self.num_units

            # check integer parameters
            for n in int_params+['nunits']:
                v = inkwargs[n]
                name = n.upper()
                pointer = np.array(v, dtype='i')
                model.create_dataset(name,
                                     data=pointer,
                                     dtype='i')

            # check integer parameters
            for n in double_params:
                v = inkwargs[n]
                name = n.upper()
                pointer = np.array(v, dtype='d')
                model.create_dataset(name,
                                     data=pointer,
                                     dtype='d')


        # write units
        ncol = 50
        npar = 10
        if with_discretization:
            ncol = disct_kwargs.pop('ncol')
            npar = disct_kwargs.pop('npar')

        for n, u in self.unit_operations():
            u._write_to_cadet_input_file(filename)
            if isinstance(u, Column):
                if with_discretization:
                    u._write_discretization_to_cadet_input_file(filename,
                                                               ncol,
                                                               npar,
                                                               **disct_kwargs)
                u._write_return_to_cadet_input_file(filename,
                                                   concentrations=concentrations,
                                                   sensitivities=sensitivities)

        # write connections
        self._write_connections_to_cadet_input_file(filename, active_sec)

        # solver
        if with_solver:
            self._write_solver_info_to_cadet_input_file(filename,
                                                        tspan,
                                                        **solver_kwargs)

    def __setattr__(self, name, value):

        if name in self.__dict__.keys():
            msg = 'Overwriting of {} not allowed '.format(name)
            msg += 'The attribute has already already created '

            if name in self.list_binding_models():
                msg += 'as a binding model'
                raise RuntimeError(msg)

            if name in self.list_unit_operations():
                msg += 'as a unit operation'
                raise RuntimeError(msg)

            if name in self.list_sections():
                msg += 'as a section'
                raise RuntimeError(msg)

        if isinstance(value, BindingModel):
            value._model = weakref.ref(self)
            value.name = name
            value._initialize_containers()
            self._binding_models.append(name)

        if isinstance(value, Section):
            value._model = weakref.ref(self)
            value._section_id = len(self._sections)
            value.name = name
            value._initialize_containers()
            self._sections.append(name)

        if isinstance(value, UnitOperation):
            value._model = weakref.ref(self)
            value.name = name
            value._unit_id = len(self._units)
            value._initialize_containers()
            self._units.append(name)
            if isinstance(value, Column):
                self._num_columns += 1



        super(ChromatographyModel, self).__setattr__(name, value)

@ChromatographyModel.register
class GRModel(ChromatographyModel):
    def __init__(self, components=None, data=None):

        # call parent binding model constructor
        super().__init__(components=components)

        # parse inputs
        if data is not None:
            args = self._parse_inputs(data)
            self._parse_scalar_params(args)
            self._parse_components(args)


