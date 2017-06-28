from pychrom.core.chromatograpy_model import GRModel
from pychrom.core.unit_operation import Column
from pychrom.core.registrar import Registrar
import matplotlib.pyplot as plt
from pychrom.modeling.results_object import ResultsDataSet
import shutil
import xarray as xr
import numpy as np
import subprocess
import warnings
import logging
import tempfile
import h5py
import uuid
import os

logger = logging.getLogger(__name__)

class CadetModeler(object):

    def __init__(self, model):
        """

        :param model: GRModel
        """
        self._model = model
        self._discretizations = dict()

    def discretize_column(self, column_name, ncol, npar, **kwargs):
        """

        :param ncol:
        :param npar:
        :param kwargs:
        :return:
        """

        if column_name in self._model.list_unit_operations(unit_type=Column):
            if column_name not in self._discretizations.keys():
                self._discretizations[column_name] = dict()
            self._discretizations[column_name]['ncol'] = ncol
            self._discretizations[column_name]['npar'] = npar
            for k, v in kwargs.items():
                if k in Registrar.discretization_parameters:
                    self._discretizations[column_name][k] = v
                else:
                    msg = "Ignoring discretization parameter {}".format(k)
                    msg += " because it is not recognized in pychrom"
                    warnings.warn(msg)
        else:
            msg = "{} is not a column of the Chromatography model".format(column_name)
            raise RuntimeError(msg)

    def list_discretized_columns(self):
        return [n for n in self._discretizations.keys()]

    def run_sim(self,
                tspan,
                retrive_c='in_out',
                retrive_sens='in_out',
                sol_times='all',
                keep_files=False,
                **kwargs):
        """

        :param retrive:
        :param keep_file:
        :param kwargs:
        :return:
        """

        #TODO: add timing flag

        if not keep_files:
            test_dir = tempfile.mkdtemp()
            filename = os.path.join(test_dir, "intmp.h5")
            outfilename = os.path.join(test_dir, "soltmp.h5")
        else:
            filename = 'in' + str(uuid.uuid4()) + '.h5'
            outfilename = 'sol' + str(uuid.uuid4()) + '.h5'

        # write the input file
        self.write_cadet_file(filename,
                              tspan,
                              retrive_c=retrive_c,
                              retrive_sens=retrive_sens,
                              sol_times=sol_times,
                              **kwargs)

        logger.info('wrote file successfully')

        cadet_location = 'cadet-cli'
        if shutil.which(cadet_location) is None:
            raise RuntimeError('Cadet executable not found. Add cadet-cli to path')

        cmd = [cadet_location, filename, outfilename]
        proc = subprocess.Popen(cmd,
                                bufsize=0,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.poll() != 0:
            raise RuntimeError(stderr)

        logger.info('run simulation successfully')

        return self._parse_results(outfilename)

    def write_cadet_file(self,
                         filename,
                         tspan,
                         retrive_c='in_out',
                         retrive_sens='in_out',
                         sol_times='all',
                         **kwargs):

        # check if all columns have been discretized
        for column_name in self._model.list_unit_operations(unit_type=Column):
            if column_name not in self.list_discretized_columns():
                msg = "{} has not been discretized".format(column_name)
                raise RuntimeError(msg)

        # this has to be extendended in results object
        if sol_times != 'all':
            raise NotImplementedError("partial times not supported yet")

        reg_disc = Registrar.discretization_defaults
        disct_kwargs = dict()
        params = ['gs_type',
                  'max_krylov',
                  'max_restarts',
                  'schur_safety']

        for n in params:
            disct_kwargs[n] = kwargs.pop(n, reg_disc[n])

        self._model._write_to_cadet_input_file(filename,
                                               tspan,
                                               disct_kwargs,
                                               kwargs,
                                               with_discretization=False,
                                               retrive_c=retrive_c,
                                               retrive_sens=retrive_sens,
                                               sol_times=sol_times)

        for n, u in self._model.unit_operations(unit_type=Column):
            ncol = self._discretizations[n]['ncol']
            npar = self._discretizations[n]['npar']
            disct_dict = dict()
            for k,v in self._discretizations.items():
                if k not in ['npar', 'ncol']:
                    disct_dict[k] = v
            u._write_discretization_to_cadet_input_file(filename,
                                                       ncol,
                                                       npar,
                                                       **disct_dict)

    def _parse_results(self, filename):

        # for each column it returns a result dataset
        results = dict()
        first_name = ''
        with h5py.File(filename) as f:

            subgroup_name = os.path.join("output", "solution")
            solutions = f[subgroup_name]

            times = solutions['SOLUTION_TIMES']
            # loop through the columns
            for n, col in self._model.unit_operations(unit_type=Column):

                uname = 'unit_' + str(col.unit_id).zfill(3)
                if uname in solutions:

                    result_set = ResultsDataSet()
                    # get list of components
                    ncomp = col.num_components
                    component_names = []
                    for i in range(ncomp):
                        comp_id = self._model._ordered_ids_for_cadet[i]
                        comp_name = self._model._comp_id_to_name[comp_id]
                        component_names.append(comp_name)

                    result_set.components = np.array(component_names)

                    # get times
                    load_times = np.zeros(times.size)
                    times.read_direct(load_times)
                    result_set.times = load_times
                    ntimes = len(load_times)

                    col_group = solutions[uname]

                    if 'SOLUTION_COLUMN_INLET' in col_group \
                            and 'SOLUTION_COLUMN_OUTLET' in col_group:

                        # get column locations
                        result_set.col_locs = np.array([0.0, col.length], dtype='d')

                        a = col_group['SOLUTION_COLUMN_INLET']
                        b = col_group['SOLUTION_COLUMN_OUTLET']

                        # define data structure
                        conc = np.empty((ncomp, ntimes, 2))

                        # load inlets
                        inlet = np.empty((ntimes, ncomp))
                        a.read_direct(inlet)


                        # load outlets
                        outlet = np.empty((ntimes, ncomp))
                        b.read_direct(outlet)


                        # store concentration data
                        conc[:, :, 0] = np.transpose(inlet)


                        conc[:, :, 1] = np.transpose(outlet)

                        # all the other data structures in data set are empty
                        result_set.C = xr.DataArray(conc,
                                                    coords=[result_set.components,
                                                            result_set.times,
                                                            result_set.col_locs],
                                                    dims=['component',
                                                          'time',
                                                          'location'])

                    elif 'SOLUTION_COLUMN' in col_group and 'SOLUTION_PARTICLE' in col_group:
                        shapes = col_group['SOLUTION_PARTICLE'].shape
                        ncol = shapes[1]
                        npar= shapes[2]

                        # get column locations
                        result_set.col_locs = np.linspace(0.0, col.length, ncol)

                        # get particle locations
                        result_set.par_locs = np.linspace(0.0, col.particle_radius, npar)

                        # store concentration data
                        a = col_group['SOLUTION_COLUMN']
                        transposed_conc = np.zeros(a.shape)
                        a.read_direct(transposed_conc)
                        conc = np.transpose(transposed_conc, axes=(1, 0, 2))

                        result_set.C = xr.DataArray(conc,
                                                    coords=[result_set.components,
                                                            result_set.times,
                                                            result_set.col_locs],
                                                    dims=['component',
                                                          'time',
                                                          'location'])

                        # store Cp data
                        b = col_group['SOLUTION_PARTICLE']
                        all_particle = np.zeros(b.shape)
                        b.read_direct(all_particle)
                        transposed_all_particle = np.transpose(all_particle, axes=(3, 0, 1, 2))

                        cp_data = transposed_all_particle[0:ncomp, :, :, :]
                        result_set.Cp = xr.DataArray(cp_data,
                                                     coords=[result_set.components,
                                                             result_set.times,
                                                             result_set.col_locs,
                                                             result_set.par_locs],
                                                     dims=['component',
                                                           'time',
                                                           'col_loc',
                                                           'par_loc'])

                        # store q data
                        q_data = transposed_all_particle[ncomp:, :, :, :]
                        result_set.Q = xr.DataArray(q_data,
                                                    coords=[result_set.components,
                                                            result_set.times,
                                                            result_set.col_locs,
                                                            result_set.par_locs],
                                                    dims=['component',
                                                          'time',
                                                          'column location',
                                                          'particle location'])

                    else:
                        raise RuntimeError('Parsing selection not found')

                    results[n] = result_set
                    first_name = n

                else:
                    raise RuntimeError('Unit not found in solution file')

        logger.info('parsed outputfile successfully')
        if len(results) > 1:
            return results
        return results[first_name]










