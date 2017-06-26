from pychrom.model.chromatograpy_model import GRModel
from pychrom.model.unit_operation import Column
from pychrom.model.registrar import Registrar
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import warnings
import tempfile
import h5py
import uuid
import os


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

        # TODO: improve this subprocess call to check success and output
        cadet_location = 'cadet-cli'
        cmd = [cadet_location, filename, outfilename]
        proc = subprocess.Popen(cmd,
                                bufsize=0,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        proc.wait()

        plotSimulation(outfilename)

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

        reg_disc = Registrar.discretization_defaults
        disct_kwargs = dict()
        params = ['gs_type',
                  'max_krylov',
                  'max_restarts',
                  'schur_safety']

        for n in params:
            disct_kwargs[n] = kwargs.pop(n, reg_disc[n])

        self._model.write_to_cadet_input_file(filename,
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
            u.write_discretization_to_cadet_input_file(filename,
                                                       ncol,
                                                       npar,
                                                       **disct_dict)

########################## temporary functions ################################
def plotSimulation(filename):
    with h5py.File(filename, 'r') as h5:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=[16, 8])
        plotInlet(ax1, h5)
        plotOutlet(ax2, h5)
        f.tight_layout()
        plt.show()


def plotInlet(axis, h5):
    solution_times = np.array(h5['/output/solution/SOLUTION_TIMES'].value)

    inlet_salt = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_INLET_COMP_000'].value)
    inlet_p1 = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_INLET_COMP_001'].value)
    inlet_p2 = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_INLET_COMP_002'].value)
    inlet_p3 = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_INLET_COMP_003'].value)

    axis.set_title("Inlet")
    axis.plot(solution_times, inlet_salt, 'b-', label="Salt")
    axis.set_xlabel('time (s)')

    # Make the y-axis label, ticks and tick labels match the line color.
    axis.set_ylabel('mMol Salt', color='b')
    axis.tick_params('y', colors='b')

    axis2 = axis.twinx()
    axis2.plot(solution_times, inlet_p1, 'r-', label="P1")
    axis2.plot(solution_times, inlet_p2, 'g-', label="P2")
    axis2.plot(solution_times, inlet_p3, 'k-', label="P3")
    axis2.set_ylabel('mMol Protein', color='r')
    axis2.tick_params('y', colors='r')

    lines, labels = axis.get_legend_handles_labels()
    lines2, labels2 = axis2.get_legend_handles_labels()
    axis2.legend(lines + lines2, labels + labels2, loc=0)


def plotOutlet(axis, h5):
    solution_times = np.array(h5['/output/solution/SOLUTION_TIMES'].value)

    outlet_salt = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_OUTLET_COMP_000'].value)
    outlet_p1 = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_OUTLET_COMP_001'].value)
    outlet_p2 = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_OUTLET_COMP_002'].value)
    outlet_p3 = np.array(h5['/output/solution/unit_001/SOLUTION_COLUMN_OUTLET_COMP_003'].value)

    axis.set_title("Output")
    axis.plot(solution_times, outlet_salt, 'b-', label="Salt")
    axis.set_xlabel('time (s)')

    # Make the y-axis label, ticks and tick labels match the line color.
    axis.set_ylabel('mMol Salt', color='b')
    axis.tick_params('y', colors='b')

    axis2 = axis.twinx()
    axis2.plot(solution_times, outlet_p1, 'r-', label="P1")
    axis2.plot(solution_times, outlet_p2, 'g-', label="P2")
    axis2.plot(solution_times, outlet_p3, 'k-', label="P3")
    axis2.set_ylabel('mMol Protein', color='r')
    axis2.tick_params('y', colors='r')

    lines, labels = axis.get_legend_handles_labels()
    lines2, labels2 = axis2.get_legend_handles_labels()
    axis2.legend(lines + lines2, labels + labels2, loc=0)







