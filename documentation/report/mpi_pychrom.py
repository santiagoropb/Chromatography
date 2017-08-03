from pychrom_example import create_model, compute_retention_time
from pychrom.modeling.cadet_modeler import CadetModeler
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

GRM = create_model(noise=True)
# create a modeler
modeler = CadetModeler(GRM)
modeler.discretize_column('column', ncol=30, npar=3)

# running a simulation
results = modeler.run_sim(tspan=np.linspace(0,1500,3000), retrive_c='all')
trs = compute_retention_time(GRM, results)
all_trs = comm.gather(trs, root=0)

if rank==0:
    
    avg = {k: 0.0 for k in GRM.list_components() if k!='salt'}
    for d in all_trs:
        for k in avg.keys():
            avg[k] += d[k]/float(size)

    print(avg)
