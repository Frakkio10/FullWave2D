#%%

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor, ceil

from fullwave2d.core.wrapper import fw2d_wrapper, OutputData

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def launch_parallel_fw2d(input_data, root=0, make_outp_dir=True):
    """

    """

    N_inputs = len(input_data)
    is_root = rank==root

    # in case N_maps > size (number of cores),
    # we will send several maps to each core:
    n_pproc = np.zeros(size).astype(int) # maps per core
    n_left = N_inputs # maps left
    for i, npp in enumerate(n_pproc):
        n_pproc[i] = ceil(n_left / (size-i))
        n_left -= n_pproc[i]

    # displacements along the list
    displ = [npp for npp in n_pproc]
    displ = np.cumsum(displ) - displ
    displ = np.append(displ,  displ[-1]+n_pproc[-1])

    if is_root:
        print('Number of inputs: ', N_inputs)
        print('number of maps per core :', n_pproc)
        print('Displacements: ', displ)

    inp_local = input_data[displ[rank]:displ[rank+1]]
    print('rank {}: input name: {}'.format(rank, [x.name for x in inp_local]))

    # the local result data to be gathered is the amplitude
    # and phase computed by the maxwell solver for each slice:
    # outp_local = np.zeros(n_pproc[rank] * 2)

    # perform on each CPU the actual
    # calculation for each input data
    for i in range(n_pproc[rank]):
        # outp_local[2*i], outp_local[2*i+1] = fw2d_wrapper(inp_local, MPI_root=True)
        fw2d_wrapper(inp_local[i], make_outp_dir=True)

        # to keep track of the progress:
        if is_root:
            print('Root core finished {} out of {} slices'.format(i+1, n_pproc[root]))



# %%
