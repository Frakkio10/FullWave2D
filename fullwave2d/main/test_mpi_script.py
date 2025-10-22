
from mpi4py import MPI
import numpy as np
import h5py as h5
import os
from tempfile import TemporaryDirectory
from pathlib import Path 
from mpi_maxwell import scatterv_maxwell_from_h5  # adjust import if needed


# Mock fw2d_wrapper to simulate computation
def fw2d_wrapper(input_data, is_root):
    """Fake simulation that returns mean and std of density map"""
    amp = np.mean(input_data.ne)
    phase = np.std(input_data.ne)
    return amp, phase

class DummyInput:
    def __init__(self):
        self.ne = None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    root = 0

    with TemporaryDirectory() as tmpdir:
        filename = Path('/home/forlacchio/FullWave2D_FO/fullwave2d/main/test.h5')

        # Only root creates the dummy HDF5 file
        if rank == root:
            Nt, Ny, Nx = 10, 32, 32
            delta_ne = np.random.randn(Nt, Ny, Nx)
            with h5.File(filename, "w") as f:
                f.create_dataset("fields/n", data=delta_ne)
        comm.Barrier()  # wait for file creation

        # Dummy input
        ne_lin = np.ones((32, 32))
        inp_dict = {"nproc": 4}
        input_data = DummyInput()

        # Run the MPI function
        results = scatterv_maxwell_from_h5(input_data, ne_lin, filename,
                                           t_start=0, t_end=10, fluct_lvl=0.5)

        if rank == root:
            print("\n--- Gathered Results ---")
            print(results)
            print("Shape:", results.shape)

if __name__ == "__main__":
    main()

