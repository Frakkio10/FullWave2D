from mpi4py import MPI
import numpy as np
import h5py
from math import ceil
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

filename = "dummy_data.h5"

def create_dummy_h5(filename, Nt=10, Ny=4, Nx=5):
    """Create a dummy HDF5 file that mimics your structure."""
    if rank == root:
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Creating dummy file {filename} with shape ({Nt}, {Ny}, {Nx})")

        with h5py.File(filename, "w") as f:
            f.create_dataset("fields/n", data=np.random.rand(Nt, Ny, Nx))
            f.create_dataset("grid/x", data=np.linspace(0, 1, Nx))
            f.create_dataset("grid/y", data=np.linspace(0, 1, Ny))
    comm.Barrier()  # wait for file creation to finish


def test_h5_time_distribution(t_start=0, t_end=None, root=0):
    """Parallel HDF5 read test (mimicking scatterv_maxwell_HW)."""
    with h5py.File(filename, "r") as f:
        Nt_total = f["fields/n"].shape[0]
        if t_end is None:
            t_end = Nt_total

    if rank == root:
        Nt_sel = t_end - t_start
        n_pproc = np.zeros(size, dtype=int)
        n_left = Nt_sel

        for i in range(size):
            n_pproc[i] = ceil(n_left / (size - i))
            n_left -= n_pproc[i]
    else:
        Nt_sel = None
        n_pproc = None

    Nt_sel = comm.bcast(Nt_sel, root=root)
    n_pproc = comm.bcast(n_pproc, root=root)

    start_idx = t_start + np.sum(n_pproc[:rank])
    end_idx   = start_idx + n_pproc[rank]
    local_times = range(start_idx, end_idx)

    print(f"Rank {rank} → timesteps {list(local_times)}")

    # Each rank opens file and reads only its part
    with h5py.File(filename, "r") as f:
        n = f["fields/n"]
        local_results = []
        for t in local_times:
            ne_map = n[t, :, :]
            local_results.append(np.mean(ne_map))  # dummy computation

    outp_local = np.array(local_results, dtype=float)
    sendcounts = n_pproc
    displs = np.cumsum(sendcounts) - sendcounts

    if rank == root:
        outp_gathered = np.zeros(Nt_sel, dtype=float)
    else:
        outp_gathered = None

    comm.Gatherv(outp_local, [outp_gathered, sendcounts, displs, MPI.DOUBLE], root=root)

    if rank == root:
        print("\n✅ Gathered mean values from all ranks:")
        print(outp_gathered)

if __name__ == "__main__":
    # Step 1: Create dummy file (only once)
    create_dummy_h5(filename, Nt=800, Ny=4, Nx=5)

    # Step 2: Run the parallel test
    test_h5_time_distribution(t_start=20, t_end=30)
