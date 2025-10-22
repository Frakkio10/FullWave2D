from mpi4py import MPI
import numpy as np
from math import ceil

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

def test_time_distribution(t_start=0, t_end=10, root=0):
    """
    Minimal test to verify MPI time-splitting and Gatherv behavior.
    """
    if rank == root:
        print(f"\nProcessor {rank}/{size} distributing timesteps")

        Nt_sel = t_end - t_start
        n_pproc = np.zeros(size, dtype=int)
        n_left = Nt_sel

        for i in range(size):
            n_pproc[i] = ceil(n_left / (size - i))
            n_left -= n_pproc[i]
    else:
        Nt_sel = None
        n_pproc = None

    # Broadcast
    Nt_sel = comm.bcast(Nt_sel, root=root)
    n_pproc = comm.bcast(n_pproc, root=root)

    # Local time indices
    start_idx = t_start + np.sum(n_pproc[:rank])
    end_idx   = start_idx + n_pproc[rank]
    local_times = range(start_idx, end_idx)

    print(f"Rank {rank} handles timesteps: {list(local_times)}")

    # Dummy "local results" (e.g., just squares of timesteps)
    outp_local = np.array([t**2 for t in local_times], dtype=float)

    # Prepare to gather
    sendcounts = n_pproc
    displs = np.cumsum(sendcounts) - sendcounts

    if rank == root:
        outp_gathered = np.zeros(Nt_sel, dtype=float)
    else:
        outp_gathered = None

    comm.Gatherv(outp_local, [outp_gathered, sendcounts, displs, MPI.DOUBLE], root=root)

    if rank == root:
        print(f"\nGathered results on root: {outp_gathered}")

if __name__ == "__main__":
    test_time_distribution(t_start=0, t_end=10)
