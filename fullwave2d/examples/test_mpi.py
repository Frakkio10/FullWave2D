from mpi4py import MPI

root = 0
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_root = rank == root

if is_root:
    print('rank: ', rank)
    print('size: ', size)
