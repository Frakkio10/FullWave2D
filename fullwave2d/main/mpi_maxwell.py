
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import floor, ceil
import h5py as h5
#from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData
from fullwave2d.core.wrapper import fw2d_wrapper, InputData, OutputData
import copy 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
    
def scatterv_maxwell_v2(ne_bkg, delta_ne, dny, n_slices, input_data, root=0, rms = 0.02):

    """
    Splits a 2d array of shape (Ny x Nx) into equally
    shaped slices (ny x nx) to be processed by each
    each CPU individually.

    Args:
        ne_bkg     : 2d experimental background density map 
        delta_ne   : 2d turbulence map to be split
        rms        : intensity of fluctuation as a function of the local background density 
        dny        : vertical displacement between slices
        n_slices   : number of slices. If None, the maximum allowed integer is chosen
        input_data : Input to be passed to the maxwell wrapper (wrapper.InputData)
        root       : master process from which to scatter

    Returns:
        outp_gathered : the merged antenna result as a (n_slices x 2) matrix,
                        where the first/second column are the amplitude/phase
    
    physically, dny is related to the rotation velocity v of the plasma, i.e. how much it has moved over a time step dt: dny = v * dt/dy
    
    """

    
    if rank == root:
        print("This is processor {}, rank {} out of {}\n".format(comm, rank, size))

        Ny, Nx = delta_ne.shape
        nx = input_data.nx
        ny = input_data.ny
        print('Read input file with entries (Ny, Nx) = {}'.format(delta_ne.shape))

        if nx > Nx or ny > Ny:
            print('slice dimensions (ny,nx) cannot exceed the size (Ny, Nx) of the density profile')
            return

        # Calculate the maximum number of slices
        max_slices = floor((Ny - ny) / dny) + 1
        n_slices   = max_slices if n_slices is None else min(n_slices, max_slices)
        print('The 2d matrix is cut into {} slices of shape (ny, nx) = ({}, {})'.format(n_slices, ny, nx))

        # Determine the number of slices per process
        n_pproc = np.zeros(size, dtype = int)
        n_left  = n_slices 
        for i, npp in enumerate(n_pproc):
            n_pproc[i] = ceil(n_left / (size-i))
            n_left    -= n_pproc[i]
        print('number of slices per core :', n_pproc)


        # Calculate displacements and send counts for Scatterv
        displacements = [nx * dny * npp for npp in n_pproc]
        displacements = np.cumsum(displacements) - displacements
        print('displacements along y:', displacements/nx)
        print('displacements along flattened array:', displacements)

        sendcounts = [nx * (ny + dny * (npp - 1)) for npp in n_pproc]
        sendcounts = np.array(sendcounts).astype(int)
        print('sendcounts :', sendcounts)


    else:
        nx            = None
        ny            = None
        displacements = None
        sendcounts    = None
        n_pproc       = None

    # Broadcast slice parameters to all processes 
    nx            = comm.bcast(nx, root=root)
    ny            = comm.bcast(ny, root=root)
    displacements = comm.bcast(displacements, root=root)
    sendcounts    = comm.bcast(sendcounts, root=root)
    n_pproc       = comm.bcast(n_pproc, root=root)

    # Broadcast the background map to all processes
    ne_bkg        = comm.bcast(ne_bkg, root = root)
    
    # Scatter the large map slices to all processes
    delta_ne_loc  = np.zeros(sendcounts[rank])
    
    comm.Scatterv([delta_ne, sendcounts, displacements,
                   MPI.DOUBLE], delta_ne_loc, root=root)
       
    # Reshape the received slice into a 2D array
    delta_ne_loc  = delta_ne_loc.reshape(-1, nx)
    
    # Initialize the output array for local results 
    outp_local    = np.zeros(n_pproc[rank] * 2)

    # Performing the computation for each slice
    for i in range(n_pproc[rank]):
        i_left        = dny * i
        i_right       = i_left + ny
        slice         = delta_ne_loc[i_left:i_right]

        ne_tot        = ne_bkg + rms * ne_bkg * slice
        input_data.ne = ne_tot
        
        # Launching the simulation 
        outp_local[2 * i], outp_local[2 * i + 1] = fw2d_wrapper(input_data, rank==root)

        # to keep track of the progress:
        if rank == root:
            print('Root core finished {} out of {} slices'.format(i+1, n_pproc[root]))

    # Gather the resuklts back to the root process
    comm.Barrier()
    if rank == root:
        outp_gathered = np.zeros(n_slices * 2)
    else:
        outp_gathered = None

    outp_sendc = 2 * n_pproc
    outp_displ = np.cumsum(outp_sendc) - outp_sendc

    comm.Barrier()
    comm.Gatherv(outp_local, [outp_gathered, outp_sendc, outp_displ, MPI.DOUBLE], root=root)
    comm.Barrier()

    if rank == root:
        outp_gathered = outp_gathered.reshape((-1,2))
        return outp_gathered
    

def scatter_maxwell_circular(ne_bkg, delta_ne, dny, n_slices, input_data, get_prof, get_prof_args=(), root=0, rms = 0.02):

    if rank == root:
        print("This is processor {}, rank {} out of {}\n".format(comm, rank, size))

        Ny, Nx = delta_ne.shape
        nx = input_data.nx
        ny = input_data.ny
        print('Read input file with entries (Ny, Nx) = {}'.format(delta_ne.shape))

        if nx > Nx or ny > Ny:
            print('slice dimensions (ny,nx) cannot exceed the size (Ny, Nx) of the density profile')
            return

        # Calculate the maximum number of slices
        max_slices = floor((Ny - ny) / dny) + 1
        n_slices   = max_slices if n_slices is None else min(n_slices, max_slices)
        print('The 2d matrix is cut into {} slices of shape (ny, nx) = ({}, {})'.format(n_slices, ny, nx))

        # Determine the number of slices per process
        n_pproc = np.zeros(size, dtype = int)
        n_left  = n_slices 
        for i, npp in enumerate(n_pproc):
            n_pproc[i] = ceil(n_left / (size-i))
            n_left    -= n_pproc[i]
        print('number of slices per core :', n_pproc)


        # Calculate displacements and send counts for Scatterv
        displacements = [nx * dny * npp for npp in n_pproc]
        displacements = np.cumsum(displacements) - displacements
        print('displacements along y:', displacements/nx)
        print('displacements along flattened array:', displacements)

        sendcounts = [nx * (ny + dny * (npp - 1)) for npp in n_pproc]
        sendcounts = np.array(sendcounts).astype(int)
        print('sendcounts :', sendcounts)


    else:
        nx            = None
        ny            = None
        displacements = None
        sendcounts    = None
        n_pproc       = None

    # Broadcast slice parameters to all processes 
    nx            = comm.bcast(nx, root=root)
    ny            = comm.bcast(ny, root=root)
    displacements = comm.bcast(displacements, root=root)
    sendcounts    = comm.bcast(sendcounts, root=root)
    n_pproc       = comm.bcast(n_pproc, root=root)

    # Broadcast the background map to all processes
    ne_bkg        = comm.bcast(ne_bkg, root = root)
    
    # Scatter the large map slices to all processes
    delta_ne_loc  = np.zeros(sendcounts[rank])
    
    comm.Scatterv([delta_ne, sendcounts, displacements,
                   MPI.DOUBLE], delta_ne_loc, root=root)
       
    # Reshape the received slice into a 2D array
    delta_ne_loc  = delta_ne_loc.reshape(-1, nx)
    
    # Initialize the output array for local results 
    outp_local    = np.zeros(n_pproc[rank] * 2)

    # Performing the computation for each slice
    for i in range(n_pproc[rank]):
        
        dn_cart = get_prof(delta_ne, *get_prof_args)
        delta_ne = np.roll(delta_ne, -dny, axis = 0)
        ne_tot        = ne_bkg * (1 + rms * dn_cart)
        input_data.ne = ne_tot
        
        #fig, ax  = plt.subplots(1, 2, figsize = (12, 5))
        
        #ax[0].imshow(ne_tot, origin = 'lower', cmap = 'Blues')
        #ax[1].imshow(dn_cart, origin = 'lower', cmap = 'seismic')
        
        # Launching the simulation 
        outp_local[2 * i], outp_local[2 * i + 1] = fw2d_wrapper(input_data, rank==root)

        # to keep track of the progress:
        if rank == root:
            print('Root core finished {} out of {} slices'.format(i+1, n_pproc[root]))

    # Gather the resuklts back to the root process
    comm.Barrier()
    if rank == root:
        outp_gathered = np.zeros(n_slices * 2)
    else:
        outp_gathered = None

    outp_sendc = 2 * n_pproc
    outp_displ = np.cumsum(outp_sendc) - outp_sendc

    comm.Barrier()
    comm.Gatherv(outp_local, [outp_gathered, outp_sendc, outp_displ, MPI.DOUBLE], root=root)
    comm.Barrier()

    if rank == root:
        outp_gathered = outp_gathered.reshape((-1,2))
        return outp_gathered


def scatterv_maxwell_HW(HW_inp, input_data, t_start=0, t_end=None, root=0, **kwargs):
    """
    Distributes time indices across MPI ranks.
    Each rank generates its own density maps using input_HW
    and runs fw2d_wrapper on them.

    Args:
        HW_inp     : class for the creating the HW turbulence map from h5py file
        input_data : Input object for Maxwell solver (wrapper.InputData)
        t_start    : First time index (inclusive)
        t_end      : Last time index (exclusive). If None, run until available end
        root       : Master process
        
    kwargs:
        fluct_lvl  : fluctuation level for the turbulence map, if not given 1 is taken

    Returns:
        outp_gathered : (Nt_selected, 2) array with amplitude/phase results
    """
    
    fluct_lvl = kwargs.pop('fluct_lvl', 1)
    
    print('fluctuation level', fluct_lvl)
    
    if rank == root:
        print(f"Processor {rank}/{size} starting time-distributed MPI run.")

        # Total number of time steps requested
        Nt     = HW_inp.time.size  
        Ny, Nx = HW_inp.n.shape

        if t_end is None:
            t_end = Nt
        Nt_sel = t_end - t_start
        if Nt_sel <= 0:
            raise ValueError("Invalid time range: check t_start and t_end")

        print(f"Processing time indices [{t_start}:{t_end}] → {Nt_sel} timesteps")

        # Distribute timesteps among ranks
        n_pproc = np.zeros(size, dtype=int)
        n_left = Nt_sel
        for i in range(size):
            n_pproc[i] = ceil(n_left / (size - i))
            n_left -= n_pproc[i]
    else:
        Nt_sel = None
        n_pproc = None

    # Broadcast counts
    Nt_sel = comm.bcast(Nt_sel, root=root)
    n_pproc = comm.bcast(n_pproc, root=root)

    # Each rank determines its local time indices
    start_idx = t_start + np.sum(n_pproc[:rank])
    end_idx   = start_idx + n_pproc[rank]
    local_times = range(start_idx, end_idx)

    if rank == root:
        print("Time indices per rank:", n_pproc)

    # Local results
    outp_local = np.zeros(len(local_times) * 2)

    for i, t in enumerate(local_times):
        x, y, ne_map = HW_inp.prepare_ne_map(HW_inp.n, t, fluct_lvl, **kwargs) #function to generate the density map

        # Example: add fluctuations if desired
        input_data.ne = ne_map
        print('inside the loop', i , t)
        # Run the simulation
        outp_local[2*i], outp_local[2*i+1] = fw2d_wrapper(input_data, rank == root)

        if rank == root:
            print(f"Root finished timestep {t}")

    # Gather results
    if rank == root:
        outp_gathered = np.zeros(Nt_sel * 2)
    else:
        outp_gathered = None

    outp_sendc = 2 * n_pproc
    outp_displ = np.cumsum(outp_sendc) - outp_sendc

    comm.Gatherv(outp_local, [outp_gathered, outp_sendc, outp_displ, MPI.DOUBLE], root=root)

    if rank == root:
        
        return outp_gathered.reshape(Nt_sel, 2)
    

def scatterv_maxwell_from_h5(input_data, ne_lin, h5_filename,
                             t_start=0, t_end=None, root=0, *, fluct_lvl=0.05, simulations_per_CPU = 1):
    """
    MPI-distributed execution of fw2d_wrapper over time steps in an HDF5 file.
    Each rank handles several timesteps sequentially.

    Args:
        inp_dict    : dictionary of parameters (e.g., may contain 'nproc')
        input_data  : object used by fw2d_wrapper, must have attribute .ne
        ne_lin      : 2D background density map
        h5_filename : HDF5 file containing dataset 'fields/n' of shape (Nt, Ny, Nx)
        t_start     : first time index (inclusive)
        t_end       : last time index (exclusive). If None → full available range
        root        : master rank
        fluct_lvl   : multiplicative factor for delta_ne

    Returns:
        On root: (Nt_sel, 2) array [amplitude, phase]
        On others: None
    """

    # Root inspects the file and decides how to split timesteps
    if rank == root:
        
        #reading h5 file 
        with h5.File(h5_filename, "r") as f:
            Nt, Ny, Nx = f["fields/n"][:].shape
            
        Ny_lin, Nx_lin = ne_lin.shape
        # if (Ny_lin, Nx_lin) != (Ny, Nx):
            # raise ValueError("Shape mismatch between ne_lin and HDF5 dataset")

        # Determine range
        # if t_end is None:
        #     # Default: each processor handles one timestep if possible
        #     t_end = min(t_start + size, Nt)
        if t_end is None:
            # Allow multiple simulations per CPU
            max_timesteps = size * simulations_per_CPU
            t_end = min(t_start + max_timesteps, Nt)
        if not (0 <= t_start < t_end <= Nt):
            raise ValueError(f"Invalid time range {t_start}:{t_end} for Nt={Nt}")
        # Nt_sel = t_end - t_start

        # # Compute balanced load: number of time steps per rank
        # n_pproc = np.zeros(size, dtype=int)
        # n_left = Nt_sel
        # for i in range(size):
        #     n_pproc[i] = ceil(n_left / (size - i))
        #     n_left -= n_pproc[i]
        Nt_sel = t_end - t_start

        # Explicitly assign simulations_per_CPU tasks per rank
        n_pproc = np.zeros(size, dtype=int)
        for i in range(size):
            start_i = t_start + i * simulations_per_CPU
            end_i   = start_i + simulations_per_CPU
            if start_i >= t_end:
                break
            n_pproc[i] = max(0, min(simulations_per_CPU, t_end - start_i))
    else:
        Nt_sel, n_pproc, t_end = None, None, None

    # Broadcast basic info
    Nt_sel = comm.bcast(Nt_sel, root=root)
    n_pproc = comm.bcast(n_pproc, root=root)
    t_end = comm.bcast(t_end, root=root)

    # Each rank determines its time indices
    start_idx = t_start + np.sum(n_pproc[:rank])
    end_idx = start_idx + n_pproc[rank]
    local_times = range(start_idx, end_idx)

    if rank == root:
        print(f"Distributed {Nt_sel} time steps among {size} ranks → {n_pproc} per rank")

    # Prepare local results
    outp_local = np.zeros(len(local_times) * 2, dtype=float)

    # # Each rank processes its local timesteps

    for i, t in enumerate(local_times):
            
        local_inp = copy.deepcopy(input_data)
        if rank == root:
            local_inp.save_diag = True
        else:
            local_inp.save_diag = False
        
        with h5.File(h5_filename, "r") as f:
            n_xky = f["fields/n"][t,:].astype(np.complex128)
        f.close()
        
        delta_ne = np.fft.irfft(n_xky, axis=1)
        ne_tot = ne_lin * (1 + fluct_lvl * delta_ne)
        local_inp.ne = ne_tot.T.astype(np.double)
        # amp, phase = fw2d_wrapper(local_inp, rank == root)
        amp, phase = fw2d_wrapper(local_inp)

        # amp, phase = fw2d_wrapper(input_data, rank == root)
        outp_local[2*i] = amp
        outp_local[2*i+1] = phase

        print(f"[Rank {rank}] Finished timestep {t}")
    # Ensure chunks is always a list of arrays
    # if simulations_per_CPU == 1:
    #     chunks = [np.arange(Nt_sel)]  # single chunk with all timesteps
    # else:
    #     local_times = np.arange(t_start, t_end)  # timesteps for this rank
    #     chunks = np.array_split(local_times, simulations_per_CPU)

    # for j, chunk in enumerate(chunks):
    #     for i, t in enumerate(chunk):
    #         # Compute the global index
    #         global_idx = 2 * (j * len(chunk) + i)
    #         print(f"Chunk {j}, Local timestep {t}, Global timestep {global_idx}")

    #         delta_ne_t = delta_ne[t, :, :]
    #         ne_tot = ne_lin * (1 + fluct_lvl * delta_ne_t)
    #         input_data.ne = ne_tot.astype(np.double)
    #         outp_local[2*i], outp_local[2*i+1] = fw2d_wrapper(input_data, rank == root)

    # Gather to root
    outp_sendc = 2 * n_pproc
    outp_displ = np.cumsum(outp_sendc) - outp_sendc

    if rank == root:
        outp_gathered = np.zeros(Nt_sel * 2, dtype=float)
    else:
        outp_gathered = None

    comm.Gatherv(outp_local, [outp_gathered, outp_sendc, outp_displ, MPI.DOUBLE], root=root)

    if rank == root:
        return outp_gathered.reshape(Nt_sel, 2)
    return None


def _scatterv_maxwell_from_h5(input_data, ne_lin, h5_filename,
                              t_start=0, t_end=None, root=0, *,
                              fluct_lvl=0.05, simulations_per_CPU=1):
    """
    MPI-distributed execution of fw2d_wrapper over time steps in an HDF5 file.
    Each rank handles several timesteps sequentially. Multiple simulations
    per CPU are treated as consecutive timesteps.

    Args:
        input_data  : object used by fw2d_wrapper, must have attribute .ne
        ne_lin      : 2D background density map
        h5_filename : HDF5 file containing dataset 'fields/n' of shape (Nt, Ny, Nx)
        t_start     : first time index (inclusive)
        t_end       : last time index (exclusive). If None → full available range
        root        : master rank
        fluct_lvl   : multiplicative factor for delta_ne
        simulations_per_CPU : number of sub-timesteps per CPU

    Returns:
        On root: (Nt_sel, 2) array [amplitude, phase]
        On others: None
    """
    from mpi4py import MPI
    import numpy as np
    import h5py as h5
    from math import ceil

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Root inspects the file and decides how to split timesteps
    if rank == root:
        with h5.File(h5_filename, "r") as f:
            Nt, Ny, Nx = f["fields/n"].shape

        Ny_lin, Nx_lin = ne_lin.shape
        if (Ny_lin, Nx_lin) != (Ny, Nx):
            raise ValueError("Shape mismatch between ne_lin and HDF5 dataset")

        if t_end is None:
            t_end = Nt
        if not (0 <= t_start < t_end <= Nt):
            raise ValueError(f"Invalid time range {t_start}:{t_end} for Nt={Nt}")

        Nt_sel = t_end - t_start

        # Determine timesteps per rank
        n_pproc = np.zeros(size, dtype=int)
        n_left = Nt_sel
        for i in range(size):
            n_pproc[i] = ceil(n_left / (size - i))
            n_left -= n_pproc[i]
    else:
        Nt_sel, n_pproc, t_end = None, None, None

    # Broadcast metadata to all ranks
    Nt_sel = comm.bcast(Nt_sel, root=root)
    n_pproc = comm.bcast(n_pproc, root=root)
    t_end = comm.bcast(t_end, root=root)

    # Determine local timesteps
    start_idx = t_start + np.sum(n_pproc[:rank])
    end_idx = start_idx + n_pproc[rank]
    local_times = np.arange(start_idx, end_idx)

    # Prepare local output
    outp_local = np.zeros(len(local_times) * 2, dtype=float)

    # Read HDF5 dataset once
    with h5.File(h5_filename, "r") as f:
        delta_ne = f["fields/n"]

    # Compute local times for this rank
    local_times = np.arange(start_idx, end_idx)

    # Split local_times into chunks
    chunks = [c for c in np.array_split(local_times, simulations_per_CPU) if len(c) > 0]

    for j, chunk in enumerate(chunks):
        for i, t in enumerate(chunk):
            global_idx = t  # you can use t directly if you want global reference
            print(f"Chunk {j}, Local timestep {t}, Global timestep {global_idx}")

            delta_ne_t = delta_ne[t, :, :]
            ne_tot = ne_lin * (1 + fluct_lvl * delta_ne_t)
            input_data.ne = ne_tot.astype(np.double)
            outp_local[2*i], outp_local[2*i+1] = fw2d_wrapper(input_data, rank == root)


    # Prepare for Gatherv
    outp_sendc = 2 * n_pproc
    outp_displ = np.cumsum(outp_sendc) - outp_sendc

    if rank == root:
        outp_gathered = np.zeros(Nt_sel * 2, dtype=float)
    else:
        outp_gathered = None

    comm.Gatherv(outp_local, [outp_gathered, outp_sendc, outp_displ, MPI.DOUBLE], root=root)

    if rank == root:
        return outp_gathered.reshape(Nt_sel, 2)
    return None
