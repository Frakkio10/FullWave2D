#!/bin/bash

#SBATCH -J mpi_gysela                  # jobname
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00            # execute time (hh:mm:ss - max 08:00:00)
#SBATCH --account=FUA37_LHPED23    # account number
#SBATCH -o log/FULL_W_%j.out           # strout filename (%j is jobid)
#SBATCH -e log/FULL_W_%j.out           # stderr filename (%j is jobid)
#SBATCH --exclusive
#SBATCH --partition=skl_fua_prod
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rienaecker.sascha@t-online.de

# module load python/3.6.4
# source $HOME/Documents/FullWave/env/bin/activate
# module load profile/base
# module load intel/pe-xe-2018--binary
# module load mkl/2018--binary
# module load numpy/1.14.0--python--3.6.4
# module load intelmpi/2018--binary
# module load mpi4py/3.0.0--python--3.6.4
# module load scipy/1.2.2--python--3.6.4

module load python/3.9.4
module load intel/pe-xe-2020--binary
module load mkl/2020--binary
module load intelmpi/2020--binary # for the mpiexec command
source $FWDIR/.fullwave_env_py39/bin/activate
# mpiexec -n 264 python3 -m mpi4py mpi_turb.py 2 -18
# mpiexec -n 264 python3 -m mpi4py mpi_turb.py 4 -15
# mpiexec -n 264 python3 -m mpi4py mpi_turb.py 4 -10
mpiexec -n 48 python3 -m mpi4py run_Doppler_measurement.py
# mpiexec -n 264 python3 -m mpi4py mpi_turb.py 8 -7
