#!/bin/bash
#SBATCH --job-name=test_FW_mpi
#SBATCH --partition=short
#SBATCH --nodelist=persee    
#SBATCH --output=/home/FO278650/Bureau/FullWave2D_FO/fullwave2d/template/output/output_%j.log
#SBATCH --error=/home/FO278650/Bureau/FullWave2D_FO/fullwave2d/template/error/error%j.log
#SBATCH --mem=100G
#SBATCH --time=12:00:00

source ~/.bashrc

cd /home/FO278650/Bureau/FullWave2D_FO/fullwave2d/template/

# echo "Running on node: $(hostname)"
# nvidia-smi
# module load python hdf5 cuda
# conda init
module load mpi/2025.3.0
# conda activate FW2D

# python template_HW.py
mpiexec -n 64 python template_HW_PCR.py
