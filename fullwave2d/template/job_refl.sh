#!/bin/bash
#SBATCH --job-name=test_FW_mpi
#SBATCH --partition=short
#SBATCH --nodelist=andromede1    
#SBATCH --output=/Home/FO278650/Zone_Travail/HWAK/output/output_%j.log
#SBATCH --error=/Home/FO278650/Zone_Travail/HWAK/error/error%j.log
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
mpiexec -n 1 python template_refl.py
