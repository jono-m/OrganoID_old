#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=bigmem2
#SBATCH --mem=128G
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --chdir=/home/jono/Repositories/OrganoID

module load python
conda activate OrganoID
python organoID.py