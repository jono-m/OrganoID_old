#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=broadwl
#SBATCH --mem=64G
#SBATCH --chdir=/home/jono/Repositories/OrganoID

module load python
conda activate OrganoID
python organoID.py
