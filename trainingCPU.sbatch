#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=broadwl
#SBATCH --chdir=/home/jono/Repositories/OrganoID

module load python
conda activate OrganoID_cpu
python organoID.py train -A -P -F /home/jono/ML/2019_originalX /home/jono/ML/2019_originalY
