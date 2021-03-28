#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=broadwl
#SBATCH --chdir=/home/jono/Repositories/OrganoID
#SBATCH --mail-type=ALL


echo 'Starting batch job...'
echo 'Loading Conda module...'
module load python
echo 'Loading OrganoID environment (CPU)'
conda activate OrganoID_cpu
echo 'Executing python script...'
python organoID.py train -A 1000 -S 512 512 -B 16 -TS 0.2 -F -O /home/jono/ML/OrganoID /home/jono/ML/2018Dec_originalX /home/jono/ML/2018Dec_originalY
