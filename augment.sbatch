#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=broadwl
#SBATCH --chdir=/home/jono/Repositories/OrganoID


echo 'Starting batch job...'
echo 'Loading Conda module...'
module load python
echo 'Loading OrganoID environment'
source activate /home/jono/ML/OrganoIDAugmentEnv
echo 'Executing python script...'
python organoID.py augment 2000 -S 512 512 -O /home/jono/ML /home/jono/ML/2018Dec_originalX /home/jono/ML/2018Dec_originalY
