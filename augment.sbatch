#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=broadwl
#SBATCH --mem=16000
#SBATCH --chdir=/home/jono/Repositories/OrganoID


echo 'Starting batch job...'
echo 'Loading Conda module...'
module load python
echo 'Loading OrganoID environment'
source activate /home/jono/ML/OrganoIDAugmentEnv
echo 'Executing python script...'
python organoID.py augment 2000 -S 512 512 -O /home/jono/ML/TrainData /home/jono/ML/Jono_2021Masks/2021_OriginalX /home/jono/ML/Jono_2021Masks/2021_OriginalY
