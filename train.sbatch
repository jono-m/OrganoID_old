#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=gpu2
#SBATCH --chdir=/home/jono/Repositories/OrganoID
#SBATCH --gres=gpu:1
#SBATCH --nodes=1


echo 'Starting batch job...'
echo 'Loading Conda module...'
module load tensorflow/2.1
echo 'Executing python script...'
python organoID.py train -S 512 512 -B 8 -TS 0.2 -O /home/jono/ML /home/jono/ML/2018Dec_originalX /home/jono/ML/2018Dec_originalY
