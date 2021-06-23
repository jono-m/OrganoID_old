#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=10:00:00
#SBATCH --partition=gpu2
#SBATCH --chdir=/home/jono/Repositories/OrganoID
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16000


echo 'Starting batch job...'
echo 'Loading Conda module...'
module load python
conda activate /home/jono/ML/Environments/OrganoID
echo 'Executing python script...'
python organoID.py train -S 512 512 -B 8 -TS 0.2 -E 200 -DR 0.1 -O /home/jono/ML/Models /home/jono/ML/TrainData/OrganoID_augment_2021_06_15_15_24_28/images /home/jono/ML/TrainData/OrganoID_augment_2021_06_15_15_24_28/segmentations
