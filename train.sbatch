#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=01:00:00
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
python organoID.py train -S 512 512 -B 16 -TS 0.1 -E 20 -O /home/jono/ML/Models /home/jono/ML/TrainData/OrganoID_augment_2021_06_01_11_41_07/images /home/jono/ML/TrainData/OrganoID_augment_2021_06_01_11_41_07/segmentations
