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
python organoID.py train -S 512 512 -B 4 -TS 0.1 -E 100 -DR 0 -O /home/jono/ML/Models /home/jono/ML/RawData/Jono_2021Masks/2021_OriginalX /home/jono/ML/RawData/Jono_2021Masks/202105_OriginalY
