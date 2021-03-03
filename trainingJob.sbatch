#!/bin/bash
#SBATCH --job-name=OrganoID
#SBATCH --time=00:30:00
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/jono/Repositories/OrganoID

module load python
conda activate OrganoID_gpu
python organoID.py
