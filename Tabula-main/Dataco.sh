#!/bin/bash
#SBATCH -p ampere
#SBATCH --account BRINTRUP-SL3-GPU
#SBATCH -D /home/yl892/rds/hpc-work/Tabular-Data-Generation/Tabula-main
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G    # RAM memory. Default: 1G
#SBATCH -t 12:00:00 # time for the job HH:MM:SS. Default: 1 min
python Dataco.py