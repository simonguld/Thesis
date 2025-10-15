#!/bin/bash

#SBATCH -A geomechanics
#SBATCH --job-name="Main2"
#SBATCH --output="Main2.%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --export=ALL

export OMP_NUM_THREADS=16
srun Main2 $1
