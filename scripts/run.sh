#!/bin/bash
#SBATCH -o out_test_20210514
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p normal
#SBATCH --gres=dcu:4
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=cores
#SBATCH --no-requeue
#SBATCH --exclusive

export OMP_NUM_THREADS=32
srun  --cpu_bind=cores  --mpi=pmix  ../builds/singleNodeImproved_path -f delaunay_n16 -k 8 -direct false -weight false

