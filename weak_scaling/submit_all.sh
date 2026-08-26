#!/bin/bash
set -e
sbatch /pfs/lustrep4/users/bouziane/iPIC3D-GPU/weak_scaling/N1/submit_N1.slurm
sbatch /pfs/lustrep4/users/bouziane/iPIC3D-GPU/weak_scaling/N2/submit_N2.slurm
