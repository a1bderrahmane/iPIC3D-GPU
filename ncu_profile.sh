#!/bin/bash
export OMP_NUM_THREADS=6

NCU_ARGS="--nvtx --nvtx-include gpuMaxwellImage/ --launch-skip 10 --launch-count 50 --set full --clock-control none --target-processes all -o maxwell_profile_rank%q{OMPI_COMM_WORLD_RANK}"

mpirun -n 2 --bind-to none --mca pml ob1 --mca btl self,vader,smcuda ncu $NCU_ARGS build/iPIC3D testGEM3D.inp > testGEM3D.log 2>&1

