#!/bin/bash

# Define NSYS variable for cleaner code
NSYS=/local/spack/linux-centos8-zen2/gcc-14.2.0/nvhpc-24.11-mcsqyuj3eddzlmyxx3kgswr2fawib3sf/Linux_x86_64/24.11/profilers/Nsight_Systems/bin/nsys

# Clean up old files silently (-f suppresses the "No such file" errors)
rm -f *.sqlite *.nsys-rep *.log

# Run the profiling
mpirun -np 2 $NSYS profile -o ipic3dTrace_with_metrics_%q{OMPI_COMM_WORLD_RANK} --trace=cuda,nvtx,osrt,mpi --gpu-metrics-device=all ./build/iPIC3D testGEM3D.inp > testGEM3D.log 2>&1

# Generate statistics for each rank's report individually
for report in ipic3dTrace_with_metrics_*.nsys-rep; do
    echo "========================================"
    echo " Generating Stats for: $report"
    echo "========================================"
    $NSYS stats "$report"
done
