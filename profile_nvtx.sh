#!/bin/bash
NSYS=/local/spack/linux-centos8-zen2/gcc-14.2.0/nvhpc-24.11-mcsqyuj3eddzlmyxx3kgswr2fawib3sf/Linux_x86_64/24.11/profilers/Nsight_Systems/bin/nsys

rm -f *.sqlite *.nsys-rep *.log

mpirun -np 2 $NSYS profile \
  -o ipic3dTrace_%q{OMPI_COMM_WORLD_RANK} \
  --trace=cuda,nvtx,osrt,mpi \
  --cuda-graph-trace=node \
  --gpu-metrics-device=all \
  --gpu-metrics-frequency=10000 \
  --force-overwrite=true \
  --stats=false \
  ./build/iPIC3D testGEM3D.inp > testGEM3D.log 2>&1

echo "Checking for generated files..."

ls -la ipic3D_*.nsys-rep 2>/dev/null || echo "No .nsys-rep files found"
ls -la *.sqlite 2>/dev/null || echo "No .sqlite files found"

for report in ipic3dTrace_*.nsys-rep; do
  echo "==== Stats: $report ===="
  $NSYS stats "$report"
done

 # --capture-range=cudaProfilerApi \
 # --capture-range-end=stop \
