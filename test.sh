
cd src/unitTest/gpu_maxwell_solver/
rm -rd build && mkdir build && cd build
cmake .. -DUSE_ADIOS2=OFF -DUSE_HDF5=OFF -DCUDA_ARCH=80 -DGPU_SOLVER=ON -DHALO_OVERLAP=ON -DCUDA_GRAPH=ON -DUSE_NCCL=ON
#make 2>&1 | tee build.log
make -j
export IPIC_FORCE_GPU_MPI=1

export OMP_NUM_THREADS=6
#CUDA_LAUNCH_BLOCKING=1 mpirun -n 2 compute-sanitizer --tool memcheck ./gpuMaxwellSolverTest --case periodic 2>&1 | head -40
mpirun -n 2 ./gpuMaxwellSolverTest --case all
cd ../../../../

