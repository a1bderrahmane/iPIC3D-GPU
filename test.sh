
cd src/unitTest/gpu_maxwell_solver/
rm -rd build && mkdir build && cd build
cmake .. -DUSE_ADIOS2=OFF -DUSE_HDF5=OFF -DCUDA_ARCH=80 -DGPU_SOLVER=ON
make -j
mpirun -n 2 ./gpuMaxwellSolverTest --case all

