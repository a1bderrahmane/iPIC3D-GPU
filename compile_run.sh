module purge --force
module load git/2.37.0-gcc-11.3.0-iri7lvp
module load cmake/3.27.9-gcc-13.2.0-ltey7k7
module load cuda/12.3.0-gcc-13.2.0-h4jfv2k 
module load openmpi/4.1.6-gcc-12.2.0-kvteryp
module load gcc/12.2.0-gcc-11.3.0-n5zahxh
module load nvtop/2.0.2-gcc-11.3.0-why26pe

# NCCL: export paths directly instead of "module load nccl" because that
# module pulls in cuda/11.7.99, which would shadow the cuda/12.3.0 toolchain.
export NCCL_ROOT=/local/spack/linux-centos8-zen2/gcc-11.3.0/nccl-2.14.3-1-myx6s7euskmy66e7zy6nypaa7672aitn
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}

rm -rd build
mkdir build && cd build
cmake .. -DUSE_ADIOS2=OFF -DUSE_HDF5=OFF -DCUDA_ARCH=80 -DGPU_SOLVER=ON -DHALO_OVERLAP=ON -DCUDA_GRAPH=OFF -DUSE_NCCL=OFF
make -j
cd ..
export IPIC_FORCE_GPU_MPI=1

export OMP_NUM_THREADS=6
mpirun   -n 2  --bind-to none --mca pml ob1 --mca btl self,vader,smcuda build/iPIC3D testGEM3D.inp > testGEM3D.log 2>&1


