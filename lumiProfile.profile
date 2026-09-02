module --force purge
module load LUMI/25.09  partition/G
module load lumi-tools/24.05 
module load PrgEnv-cray/8.6.0
module load craype-accel-amd-gfx90a
module load rocm/6.4.4

export MPICH_GPU_SUPPORT_ENABLED=1
module load cray-mpich/9.0.1
module load buildtools/25.09
# module load cray-hdf5-parallel/1.14.3.5
# ###################
# export NCCL_IGNORE_CPU_AFFINITY=1
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_NCHANNELS_PER_PEER=32

# ## DEBUGGING RCCL
# # export NCCL_DEBUG=INFO
# # export NCCL_DEBUG_SUBSYS=INIT,GRAPH
# ###################

export MPICH_GPU_SUPPORT_ENABLED=1

export PATH=$ROCM_PATH/llvm/bin:$PATH
export CC=hipcc
export CXX=hipcc

export CXXFLAGS="$CXXFLAGS -I${MPICH_DIR}/include"
export HIPFLAGS="$CXXFLAGS --offload-arch=gfx90a"
export LDFLAGS="$LDFLAGS -L${MPICH_DIR}/lib -lmpi ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a}"

# cmake .. -DHIP_ON=ON  -DUSE_ADIOS2=OFF -DUSE_HDF5=OFF -DAMDGPU_TARGETS="gfx90a" -DHIP_ARCH="gfx90a" -DGPU_SOLVER=ON -DCUDA_GRAPH=ON -DHALO_OVERLAP=ON -DUSE_NCCL=ON
