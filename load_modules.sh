module purge --force
module load git/2.37.0-gcc-11.3.0-iri7lvp
module load cmake/3.27.9-gcc-13.2.0-ltey7k7
module load cuda/12.3.0-gcc-13.2.0-h4jfv2k 
module load openmpi/4.1.6-gcc-12.2.0-kvteryp
module load gcc/12.2.0-gcc-11.3.0-n5zahxh
module load nvtop/2.0.2-gcc-11.3.0-why26pe

export NCCL_ROOT=/local/spack/linux-centos8-zen2/gcc-11.3.0/nccl-2.14.3-1-myx6s7euskmy66e7zy6nypaa7672aitn
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}
