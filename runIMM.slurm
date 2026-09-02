#!/bin/bash -l
#SBATCH -J iPIC3D
#SBATCH -t 00:18:00
#SBATCH --account=project_465002552
#SBATCH --partition=standard-g
#SBATCH --nodes=1              # Total number of nodes 
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node
#SBATCH --gpus-per-node=8 
#SBATCH --cpus-per-task=6
#SBATCH -o stdout
#SBATCH -e stderr

###############################
# libiPIC3Dlib.so is linked against LUMI/25.09 + rocm/6.4.4 + cray-mpich/9.0.1
# (see CMakeCache.txt / readelf -d build/libiPIC3Dlib.so). lumiGPUrocm644.profile
# below loads that exact stack via `module --force purge`, so no separate
# LUMI/25.03 module loads belong here - they'd only get purged anyway, and
# reintroducing cray-mpich/8.1.32 later doesn't even exist under LUMI/25.09.

rccl_setup_root=/pfs/lustrep4/users/bouziane/RCCLtest
source "${rccl_setup_root}/lumiGPUrocm644.profile"

# The aws-ofi-nccl plugin is only for INTER-node traffic over Slingshot/CXI.
# On a single node, RCCL talks to the other 7 local GPUs over P2P/XGMI and
# never needs libfabric at all; forcing NCCL_NET=OFI here made every rank on
# the same node fight to open a CXI domain on the same NICs, which is exactly
# what threw "NET/OFI Couldn't create domain: ENOSYS". Only force OFI when
# actually spanning multiple nodes (see RCCLtest README: "For a single-node
# job ... omit the remaining plugin setup and --network option").
srun_network_opt=()
if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
  export OFI_RCCL_PREFIX="${rccl_setup_root}/.local/aws-ofi-nccl-1.21.1-rocm-6.4.4"
  export NCCL_NET_PLUGIN="${OFI_RCCL_PREFIX}/lib/librccl-net.so"
  if [[ ! -r "${NCCL_NET_PLUGIN}" ]]; then
      echo "Plugin not found: ${NCCL_NET_PLUGIN}" >&2
      exit 1
  fi
  export LD_LIBRARY_PATH="${OFI_RCCL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

  export NCCL_NET=OFI
  export FI_PROVIDER=cxi
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,GRAPH
  # LUMI-G Slingshot tuning (RCCLtest README "Configuration defaults").
  export HSA_FORCE_FINE_GRAIN_PCIE=1
  export FI_MR_CACHE_MONITOR=userfaultfd
  export FI_CXI_DISABLE_HOST_REGISTER=1
  export FI_CXI_DEFAULT_CQ_SIZE=131072
  export FI_CXI_RDZV_PROTO=alt_read
  export FI_CXI_RDZV_EAGER_SIZE=0
  export FI_CXI_RDZV_THRESHOLD=0
  export FI_CXI_RDZV_GET_MIN=0
  export FI_CXI_DEFAULT_TX_SIZE=2048
  export FI_CXI_RX_MATCH_MODE=hybrid
  export NCCL_CROSS_NIC=1
  export NCCL_NET_GDR_LEVEL=PHB
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

  srun_network_opt=(--network=disable_rdzv_get)
fi

export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NCHANNELS_PER_PEER=32
export MPICH_GPU_SUPPORT_ENABLED=1

cat <<'EOF' > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export HIP_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF


chmod +x ./select_gpu

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
##CPU_BIND="${CPU_BIND},7e00000000"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"
 

export OMP_NUM_THREADS=6
export MPICH_GPU_SUPPORT_ENABLED=1

export IPIC_FORCE_GPU_MPI=1
executable=/users/bouziane/iPIC3D-GPU/build/iPIC3D
inputfile=/users/bouziane/iPIC3D-GPU/testGEM3D.inp
output_file=/users/bouziane/iPIC3D-GPU/testGEM3D.log

srun "${srun_network_opt[@]}" bash -c 'echo "rank=$SLURM_PROCID local=$SLURM_LOCALID ROCR=$ROCR_VISIBLE_DEVICES HIP=$HIP_VISIBLE_DEVICES";'
srun "${srun_network_opt[@]}" --cpu-bind=${CPU_BIND} ${executable} ${inputfile} > ${output_file}
rm -rf ./select_gpu