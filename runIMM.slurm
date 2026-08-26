#!/bin/bash -l
#SBATCH -J iPIC3D
#SBATCH -t 00:18:00
#SBATCH --account=project_465002552
#SBATCH --partition=standard-g
#SBATCH --nodes=3              # Total number of nodes 
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node
#SBATCH --gpus-per-node=8 
#SBATCH --cpus-per-task=6
#SBATCH -o stdout
#SBATCH -e stderr

module load LUMI/25.03  partition/G
module load PrgEnv-cray/8.6.0
module load craype-accel-amd-gfx90a
module load rocm/6.3.4

##############
###################
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET_GDR_LEVEL=3
export NCCL_NCHANNELS_PER_PEER=32

## DEBUGGING RCCL
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH
###################
### in case we are using mutiple nodes , it prevents the fall back to tcp sockets for communication



# Force RCCL to use the OFI plugin for Slingshot
# export NCCL_NET_GDR_LEVEL=3
# export FI_CXI_RX_MATCH_MODE=software

# export NCCL_BUFFSIZE=2097152
##########
export MPICH_GPU_SUPPORT_ENABLED=1
module load cray-mpich/8.1.32

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

srun bash -c 'echo "rank=$SLURM_PROCID local=$SLURM_LOCALID ROCR=$ROCR_VISIBLE_DEVICES HIP=$HIP_VISIBLE_DEVICES";'
srun  --cpu-bind=${CPU_BIND} ${executable} ${inputfile} > ${output_file}
rm -rf ./select_gpu