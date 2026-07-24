#!/bin/bash -l
#SBATCH -J iPIC3D
#SBATCH -t 00:18:00
#SBATCH --account=project_465002552
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH -o stdout
#SBATCH -e stderr

module load LUMI/25.03  partition/G
module load PrgEnv-cray/8.6.0
module load craype-accel-amd-gfx90a
module load rocm/6.3.4

export MPICH_GPU_SUPPORT_ENABLED=1
module load cray-mpich/8.1.32
# RCCL / NCCL tuning (LUMI-specific, per De Sensi et al. SC24)
# export NCCL_IGNORE_CPU_AFFINITY=1
# export NCCL_NET_GDR_LEVEL=3
# export NCCL_NCHANNELS_PER_PEER=32
# --- GPU-selection wrapper ---
cat <<'EOF' > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export HIP_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x ./select_gpu

# --- rocprofv3 wrapper: gives each rank its own output dir/file ---
mkdir -p ./rocprof_traces
cat <<'EOF' > profile_wrap
#!/bin/bash
rocprofv3 --output-format pftrace \
          --summary \
          --hip-trace \
          --hsa-trace \
          --kernel-trace \
          --memory-copy-trace \
          --marker-trace \
          --rccl-trace \
          -d ./rocprof_traces \
          -o trace_rank${SLURM_PROCID} \
          -- "$@"
EOF
chmod +x ./profile_wrap
# CPU_BIND="mask_cpu:7e,7e00"

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
##CPU_BIND="${CPU_BIND},7e00000000"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=6
export IPIC_FORCE_GPU_MPI=1

executable=/users/bouziane/iPIC3D-GPU/build/iPIC3D
inputfile=/users/bouziane/iPIC3D-GPU/testGEM3D.inp
output_file=/users/bouziane/iPIC3D-GPU/testGEM3D.log

# sanity check GPU binding before the real run
srun --cpu-bind=${CPU_BIND} ./profile_wrap ${executable} ${inputfile} > ${output_file}

# profiled run: select_gpu -> profile_wrap -> rocprofv3 -> executable
# srun --cpu-bind=${CPU_BIND}  ./profile_wrap ${executable} ${inputfile} > ${output_file}

rm -rf ./select_gpu ./profile_wrap