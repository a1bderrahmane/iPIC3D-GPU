#!/bin/bash -l
#SBATCH -J iPIC3D-att
#SBATCH -t 00:20:00
#SBATCH --account=project_465002552
#SBATCH --partition=standard-g
#SBATCH --nodes=1              # Must match XLEN*YLEN*ZLEN=8 in testGEM3D_kernels.inp
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH -o stdout_att
#SBATCH -e stderr_att

# ============================================================================
# Instruction-level GPU profiling (rocprofv3 --att / "Advanced Thread Trace")
#
# This produces the raw .att trace files that AMD's ROCprof Compute Viewer
# (https://github.com/ROCm/rocprof-compute-viewer) opens for instruction-level
# analysis (per-instruction hitcount/latency, occupancy, stalls).
#
# HOW TO VIEW THE RESULT:
#   1. This job writes traces to ./att_traces/ on LUMI.
#   2. Copy that directory to a machine with a display, e.g.:
#        rsync -av lumi:/users/bouziane/iPIC3D-GPU/att_traces/ ./att_traces/
#   3. Build/run rocprof-compute-viewer locally (see its README) and open the
#      att_traces/ directory (or point it at the individual .att/.out files).
#
# WHY THIS MODULE COMBO:
#   LUMI's default rocm/6.3.4 module ships an older rocprofv3 build that does
#   not expose --att (no bundled trace-decoder plugin). rocm/6.4.4 (only
#   available under the experimental LUMI/25.09 stack) does expose --att, but
#   switching to LUMI/25.09 wholesale would also switch cray-mpich from
#   8.1.32 -> 9.0.1 and cce 19->20, which the iPIC3D binary was NOT built
#   against. To avoid an MPI/compiler-runtime mismatch, we keep the proven
#   LUMI/25.03 + rocm/6.3.4 + cray-mpich/8.1.32 environment (same as
#   profileIMM.slurm/runIMM.slurm) and only prepend rocm/6.4.4's rocprofv3
#   tool + its HIP/HSA runtime libs onto PATH/LD_LIBRARY_PATH so the app and
#   the profiler share a consistent ROCm 6.4 runtime while MPI stays 8.1.32.
#
#   The decoder plugin itself (librocprof-trace-decoder.so, from
#   https://github.com/ROCm/rocprof-trace-decoder) is not bundled in any
#   LUMI module -- it was downloaded once into env/rocprof_tools/lib and
#   symlinked to the name rocprofv3 6.4.4 looks for (libatt_decoder_trace.so).
#
#   IMPORTANT (confirmed by real test runs on this cluster): LUMI's
#   rocm/6.4.4 is an early/experimental rocprofiler-sdk build (v0.6.0) whose
#   tool library expects decoder symbols named `rocprofiler_att_decoder_*`,
#   while every public rocprof-trace-decoder release exports
#   `rocprof_trace_decoder_*` (renamed before AMD open-sourced it). rocprofv3
#   also *requires* an --att-parse choice whenever --att is used -- it will
#   refuse to run at all otherwise ("provide the parser choice"). The result:
#   this job WILL always end with rank 0 hitting SIGABRT ("Decoder library
#   not found...") and the whole srun step showing CANCELLED in `sacct`.
#   THIS IS EXPECTED AND HARMLESS: the abort happens during an on-cluster
#   JSON-decode step that runs strictly *after* the raw trace files
#   (.att/.out/agent_info.csv) are already fully written to ./att_traces/.
#   rocprof-compute-viewer decodes those raw files itself (with its own
#   correctly-named decoder) once you open them locally -- the on-cluster
#   decode this script skips past isn't needed for that. `|| true` below
#   keeps the script going past the expected abort so it can report what
#   was actually collected.
# ============================================================================

module load LUMI/25.03  partition/G
module load PrgEnv-cray/8.6.0
module load craype-accel-amd-gfx90a
module load rocm/6.3.4

export MPICH_GPU_SUPPORT_ENABLED=1
module load cray-mpich/8.1.32

# --- Pull in rocprofv3 from rocm/6.4.4 (has ATT support) without switching
#     the rest of the programming environment away from what iPIC3D was built with.
ROCM64_DIR=/appl/lumi/SW/LUMI-25.09/G/EB/rocm/6.4.4
export PATH=${ROCM64_DIR}/bin:${PATH}
export LD_LIBRARY_PATH=${ROCM64_DIR}/lib:${LD_LIBRARY_PATH}

# --- Trace decoder plugin (downloaded from ROCm/rocprof-trace-decoder releases,
#     symlinked to the filename rocprofv3 auto-detects: libatt_decoder_trace.so)
export ROCPROF_ATT_LIBRARY_PATH=/users/bouziane/iPIC3D-GPU/env/rocprof_tools/lib

# --- GPU-selection wrapper ---
cat <<'EOF' > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export HIP_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x ./select_gpu

# --- ATT profiling wrapper: only rank 0 is traced (ATT is heavy per-GPU;
#     tracing all 8 ranks at once would be both enormous and pointless --
#     the kernels are identical SPMD code across ranks). Adjust
#     PROFILE_RANK below to trace a different rank instead.
mkdir -p ./att_traces
PROFILE_RANK=0
cat <<EOF > profile_wrap
#!/bin/bash
if [ "\$SLURM_PROCID" = "${PROFILE_RANK}" ]; then
  exec rocprofv3 \\
    --att \\
    --att-target-cu 0 \\
    --att-simd-select 0 \\
    --att-parse trace \\
    --kernel-include-regex "mover" \\
    --kernel-iteration-range "[1-3]" \\
    -d ./att_traces \\
    -o att_rank\${SLURM_PROCID} \\
    -- "\$@"
else
  exec "\$@"
fi
EOF
chmod +x ./profile_wrap

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=6
export IPIC_FORCE_GPU_MPI=1

executable=/users/bouziane/iPIC3D-GPU/build/iPIC3D
inputfile=/users/bouziane/iPIC3D-GPU/testGEM3D_kernels.inp
output_file=/users/bouziane/iPIC3D-GPU/testGEM3D_kernels.log

# profile_wrap -> (rocprofv3 --att, rank ${PROFILE_RANK} only) -> executable
# (matches profileIMM.slurm/runIMM.slurm: select_gpu is created above for
#  reference/manual use, but Slurm's own GPU binding already scopes each
#  rank to one GPU, so it isn't inserted into the srun pipeline here either.)
# `|| true`: rank 0 is expected to SIGABRT after the raw trace files are
# already written (see the note at the top of this file), which would
# otherwise make `srun` return non-zero and stop the script before it gets
# to report what was collected.
srun --cpu-bind=${CPU_BIND} ./profile_wrap ${executable} ${inputfile} > ${output_file} || true

rm -rf ./select_gpu ./profile_wrap

echo "ATT trace(s) for rank ${PROFILE_RANK} (expect a SIGABRT further up in stderr -- that's normal, see top-of-file note):"
ls -la ./att_traces/ || true
