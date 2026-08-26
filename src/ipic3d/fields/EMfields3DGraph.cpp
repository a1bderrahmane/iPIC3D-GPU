#ifdef GPU_SOLVER

#include <mpi.h>
#include "EMfields3D.h"
#include "Collective.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "Com3DNonblk.h"
#include "Parameters.h"
#include "TimeTasks.h"
#include "errors.h"

#include "cudaTypeDef.cuh"
#include "GPUSolverMPITypes.h"
#include "GPUBlas.cuh"
#include "GPUStencils.cuh"
#include "GPUPhysicsKernels.cuh"
#include "GPUMaxwellLocal.cuh"
#include "GPUChebyshev.cuh"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <MPIdata.h>
#include <chrono>
#ifdef CUDA_GRAPH
#ifdef HIPIFLY
#include <roctracer/roctx.h>
#else
#include <nvtx3/nvToolsExt.h>
#endif
#endif
namespace
{
    // Variant suffix appended to every timer label. A given binary only ever
    // runs ONE implementation of each timed function (traditional, cuda_graph
    // or nccl), selected by the compiler flags, so the label printed in the
    // summary records which one this run measured.
#if defined(CUDA_GRAPH) && defined(USE_NCCL)
#define GPU_TIMER_SUFFIX "_nccl"
#elif defined(CUDA_GRAPH)
#define GPU_TIMER_SUFFIX "_cuda_graph"
#else
#define GPU_TIMER_SUFFIX ""
#endif

    struct GpuStageTimer
    {
        const char *label;   // printed in the stderr summary (flag-dependent)
        const char *csvStem; // per-rank CSV file stem
        std::vector<double> ms;
        cudaEvent_t startEvt = nullptr;
        cudaEvent_t stopEvt = nullptr;
        int rank = 0;
        bool inited = false;

        GpuStageTimer(const char *label_, const char *csvStem_)
            : label(label_), csvStem(csvStem_) {}

        void init()
        {
            if (inited)
                return;
            ms.reserve(1 << 16);
            rank = MPIdata::get_rank();
            cudaErrChk(cudaEventCreate(&startEvt));
            cudaErrChk(cudaEventCreate(&stopEvt));
            inited = true;
        }
        // begin/end bracket work enqueued on `stream` with timing-enabled CUDA
        // events rather than host std::chrono. cudaGraphLaunch (and the NCCL
        // kernels it captures) is asynchronous, so a host-side chrono sandwich
        // around it only measures launch overhead, not actual GPU execution
        // time. Recording events *on the stream* and syncing on stopEvt in
        // end() captures the true GPU-side elapsed time instead.
        void begin(cudaStream_t stream)
        {
            init();
            cudaErrChk(cudaEventRecord(startEvt, stream));
        }
        void end(cudaStream_t stream)
        {
            cudaErrChk(cudaEventRecord(stopEvt, stream));
            cudaErrChk(cudaEventSynchronize(stopEvt));
            float elapsed_ms = 0.0f;
            cudaErrChk(cudaEventElapsedTime(&elapsed_ms, startEvt, stopEvt));
            ms.push_back((double)elapsed_ms);
        }

        ~GpuStageTimer() { report(); } // runs at program exit

        void report()
        {
            if (ms.empty())
                return;
            std::vector<double> v = ms;
            std::sort(v.begin(), v.end());
            const size_t n = v.size();

            double sum = 0.0;
            for (double x : v)
                sum += x;
            const double mean = sum / n;

            double var = 0.0;
            for (double x : v)
                var += (x - mean) * (x - mean);
            var /= (n > 1 ? n - 1 : 1); // sample stdev (n-1)
            const double sd = std::sqrt(var);

            const double median =
                (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
            fprintf(stderr,
                    "[rank %d] %s over %zu calls (ms):\n"
                    "sum = %.6f\n  mean/avg = %.6f\n  median   = %.6f\n"
                    "  min      = %.6f\n  max      = %.6f\n  stdev    = %.6f\n",
                    rank, label, n, sum, mean, median, v.front(), v.back(), sd);
            // Raw per-call samples, one rank-local file, for later analysis.
            char fname[256];
            std::snprintf(fname, sizeof(fname),
                          "%s_rank%d.csv", csvStem, rank);
            if (FILE *f = std::fopen(fname, "w"))
            {
                std::fprintf(f, "call,ms\n");
                for (size_t i = 0; i < ms.size(); ++i)
                    std::fprintf(f, "%zu,%.6f\n", i, ms[i]); // ms = call order, not sorted
                std::fclose(f);
            }
        }
    };
    // One timer per function family; each of the three implementations of a
    // family brackets its whole body with begin()/end(), and the flag-derived
    // suffix in the label says which implementation this binary ran.
    static GpuStageTimer g_maxwellImageTimer("gpuMaxwellImage" GPU_TIMER_SUFFIX,
                                             "maxwell_image_timing");
    static GpuStageTimer g_calculateBTimer("gpuCalculateB" GPU_TIMER_SUFFIX,
                                           "calculate_b_timing");
    static GpuStageTimer g_hatFunctionsTimer("gpuCalculateHatFunctions" GPU_TIMER_SUFFIX,
                                             "hat_functions_timing");
} // namespace

// #ifdef HALO_OVERLAP

// Forward declarations for BC face functions (defined in GPUHaloComm.cu).
// Cannot include GPUHaloComm.cuh here because it contains __global__ decls
// and this file is compiled by the host compiler.
void gpuBCface(int nx, int ny, int nz, GPUFieldArray3 &gpuArr,
               int bcFaceXright, int bcFaceXleft,
               int bcFaceYright, int bcFaceYleft,
               int bcFaceZright, int bcFaceZleft,
               const VirtualTopology3D *vct,
               cudaStream_t stream);
void gpuBCface_P(int nx, int ny, int nz, GPUFieldArray3 &gpuArr,
                 int bcFaceXright, int bcFaceXleft,
                 int bcFaceYright, int bcFaceYleft,
                 int bcFaceZright, int bcFaceZleft,
                 const VirtualTopology3D *vct,
                 cudaStream_t stream);
// #endif // HALO_OVERLAP

#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// =========================================================================
//  GPU Solver: allocation / deallocation / synchronisation
// =========================================================================

#if defined(CUDA_GRAPH) && defined(USE_NCCL)
void EMfields3D::gpuNcclInit()
{
    if (ncclInitialized_)
        return;

    const VirtualTopology3D *vct = &get_vct();
    MPI_Comm fieldcomm = vct->getFieldComm();

    
    int rank, size;
    MPI_Comm_rank(fieldcomm, &rank);
    MPI_Comm_size(fieldcomm, &size);

    // NCCL requires exactly ONE rank per GPU
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(fieldcomm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int nodeSize, myDev;
        MPI_Comm_size(nodeComm, &nodeSize);
        cudaErrChk(cudaGetDevice(&myDev));
        std::vector<int> devs(nodeSize);
        MPI_Allgather(&myDev, 1, MPI_INT, devs.data(), 1, MPI_INT, nodeComm);
        MPI_Comm_free(&nodeComm);
        if (std::count(devs.begin(), devs.end(), myDev) > 1)
            eprintf("USE_NCCL requires one MPI rank per GPU, but multiple ranks on "
                    "this node share CUDA device %d. Run with at most one rank per "
                    "GPU, or rebuild with -DUSE_NCCL=OFF to use the MPI halo path.",
                    myDev);
    }

    setenv("NCCL_GRAPH_REGISTER", "1", 0);

    ncclUniqueId id;
    if (rank == 0)
        NCCLCHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, fieldcomm);

    NCCLCHECK(ncclCommInitRank(&fieldNcclComm_, size, id, rank));

    // Peer ranks: identical values you already read for MPI_Isend/Irecv.
    ncclPeerXL_ = vct->getXleft_neighbor();
    ncclPeerXR_ = vct->getXright_neighbor();
    ncclPeerYL_ = vct->getYleft_neighbor();
    ncclPeerYR_ = vct->getYright_neighbor();
    ncclPeerZL_ = vct->getZleft_neighbor();
    ncclPeerZR_ = vct->getZright_neighbor();

    // Secondary streams for the fork/join overlap pattern.
    cudaErrChk(cudaStreamCreateWithFlags(&ncclCommStream_, cudaStreamNonBlocking));
    cudaErrChk(cudaStreamCreateWithFlags(&interiorStream_, cudaStreamNonBlocking));
    cudaErrChk(cudaEventCreateWithFlags(&ncclForkEvent_, cudaEventDisableTiming));
    cudaErrChk(cudaEventCreateWithFlags(&ncclJoinEvent_, cudaEventDisableTiming));

    ncclInitialized_ = true;
}

void EMfields3D::gpuNcclFree()
{
    if (!ncclInitialized_)
        return;
    if (maxwellImageNcclExec_)
    {
        cudaGraphExecDestroy(maxwellImageNcclExec_);
        maxwellImageNcclExec_ = nullptr;
    }
    if (ncclForkEvent_)
        cudaEventDestroy(ncclForkEvent_);
    if (ncclJoinEvent_)
        cudaEventDestroy(ncclJoinEvent_);
    if (ncclCommStream_)
        cudaStreamDestroy(ncclCommStream_);
    if (interiorStream_)
        cudaStreamDestroy(interiorStream_);
    ncclCommDestroy(fieldNcclComm_);
    ncclInitialized_ = false;
}
#endif // CUDA_GRAPH && USE_NCCL

void EMfields3D::gpuSolverFree()
{
    if (!gpuSolverAllocated_)
        return;
#ifdef CUDA_GRAPH
    // Drain all outstanding work on solverStream_/interiorStream_ before
    // tearing down anything below. gpuNcclFree() (called at the end of this
    // block, after every cudaGraphExecDestroy) destroys the NCCL communicator
    // and the shared fork/join events/streams that bNcclExec_ and every
    // hatNcclExec_[is] graph reference internally -- destroying those graphs
    // (or leaving graph-launched work in flight) after the comm/events/streams
    // are gone is undefined behavior and was observed to hang at shutdown.
    cudaErrChk(cudaStreamSynchronize(solverStream_));
#ifdef USE_NCCL
    if (interiorStream_)
        cudaErrChk(cudaStreamSynchronize(interiorStream_));
#endif
    if (s1Exec_)
    {
        cudaGraphExecDestroy(s1Exec_);
        s1Exec_ = nullptr;
    }
    if (s2Exec_)
    {
        cudaGraphExecDestroy(s2Exec_);
        s2Exec_ = nullptr;
    }
    if (s5Exec_)
    {
        cudaGraphExecDestroy(s5Exec_);
        s5Exec_ = nullptr;
    }
    if (bFieldUpdateExec_)
    {
        cudaGraphExecDestroy(bFieldUpdateExec_);
        bFieldUpdateExec_ = nullptr;
    }
    if (bInteriorExec_)
    {
        cudaGraphExecDestroy(bInteriorExec_);
        bInteriorExec_ = nullptr;
    }
    if (bBcPostExec_)
    {
        cudaGraphExecDestroy(bBcPostExec_);
        bBcPostExec_ = nullptr;
    }
#ifdef USE_NCCL
    if (bNcclExec_)
    {
        cudaGraphExecDestroy(bNcclExec_);
        bNcclExec_ = nullptr;
    }
#endif
    if (hatInteriorExec_)
    {
        cudaGraphExecDestroy(hatInteriorExec_);
        hatInteriorExec_ = nullptr;
    }
    if (hatBoundaryExec_)
    {
        cudaGraphExecDestroy(hatBoundaryExec_);
        hatBoundaryExec_ = nullptr;
    }
    if (hatBlockingExec_)
    {
        cudaGraphExecDestroy(hatBlockingExec_);
        hatBlockingExec_ = nullptr;
    }
    if (hatRhohatExec_)
    {
        cudaGraphExecDestroy(hatRhohatExec_);
        hatRhohatExec_ = nullptr;
    }
    for (auto &g : hatPreExec_)
        if (g)
            cudaGraphExecDestroy(g);
    for (auto &g : hatPostExec_)
        if (g)
            cudaGraphExecDestroy(g);
    hatPreExec_.clear();
    hatPostExec_.clear();

#ifdef USE_NCCL
    for (auto &g : hatNcclExec_)
        if (g)
            cudaGraphExecDestroy(g);
    hatNcclExec_.clear();
    if (maxwellImageNcclExec_)
    {
        cudaGraphExecDestroy(maxwellImageNcclExec_);
        maxwellImageNcclExec_ = nullptr;
    }

    // Per-graph device pointer arrays: safe to free only after every graph
    // exec that reads them (above) is destroyed.
    if (d_maxwellNcclPtrs_)
    {
        cudaFree(d_maxwellNcclPtrs_);
        d_maxwellNcclPtrs_ = nullptr;
    }
    if (d_bNcclPtrs_)
    {
        cudaFree(d_bNcclPtrs_);
        d_bNcclPtrs_ = nullptr;
    }
    if (d_hatNcclPtrs_)
    {
        cudaFree(d_hatNcclPtrs_);
        d_hatNcclPtrs_ = nullptr;
    }

    // Must run LAST: destroys fieldNcclComm_ and the shared fork/join
    // events/streams that the graph execs above were captured against.
    gpuNcclFree();
#endif
#endif
    // Electric field
    d_Ex.free();
    d_Ey.free();
    d_Ez.free();
    d_Exth.free();
    d_Eyth.free();
    d_Ezth.free();

    // Magnetic field
    d_Bxc.free();
    d_Byc.free();
    d_Bzc.free();
    d_Bxn.free();
    d_Byn.free();
    d_Bzn.free();

    // Charge / current densities
    d_rhon.free();
    d_rhoc.free();
    d_rhoh.free();
    d_Jx.free();
    d_Jy.free();
    d_Jz.free();
    d_Jxh.free();
    d_Jyh.free();
    d_Jzh.free();

    // Per-species
    d_rhons.free();
    d_Jxs.free();
    d_Jys.free();
    d_Jzs.free();
    d_pXXsn.free();
    d_pXYsn.free();
    d_pXZsn.free();
    d_pYYsn.free();
    d_pYZsn.free();
    d_pZZsn.free();

    // Potentials
    d_PHI.free();
    d_PSI.free();

    // External B
    d_Bx_ext.free();
    d_By_ext.free();
    d_Bz_ext.free();

    // Temporary arrays
    d_tempXC.free();
    d_tempYC.free();
    d_tempZC.free();
    d_tempXN.free();
    d_tempYN.free();
    d_tempZN.free();
    d_tempC.free();
    d_tempX.free();
    d_tempY.free();
    d_tempZ.free();
    d_temp2X.free();
    d_temp2Y.free();
    d_temp2Z.free();
    d_imageX.free();
    d_imageY.free();
    d_imageZ.free();
    d_Dx.free();
    d_Dy.free();
    d_Dz.free();
    d_vectX.free();
    d_vectY.free();
    d_vectZ.free();
    d_divC.free();

    // divB cleaning
    d_divBwork.free();
    d_gradPSIX.free();
    d_gradPSIY.free();
    d_gradPSIZ.free();

    // Krylov
    d_xkrylovMaxwell.free();
    d_bkrylovMaxwell.free();
    d_xkrylovPoisson_B.free();
    d_bkrylovPoisson_B.free();
    d_xkrylovPoisson_E.free();
    d_bkrylovPoisson_E.free();

    // calculateE work
    d_divE_work.free();
    d_gradPHIX_work.free();
    d_gradPHIY_work.free();
    d_gradPHIZ_work.free();

    // Poisson image
    d_poissonTemp.free();
    d_poissonIm.free();

    // Smooth temp
    d_smoothTemp.free();

    // qom device copy
    if (d_qom)
    {
        cudaFree(d_qom);
        d_qom = nullptr;
    }
    if (d_blasScratch)
    {
        cudaFree(d_blasScratch);
        d_blasScratch = nullptr;
    }
    if (d_gmresV)
    {
        cudaFree(d_gmresV);
        d_gmresV = nullptr;
    }
    if (d_gmresW)
    {
        cudaFree(d_gmresW);
        d_gmresW = nullptr;
    }
    gmresVAlloc = 0;

    // Free Chebyshev workspace
    if (d_chebY)
    {
        cudaFree(d_chebY);
        d_chebY = nullptr;
    }
    if (d_chebW)
    {
        cudaFree(d_chebW);
        d_chebW = nullptr;
    }
    if (d_chebZ)
    {
        cudaFree(d_chebZ);
        d_chebZ = nullptr;
    }
    if (d_chebTmp)
    {
        cudaFree(d_chebTmp);
        d_chebTmp = nullptr;
    }
    chebAlloc = 0;

    // Free FGMRES workspace
    if (d_fgmresZ)
    {
        cudaFree(d_fgmresZ);
        d_fgmresZ = nullptr;
    }
    fgmresZAlloc = 0;

    // Free Block-Jacobi scratch
    if (d_bjScratch1)
    {
        cudaFree(d_bjScratch1);
        d_bjScratch1 = nullptr;
    }
    if (d_bjScratch2)
    {
        cudaFree(d_bjScratch2);
        d_bjScratch2 = nullptr;
    }
    bjScratchAlloc = 0;

    // Free pinned GMRES host buffers
    if (h_gmresReduceLocal)
    {
        cudaFreeHost(h_gmresReduceLocal);
        h_gmresReduceLocal = nullptr;
    }
    if (h_gmresReduceGlobal)
    {
        cudaFreeHost(h_gmresReduceGlobal);
        h_gmresReduceGlobal = nullptr;
    }
    if (h_gmresH)
    {
        cudaFreeHost(h_gmresH);
        h_gmresH = nullptr;
    }
    if (h_gmresG)
    {
        cudaFreeHost(h_gmresG);
        h_gmresG = nullptr;
    }
    if (h_gmresCS)
    {
        cudaFreeHost(h_gmresCS);
        h_gmresCS = nullptr;
    }
    if (h_gmresSN)
    {
        cudaFreeHost(h_gmresSN);
        h_gmresSN = nullptr;
    }
    if (h_gmresY)
    {
        cudaFreeHost(h_gmresY);
        h_gmresY = nullptr;
    }

    // Free batched halo buffers
    gpuFreeHaloBuffers();

    // Destroy solver stream
    if (solverStream_)
    {
        cudaStreamDestroy(solverStream_);
        solverStream_ = 0;
    }

    gpuSolverAllocated_ = false;
}

// =========================================================================
//  Persistent batched halo-exchange buffer management
// =========================================================================

void EMfields3D::gpuAllocateHaloBuffers()
{
    if (haloBufsAllocated_)
        return;

    // Compute max per-field element count per direction across ALL phases
    // (face, edge, corner) to avoid overflow for skinny local domains.
    //
    // Face phase (per field):
    //   Dir 0,1 (XL,XR): (nyn-2)*(nzn-2)
    //   Dir 2,3 (YL,YR): (nxn-2)*(nzn-2)
    //   Dir 4,5 (ZL,ZR): (nxn-2)*(nyn-2)
    // Edge phase (per field, worst case both cross-edges active):
    //   Dir 0,1: 2*(nyn-2)   [Y-edges to X neighbours]
    //   Dir 2,3: 2*(nzn-2)   [Z-edges to Y neighbours]
    //   Dir 4,5: 2*(nxn-2)   [X-edges to Z neighbours]
    // Corner phase (per field): 4 per direction

    size_t faceSz[6], edgeSz[6];
    faceSz[0] = faceSz[1] = (size_t)(nyn - 2) * (nzn - 2);
    faceSz[2] = faceSz[3] = (size_t)(nxn - 2) * (nzn - 2);
    faceSz[4] = faceSz[5] = (size_t)(nxn - 2) * (nyn - 2);

    edgeSz[0] = edgeSz[1] = 2 * (size_t)(nyn - 2);
    edgeSz[2] = edgeSz[3] = 2 * (size_t)(nzn - 2);
    edgeSz[4] = edgeSz[5] = 2 * (size_t)(nxn - 2);

    constexpr size_t cornerSz = 4; // 4 corners per direction

    for (int d = 0; d < 6; ++d)
    {
        size_t maxPerField = faceSz[d];
        if (edgeSz[d] > maxPerField)
            maxPerField = edgeSz[d];
        if (cornerSz > maxPerField)
            maxPerField = cornerSz;
        size_t bytes = maxPerField * HALO_MAX_BATCH * sizeof(cudaSolverType);
        cudaErrChk(cudaMalloc(&d_haloBuf_send_[d], bytes));
        cudaErrChk(cudaMalloc(&d_haloBuf_recv_[d], bytes));
    }
    cudaErrChk(cudaMalloc(&d_ptrArray_, HALO_MAX_BATCH * sizeof(cudaSolverType *)));
    cudaErrChk(cudaHostAlloc(&h_ptrArray_, HALO_MAX_BATCH * sizeof(cudaSolverType *), cudaHostAllocDefault));

    haloBufsAllocated_ = true;
}

void EMfields3D::gpuFreeHaloBuffers()
{
    if (!haloBufsAllocated_)
        return;

    for (int d = 0; d < 6; ++d)
    {
        if (d_haloBuf_send_[d])
        {
            cudaFree(d_haloBuf_send_[d]);
            d_haloBuf_send_[d] = nullptr;
        }
        if (d_haloBuf_recv_[d])
        {
            cudaFree(d_haloBuf_recv_[d]);
            d_haloBuf_recv_[d] = nullptr;
        }
    }
    if (d_ptrArray_)
    {
        cudaFree(d_ptrArray_);
        d_ptrArray_ = nullptr;
    }
    if (h_ptrArray_)
    {
        cudaFreeHost(h_ptrArray_);
        h_ptrArray_ = nullptr;
    }

    haloBufsAllocated_ = false;
}

// =========================================================================
//  Fused triple Laplacian: 3 independent lap(fieldN) with ONE halo exchange
//  Uses 9 center-sized scratch arrays:
//    fieldA → d_tempXC / d_tempYC / d_tempZC
//    fieldB → d_divC   / d_poissonTemp / d_poissonIm
//    fieldC → d_divBwork / d_divE_work / d_tempC
// =========================================================================
//
void EMfields3D::gpuLapN2N_3_finish(GPUFieldArray3 &lapA,
                                    GPUFieldArray3 &lapB,
                                    GPUFieldArray3 &lapC)
{
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

#ifdef HALO_OVERLAP
    // ---- Begin halo exchange: pack faces + post MPI ----
    cudaSolverType *ptrs9[9] = {d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                                d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                                d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr()};
    gpuBatchedHaloBeginExchange(ptrs9, 9, nxc, nyc, nzc,
                                true, false, false, false, solverStream_);

    // ---- Interior divC2N while MPI is in flight ----
    gpuDivC2N_interior(lapA.devPtr(),
                       d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_interior(lapB.devPtr(),
                       d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_interior(lapC.devPtr(),
                       d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

    // ---- End halo exchange: MPI_Waitall + unpack + edges/corners ----
    gpuBatchedHaloEndExchange(ptrs9, 9, nxc, nyc, nzc,
                              true, false, false, false, solverStream_);

    // ---- BC face application (type 1 on all faces) ----
    gpuBCface(nxc, nyc, nzc, d_tempXC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempYC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempZC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_poissonTemp, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_poissonIm, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divBwork, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divE_work, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);

    // ---- Boundary divC2N (ghost + BC data now available) ----
    gpuDivC2N_boundary(lapA.devPtr(),
                       d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_boundary(lapB.devPtr(),
                       d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_boundary(lapC.devPtr(),
                       d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#else
    // ---- Original blocking path ----
    gpuCommunicateCenterBC_9(nxc, nyc, nzc,
                             d_tempXC, d_tempYC, d_tempZC,
                             d_divC, d_poissonTemp, d_poissonIm,
                             d_divBwork, d_divE_work, d_tempC,
                             1, 1, 1, 1, 1, 1);

    gpuDivC2N(lapA.devPtr(),
              d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
              nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N(lapB.devPtr(),
              d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
              nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N(lapC.devPtr(),
              d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
              nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#endif
}

void EMfields3D::gpuLapN2N_3_gradients(GPUFieldArray3 &fieldA,
                                       GPUFieldArray3 &fieldB,
                                       GPUFieldArray3 &fieldC)
{
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

    gpuGradN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
               fieldA.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
    gpuGradN2C(d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
               fieldB.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
    gpuGradN2C(d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
               fieldC.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
}
void EMfields3D::gpuLapN2N_3(
    GPUFieldArray3 &lapA, GPUFieldArray3 &fieldA,
    GPUFieldArray3 &lapB, GPUFieldArray3 &fieldB,
    GPUFieldArray3 &lapC, GPUFieldArray3 &fieldC)
{
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

    // Gradient A → d_tempXC, d_tempYC, d_tempZC
    gpuGradN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
               fieldA.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

    // Gradient B → d_divC, d_poissonTemp, d_poissonIm  (center-sized scratch)
    gpuGradN2C(d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
               fieldB.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

    // Gradient C → d_divBwork, d_divE_work, d_tempC  (center-sized scratch)
    gpuGradN2C(d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
               fieldC.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

#ifdef HALO_OVERLAP
    // ---- Begin halo exchange: pack faces + post MPI ----
    cudaSolverType *ptrs9[9] = {d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                                d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                                d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr()};
    gpuBatchedHaloBeginExchange(ptrs9, 9, nxc, nyc, nzc,
                                true, false, false, false, solverStream_);

    // ---- Interior divC2N while MPI is in flight ----
    gpuDivC2N_interior(lapA.devPtr(),
                       d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_interior(lapB.devPtr(),
                       d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_interior(lapC.devPtr(),
                       d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

    // ---- End halo exchange: MPI_Waitall + unpack + edges/corners ----
    gpuBatchedHaloEndExchange(ptrs9, 9, nxc, nyc, nzc,
                              true, false, false, false, solverStream_);

    // ---- BC face application (type 1 on all faces) ----
    gpuBCface(nxc, nyc, nzc, d_tempXC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempYC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempZC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_poissonTemp, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_poissonIm, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divBwork, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divE_work, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);

    // ---- Boundary divC2N (ghost + BC data now available) ----
    gpuDivC2N_boundary(lapA.devPtr(),
                       d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_boundary(lapB.devPtr(),
                       d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_boundary(lapC.devPtr(),
                       d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

#else
    // ---- Original blocking path ----
    gpuCommunicateCenterBC_9(nxc, nyc, nzc,
                             d_tempXC, d_tempYC, d_tempZC,
                             d_divC, d_poissonTemp, d_poissonIm,
                             d_divBwork, d_divE_work, d_tempC,
                             1, 1, 1, 1, 1, 1);

    // Divergence A
    gpuDivC2N(lapA.devPtr(),
              d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
              nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

    // Divergence B
    gpuDivC2N(lapB.devPtr(),
              d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
              nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

    // Divergence C
    gpuDivC2N(lapC.devPtr(),
              d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
              nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#endif
}

// =========================================================================
//  GPU MaxwellImage:  im = A * vector  (Krylov ↔ Krylov)
// =========================================================================
#if defined(CUDA_GRAPH) && defined(USE_NCCL)
void EMfields3D::gpuMaxwellImage_nccl(cudaSolverType *d_im, cudaSolverType *d_vector)
{
    // gpuNcclInit();  // no-op after first call
    g_maxwellImageTimer.begin(solverStream_);
    const Grid *grid = &get_grid();
    size_t nodeSize = (size_t)nxn * nyn * nzn;

    // ---- eager: Krylov -> physical (d_vector varies, cannot capture) ----
    gpuSolver2Phys3(d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                    d_vector, nxn, nyn, nzn, solverStream_);

    if (maxwellImageNcclExec_ == nullptr)
        gpuBuildMaxwellImageNcclGraph(d_im);

    nvtxRangePush("gpuMaxwellImage_nccl");
    cudaGraphLaunch(maxwellImageNcclExec_, solverStream_);
    nvtxRangePop();
    g_maxwellImageTimer.end(solverStream_);
}

void EMfields3D::gpuBuildMaxwellImageNcclGraph(cudaSolverType *d_im)
{
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();
    size_t nodeSize = (size_t)nxn * nyn * nzn;
    cudaSolverType *d_divD = d_PHI.devPtr();

    bool hasLeftX = (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0);
    bool hasRightX = (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0);
    bool hasLeftY = (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0);
    bool hasRightY = (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0);
    bool hasLeftZ = (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0);
    bool hasRightZ = (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0);

    // Immutable device pointer array for the captured halo exchange
    // (must exist before capture; lives as long as the graph).
    if (d_maxwellNcclPtrs_ == nullptr)
    {
        cudaSolverType *h10[10] = {
            d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
            d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
            d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
            d_divD};
        d_maxwellNcclPtrs_ = gpuMakeNcclPtrArray(h10, 10);
    }

    // NCCL P2P capture requires thread-local capture mode.
    cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal);

    // ---- g_pre: zero scratch, gradients, MUdot, div(D) -> d_PHI ----
    cudaSolverType *zptrs[9] = {d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                                d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                                d_Dx.devPtr(), d_Dy.devPtr(), d_Dz.devPtr()};
    gpuSetAll0_N(zptrs, 9, nodeSize, solverStream_);
    gpuLapN2N_3_gradients(d_vectX, d_vectY, d_vectZ);
    gpuMUdot(d_Dx, d_Dy, d_Dz, d_vectX, d_vectY, d_vectZ);
    gpuDivN2C(d_divD, d_Dx.devPtr(), d_Dy.devPtr(), d_Dz.devPtr(),
              nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

    // ---- Fork: interior compute on interiorStream_ overlaps NCCL comm on
    //      solverStream_. Both forks are recorded into the SAME graph because
    //      both streams are actively capturing (joined to solverStream_'s
    //      capture via the event below — this is standard multi-stream graph
    //      capture, not a separate graph). ----
    cudaEventRecord(ncclForkEvent_, solverStream_);
    cudaStreamWaitEvent(interiorStream_, ncclForkEvent_, 0);

    // ---- Branch A (solverStream_): NCCL halo exchange, captured ----
    gpuBatchedHaloExchangeNCCL(d_maxwellNcclPtrs_, 10, nxc, nyc, nzc,
                               /*isCenterFlag=*/true, /*isFaceOnlyFlag=*/false,
                               solverStream_);

    // ---- Branch B (interiorStream_): interior divC2N + gradC2N, overlaps
    //      the NCCL send/recv above ----
    gpuDivC2N_interior(d_imageX.devPtr(), d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, interiorStream_);
    gpuDivC2N_interior(d_imageY.devPtr(), d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, interiorStream_);
    gpuDivC2N_interior(d_imageZ.devPtr(), d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, interiorStream_);
    gpuGradC2N_interior(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                        d_divD, nxn, nyn, nzn, _invdx, _invdy, _invdz, interiorStream_);

    // ---- Join: both branches must finish before boundary compute ----
    cudaEventRecord(ncclJoinEvent_, interiorStream_);
    cudaStreamWaitEvent(solverStream_, ncclJoinEvent_, 0);

    // ---- g_bc_post: BC faces + boundary divC2N/gradC2N + arithmetic + BCs ----
    gpuBCface(nxc, nyc, nzc, d_tempXC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempYC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempZC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_poissonTemp, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_poissonIm, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divBwork, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divE_work, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_tempC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_PHI, 2, 2, 2, 2, 2, 2, &_vct, solverStream_);

    gpuDivC2N_boundary(d_imageX.devPtr(), d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_boundary(d_imageY.devPtr(), d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuDivC2N_boundary(d_imageZ.devPtr(), d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                       nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
    gpuGradC2N_boundary(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                        d_divD, nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

    gpuNeg3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(), nodeSize, solverStream_);
    gpuSub3(d_imageX.devPtr(), d_tempX.devPtr(), d_imageY.devPtr(), d_tempY.devPtr(),
            d_imageZ.devPtr(), d_tempZ.devPtr(), nodeSize, solverStream_);
    gpuScale3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(), delt * delt, nodeSize, solverStream_);
    gpuSumAddTwo3(d_imageX.devPtr(), d_Dx.devPtr(), d_vectX.devPtr(),
                  d_imageY.devPtr(), d_Dy.devPtr(), d_vectY.devPtr(),
                  d_imageZ.devPtr(), d_Dz.devPtr(), d_vectZ.devPtr(),
                  nodeSize, solverStream_);

    if (hasLeftX)
        gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
    if (hasRightX)
        gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
    if (hasLeftY)
        gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
    if (hasRightY)
        gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
    if (hasLeftZ)
        gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);
    if (hasRightZ)
        gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);

    if (get_col().getApplyInflowBcsEImage())
        gpuOpenBoundaryInflowEImage(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                                    d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                                    nxn, nyn, nzn);

    gpuPhys2Solver3(d_im, d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                    nxn, nyn, nzn, solverStream_);

    cudaGraph_t g;
    cudaStreamEndCapture(solverStream_, &g);
    cudaGraphInstantiate(&maxwellImageNcclExec_, g, NULL, NULL, 0);
    cudaGraphDestroy(g);
}
#endif // CUDA_GRAPH && USE_NCCL
void EMfields3D::gpuMaxwellImage_cuda_graph_refactored(cudaSolverType *d_im, cudaSolverType *d_vector)
{
#ifdef CUDA_GRAPH
    g_maxwellImageTimer.begin(solverStream_);
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();
    size_t nodeSize = (size_t)nxn * nyn * nzn;

    cudaSolverType *d_divD = d_PHI.devPtr(); // div(D) scratch (dead during Maxwell solve)

    // ---- eager: Krylov -> physical (d_vector varies, cannot capture) ----
    gpuSolver2Phys3(d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                    d_vector, nxn, nyn, nzn, solverStream_);

    // =====================================================================
    //  g_pre (s1Exec_): 9 memsets + 3 gradN2C + MUdot + divN2C(->d_PHI)
    // =====================================================================
    if (s1Exec_ == nullptr)
    {
        cudaSolverType *zptrs[9] = {d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                                    d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                                    d_Dx.devPtr(), d_Dy.devPtr(), d_Dz.devPtr()};
        cudaGraph_t g;
        cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal);
        gpuSetAll0_N(zptrs, 9, nodeSize, solverStream_);
        gpuLapN2N_3_gradients(d_vectX, d_vectY, d_vectZ);
        gpuMUdot(d_Dx, d_Dy, d_Dz, d_vectX, d_vectY, d_vectZ);
        gpuDivN2C(d_divD, d_Dx.devPtr(), d_Dy.devPtr(), d_Dz.devPtr(),
                  nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
        cudaStreamEndCapture(solverStream_, &g);
        cudaGraphInstantiate(&s1Exec_, g, NULL, NULL, 0);
        cudaGraphDestroy(g);
    }
    nvtxRangePush("g_pre");
    cudaGraphLaunch(s1Exec_, solverStream_);
    nvtxRangePop();

#ifdef HALO_OVERLAP
    cudaSolverType *ptrs10[10] = {
        d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
        d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
        d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
        d_divD};

    // ---- halo_begin: pack + post MPI for all 10 (eager, uncapturable) ----
    nvtxRangePush("halo_begin");
    gpuBatchedHaloBeginExchange(ptrs10, 10, nxc, nyc, nzc,
                                true, false, false, false, solverStream_);
    nvtxRangePop();

    // =====================================================================
    //  g_interior (s2Exec_): interior divC2N + gradC2N, overlaps MPI.
    //  Captured ONCE between begin/end; the kernels touch only interior
    //  nodes (no ghost dependency) so capture-once/launch-every is valid.
    // =====================================================================
    if (s2Exec_ == nullptr)
    {
        cudaGraph_t gi;
        cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal);
        gpuDivC2N_interior(d_imageX.devPtr(),
                           d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                           nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuDivC2N_interior(d_imageY.devPtr(),
                           d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                           nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuDivC2N_interior(d_imageZ.devPtr(),
                           d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                           nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuGradC2N_interior(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                            d_divD, nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        cudaStreamEndCapture(solverStream_, &gi);
        cudaGraphInstantiate(&s2Exec_, gi, NULL, NULL, 0);
        cudaGraphDestroy(gi);
    }
    nvtxRangePush("g_interior");
    cudaGraphLaunch(s2Exec_, solverStream_);
    nvtxRangePop();

    // ---- halo_end: MPI_Waitall + unpack (eager, uncapturable) ----
    nvtxRangePush("halo_end");
    gpuBatchedHaloEndExchange(ptrs10, 10, nxc, nyc, nzc,
                              true, false, false, false, solverStream_);
    nvtxRangePop();

    // =====================================================================
    //  g_bc_post (s5Exec_): BC faces + boundary divC2N/gradC2N
    //  + neg3 + sub3 + scale3 + sumAddTwo3 + conductor/openBC.
    //  One long graph — the old g_post is folded in (no sync separates them).
    // =====================================================================
    bool hasLeftX = (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0);
    bool hasRightX = (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0);
    bool hasLeftY = (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0);
    bool hasRightY = (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0);
    bool hasLeftZ = (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0);
    bool hasRightZ = (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0);

    if (s5Exec_ == nullptr)
    {
        cudaGraph_t g5;
        cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal);

        // BC faces: type 1 on the 9 gradients, type 2 on div(D)
        gpuBCface(nxc, nyc, nzc, d_tempXC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_tempYC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_tempZC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_divC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_poissonTemp, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_poissonIm, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_divBwork, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_divE_work, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_tempC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
        gpuBCface(nxc, nyc, nzc, d_PHI, 2, 2, 2, 2, 2, 2, &_vct, solverStream_);

        // boundary compute (ghost + BC now available)
        gpuDivC2N_boundary(d_imageX.devPtr(),
                           d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                           nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuDivC2N_boundary(d_imageY.devPtr(),
                           d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                           nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuDivC2N_boundary(d_imageZ.devPtr(),
                           d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                           nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuGradC2N_boundary(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                            d_divD, nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

        // arithmetic (folded-in old g_post): image = dt^2*(-lap - grad(divD)) + D + vect
        gpuNeg3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(), nodeSize, solverStream_);
        gpuSub3(d_imageX.devPtr(), d_tempX.devPtr(),
                d_imageY.devPtr(), d_tempY.devPtr(),
                d_imageZ.devPtr(), d_tempZ.devPtr(), nodeSize, solverStream_);
        gpuScale3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                  delt * delt, nodeSize, solverStream_);
        gpuSumAddTwo3(d_imageX.devPtr(), d_Dx.devPtr(), d_vectX.devPtr(),
                      d_imageY.devPtr(), d_Dy.devPtr(), d_vectY.devPtr(),
                      d_imageZ.devPtr(), d_Dz.devPtr(), d_vectZ.devPtr(),
                      nodeSize, solverStream_);

        if (hasLeftX)
            gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
        if (hasRightX)
            gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
        if (hasLeftY)
            gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
        if (hasRightY)
            gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
        if (hasLeftZ)
            gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);
        if (hasRightZ)
            gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);

        if (get_col().getApplyInflowBcsEImage())
            gpuOpenBoundaryInflowEImage(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                                        d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                                        nxn, nyn, nzn);
        gpuPhys2Solver3(d_im,
                        d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                        nxn, nyn, nzn, solverStream_);
        cudaStreamEndCapture(solverStream_, &g5);
        cudaGraphInstantiate(&s5Exec_, g5, NULL, NULL, 0);
        cudaGraphDestroy(g5);
    }
    nvtxRangePush("g_bc_post");
    cudaGraphLaunch(s5Exec_, solverStream_);
    nvtxRangePop();

#else // ---------- blocking fallback  ----------
    nvtxRangePush("halo_A_blocking");
    gpuCommunicateCenterBC_9(nxc, nyc, nzc,
                             d_tempXC, d_tempYC, d_tempZC,
                             d_divC, d_poissonTemp, d_poissonIm,
                             d_divBwork, d_divE_work, d_tempC,
                             1, 1, 1, 1, 1, 1);
    nvtxRangePop();
    nvtxRangePush("halo_B_blocking");
    gpuCommunicateCenterBC(nxc, nyc, nzc, d_PHI, 2, 2, 2, 2, 2, 2);
    nvtxRangePop();

    // =====================================================================
    //  g_post_halo (s3Exec_): 3x divC2N (full domain) -> image = lap(vect),
    //  + gradC2N(divD) -> temp.  Both blocking halo exchanges above have
    //  fully completed before this launches, so these kernels only ever
    //  read finished ghost data -- capture-once/launch-every-call is safe,
    //  exactly like s1Exec_/s5Exec_ above.
    // =====================================================================
    if (s2Exec_ == nullptr)
    {
        cudaGraph_t g2;
        cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
        gpuDivC2N(d_imageX.devPtr(),
                  d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                  nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuDivC2N(d_imageY.devPtr(),
                  d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                  nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuDivC2N(d_imageZ.devPtr(),
                  d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
                  nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        gpuGradC2N(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                   d_divD, nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
        cudaErrChk(cudaStreamEndCapture(solverStream_, &g2));
        cudaErrChk(cudaGraphInstantiate(&s2Exec_, g2, NULL, NULL, 0));
        cudaErrChk(cudaGraphDestroy(g2));
    }
    nvtxRangePush("g_post_halo_blocking");
    cudaErrChk(cudaGraphLaunch(s2Exec_, solverStream_));
    nvtxRangePop();
    bool hasLeftX = (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0);
    bool hasRightX = (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0);
    bool hasLeftY = (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0);
    bool hasRightY = (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0);
    bool hasLeftZ = (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0);
    bool hasRightZ = (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0);

    if (s5Exec_ == nullptr)
    {
        cudaGraph_t g5;
        cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));

        gpuNeg3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(), nodeSize, solverStream_);
        gpuSub3(d_imageX.devPtr(), d_tempX.devPtr(),
                d_imageY.devPtr(), d_tempY.devPtr(),
                d_imageZ.devPtr(), d_tempZ.devPtr(), nodeSize, solverStream_);
        gpuScale3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                  delt * delt, nodeSize, solverStream_);
        gpuSumAddTwo3(d_imageX.devPtr(), d_Dx.devPtr(), d_vectX.devPtr(),
                      d_imageY.devPtr(), d_Dy.devPtr(), d_vectY.devPtr(),
                      d_imageZ.devPtr(), d_Dz.devPtr(), d_vectZ.devPtr(),
                      nodeSize, solverStream_);

        if (hasLeftX)
            gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
        if (hasRightX)
            gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
        if (hasLeftY)
            gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
        if (hasRightY)
            gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
        if (hasLeftZ)
            gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);
        if (hasRightZ)
            gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);

        if (get_col().getApplyInflowBcsEImage())
            gpuOpenBoundaryInflowEImage(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                                        d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                                        nxn, nyn, nzn);

        gpuPhys2Solver3(d_im,
                        d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                        nxn, nyn, nzn, solverStream_);

        cudaErrChk(cudaStreamEndCapture(solverStream_, &g5));
        cudaErrChk(cudaGraphInstantiate(&s5Exec_, g5, NULL, NULL, 0));
        cudaErrChk(cudaGraphDestroy(g5));
    }

    nvtxRangePush("g_bc_post_blocking");
    cudaErrChk(cudaGraphLaunch(s5Exec_, solverStream_));
    nvtxRangePop();
#endif
    //  ---- eager: physical -> Krylov (d_im varies, cannot capture) ----
    /* gpuPhys2Solver3(d_im,
                   d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                 nxn, nyn, nzn, solverStream_);*/
    g_maxwellImageTimer.end(solverStream_);
#endif
}

// =========================================================================
//  GPU MaxwellSource:  build RHS of Maxwell system
// =========================================================================

// =========================================================================
//  GPU FGMRES(m) with Block-Jacobi preconditioner
// =========================================================================

void EMfields3D::gpuCalculateB_cuda_graph(int cycle)
{
#ifdef CUDA_GRAPH
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

    if (vct->getCartesian_rank() == 0)
        cout << "*** B CALCULATION [GPU] ***" << endl;

    g_calculateBTimer.begin(solverStream_);

    size_t centSize = (size_t)nxc * nyc * nzc;

    // =====================================================================
    //  bFieldUpdateExec_: curl(Eth) -> tempXC/YC/ZC, then
    //  B^{n+1} = B^n - c*dt*curl(Eth). Captured once: grid dims, c and dt
    //  are fixed for the lifetime of the run, and this pair of kernels
    //  always executes (no cycle-dependent branch), so capture-once /
    //  launch-every-cycle is safe.
    // =====================================================================
    if (bFieldUpdateExec_ == nullptr)
    {
        cudaGraph_t g;
        cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal);
        gpuCurlN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                   d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
                   nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
        gpuAddscale3(-c * dt,
                     d_Bxc.devPtr(), d_tempXC.devPtr(),
                     d_Byc.devPtr(), d_tempYC.devPtr(),
                     d_Bzc.devPtr(), d_tempZC.devPtr(), centSize, solverStream_);
        cudaStreamEndCapture(solverStream_, &g);
        cudaGraphInstantiate(&bFieldUpdateExec_, g, NULL, NULL, 0);
        cudaGraphDestroy(g);
    }
    nvtxRangePush("gB_field_update");
    cudaGraphLaunch(bFieldUpdateExec_, solverStream_);
    nvtxRangePop();

    // Communicate center B ghost cells (batched: 3 fields in 1 MPI round)
#ifdef HALO_OVERLAP
    {
        cudaSolverType *bptrs[3] = {d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr()};

        // ---- halo_begin: pack + post MPI (eager, uncapturable) ----
        nvtxRangePush("gB_halo_begin");
        gpuBatchedHaloBeginExchange(bptrs, 3, nxc, nyc, nzc,
                                    true, false, false, false, solverStream_);
        nvtxRangePop();

        // =====================================================================
        //  bInteriorExec_: interior interpC2N, overlaps MPI. Captured once;
        //  it touches only interior nodes (no ghost dependency), so
        //  capture-once/launch-every-cycle is valid.
        // =====================================================================
        if (bInteriorExec_ == nullptr)
        {
            cudaGraph_t gi;
            cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal);
            gpuInterpC2N_interior(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_interior(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_interior(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);
            cudaStreamEndCapture(solverStream_, &gi);
            cudaGraphInstantiate(&bInteriorExec_, gi, NULL, NULL, 0);
            cudaGraphDestroy(gi);
        }
        nvtxRangePush("gB_interior");
        cudaGraphLaunch(bInteriorExec_, solverStream_);
        nvtxRangePop();

        // ---- halo_end: MPI_Waitall + unpack (eager, uncapturable) ----
        nvtxRangePush("gB_halo_end");
        gpuBatchedHaloEndExchange(bptrs, 3, nxc, nyc, nzc,
                                  true, false, false, false, solverStream_);
        nvtxRangePop();

        // =====================================================================
        //  bBcPostExec_: BC faces + open-boundary inflow + case-specific
        //  fixups on center B, then boundary interpC2N (needs ghost + BC +
        //  fixup data, all now available). The host-side branches below
        //  (rank-local topology, bcEMface*, simCase) are constant for the
        //  lifetime of the run, so capture-once/launch-every-cycle is valid.
        // =====================================================================
        if (bBcPostExec_ == nullptr)
        {
            cudaGraph_t g5;
            cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal);

            gpuBCface(nxc, nyc, nzc, d_Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], &_vct, solverStream_);
            gpuBCface(nxc, nyc, nzc, d_Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], &_vct, solverStream_);
            gpuBCface(nxc, nyc, nzc, d_Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], &_vct, solverStream_);

            gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc, nyc, nzc);

            {
                const string &simCase = col->getCase();
                if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
                    gpuFixBcGEM();
                if (simCase == "ForceFree")
                    gpuFixBforcefree();
            }

            gpuInterpC2N_boundary(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_boundary(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_boundary(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);

            cudaStreamEndCapture(solverStream_, &g5);
            cudaGraphInstantiate(&bBcPostExec_, g5, NULL, NULL, 0);
            cudaGraphDestroy(g5);
        }
        nvtxRangePush("gB_bc_post");
        cudaGraphLaunch(bBcPostExec_, solverStream_);
        nvtxRangePop();
    }
#else
    gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc,
                                  d_Bxc, col->bcBx, d_Byc, col->bcBy, d_Bzc, col->bcBz);

    // Open boundary conditions on center-based B
    gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc, nyc, nzc);

    // Case-specific fixes on center-based B
    {
        const string &simCase = col->getCase();
        if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
            gpuFixBcGEM();
        if (simCase == "ForceFree")
            gpuFixBforcefree();
    }

    // Interpolate center → node
    gpuInterpC2N(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);
#endif

    // Communicate node B ghost cells (batched: 3 fields in 1 MPI round)
    gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn,
                                d_Bxn, col->bcBx, d_Byn, col->bcBy, d_Bzn, col->bcBz);

    // Case-specific fixes on node-based B
    {
        const string &simCase = col->getCase();
        if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
            gpuFixBnGEM();
    }

    // Divergence cleaning: lap(PSI) = div(B), B = B - grad(PSI)
    if (divBCorrection && cycle % divBCorrectionCycle == 0)
        gpuApplyDivBCleaning();

    g_calculateBTimer.end(solverStream_);
#endif
}

#if defined(CUDA_GRAPH) && defined(USE_NCCL)
// =========================================================================
//  GPU calculateB (NCCL): same overall structure as
//  gpuCalculateB_cuda_graph(), but the center-B halo exchange uses NCCL
//  point-to-point (gpuBatchedHaloExchangeNCCL) instead of MPI. Since NCCL
//  calls are stream-ordered and graph-capturable, the whole per-cycle step
//  (field update + halo + interior/boundary interp + BC/fixups) is captured
//  as a SINGLE graph, replayed every cycle -- no eager/uncapturable
//  begin/end split is needed the way HALO_OVERLAP's MPI path requires.
// =========================================================================
void EMfields3D::gpuBuildBNcclGraph()
{
    const Collective *col = &get_col();
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

    size_t centSize = (size_t)nxc * nyc * nzc;

    // Immutable device pointer array for the captured halo exchange
    // (must exist before capture; lives as long as the graph).
    if (d_bNcclPtrs_ == nullptr)
    {
        cudaSolverType *h3[3] = {d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr()};
        d_bNcclPtrs_ = gpuMakeNcclPtrArray(h3, 3);
    }

    cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));

    // ---- field update: curl(Eth) -> tempXC/YC/ZC, then B -= c*dt*curl(Eth) ----
    gpuCurlN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
               d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
               nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
    gpuAddscale3(-c * dt,
                 d_Bxc.devPtr(), d_tempXC.devPtr(),
                 d_Byc.devPtr(), d_tempYC.devPtr(),
                 d_Bzc.devPtr(), d_tempZC.devPtr(), centSize, solverStream_);

    // ---- Fork: interior interpC2N on interiorStream_ overlaps the NCCL
    //      send/recv on solverStream_. Both forks land in the same graph
    //      because both streams are actively capturing (joined to
    //      solverStream_'s capture via ncclForkEvent_/ncclJoinEvent_). ----
    cudaErrChk(cudaEventRecord(ncclForkEvent_, solverStream_));
    cudaErrChk(cudaStreamWaitEvent(interiorStream_, ncclForkEvent_, 0));

    // ---- Branch A (solverStream_): NCCL halo exchange, captured ----
    gpuBatchedHaloExchangeNCCL(d_bNcclPtrs_, 3, nxc, nyc, nzc,
                               /*isCenterFlag=*/true, /*isFaceOnlyFlag=*/false, solverStream_);

    // ---- Branch B (interiorStream_): interior interpC2N, overlaps the NCCL
    //      send/recv above ----
    gpuInterpC2N_interior(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, interiorStream_);
    gpuInterpC2N_interior(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, interiorStream_);
    gpuInterpC2N_interior(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, interiorStream_);

    // ---- Join: both branches must finish before boundary compute ----
    cudaErrChk(cudaEventRecord(ncclJoinEvent_, interiorStream_));
    cudaErrChk(cudaStreamWaitEvent(solverStream_, ncclJoinEvent_, 0));

    // ---- BC faces + open-boundary inflow + case-specific fixups on center B,
    //      then boundary interpC2N (needs ghost + BC + fixup data) ----
    gpuBCface(nxc, nyc, nzc, d_Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], &_vct, solverStream_);

    gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc, nyc, nzc);

    {
        const string &simCase = col->getCase();
        if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
            gpuFixBcGEM();
        if (simCase == "ForceFree")
            gpuFixBforcefree();
    }

    gpuInterpC2N_boundary(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_boundary(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_boundary(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);

    cudaGraph_t g;
    cudaErrChk(cudaStreamEndCapture(solverStream_, &g));
    cudaErrChk(cudaGraphInstantiate(&bNcclExec_, g, NULL, NULL, 0));
    cudaErrChk(cudaGraphDestroy(g));
}

void EMfields3D::gpuCalculateB_nccl(int cycle)
{
    const Collective *col = &get_col();
    const VirtualTopology3D *vct = &get_vct();

    if (vct->getCartesian_rank() == 0)
        cout << "*** B CALCULATION [GPU] ***" << endl;

    if (bNcclExec_ == nullptr)
        gpuBuildBNcclGraph();

    g_calculateBTimer.begin(solverStream_);

    nvtxRangePush("gB_nccl");
    cudaErrChk(cudaGraphLaunch(bNcclExec_, solverStream_));
    nvtxRangePop();

    // Communicate node B ghost cells (batched: 3 fields in 1 MPI round) --
    // eager, unrelated to the NCCL center-B halo captured above.
    gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn,
                                d_Bxn, col->bcBx, d_Byn, col->bcBy, d_Bzn, col->bcBz);

    // Case-specific fixes on node-based B
    {
        const string &simCase = col->getCase();
        if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
            gpuFixBnGEM();
    }

    // Divergence cleaning: lap(PSI) = div(B), B = B - grad(PSI)
    if (divBCorrection && cycle % divBCorrectionCycle == 0)
        gpuApplyDivBCleaning();

    g_calculateBTimer.end(solverStream_);
}
#endif // CUDA_GRAPH && USE_NCCL

// =========================================================================
//  GPU calculateHatFunctions: compute Jhat and rhohat
// =========================================================================
void EMfields3D::gpuCalculateHatFunctions_cuda_graph()
{
#ifdef CUDA_GRAPH
    g_hatFunctionsTimer.begin(solverStream_);
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

    size_t nodeSize = (size_t)nxn * nyn * nzn;
    size_t centSize = (size_t)nxc * nyc * nzc;

    // Lazy (re)size per-species graph slots -- addresses returned by
    // speciesPtr(is) are fixed for the lifetime of the run, so once a
    // slot is captured it is valid to replay on every subsequent call.
    if ((int)hatPreExec_.size() != ns)
    {
        hatPreExec_.assign(ns, nullptr);
        hatPostExec_.assign(ns, nullptr);
    }

    // Smooth rhoc
    gpuSmooth(d_rhoc, 0);

    // Initialise Jxh/Jyh/Jzh = 0
    cudaSolverType *jptrs[3] = {d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr()};
    gpuSetAll0_N(jptrs, 3, nodeSize, solverStream_);

    for (int is = 0; is < ns; is++)
    {
        // =====================================================================
        //  g_hat_pre (hatPreExec_[is]): divSymmTensorN2C + scale3.
        //  Reads species-specific tensor pointers (fixed address per `is`
        //  for the whole run) and writes the shared d_tempXC/YC/ZC scratch.
        //  One graph per species, captured once, replayed every call.
        // =====================================================================
        if (hatPreExec_[is] == nullptr)
        {
            cudaGraph_t gpre;
            cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
            gpuDivSymmTensorN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                                d_pXXsn.speciesPtr(is), d_pXYsn.speciesPtr(is), d_pXZsn.speciesPtr(is),
                                d_pYYsn.speciesPtr(is), d_pYZsn.speciesPtr(is), d_pZZsn.speciesPtr(is),
                                nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
            gpuScale3(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                      -dt / 2.0, centSize, solverStream_);
            cudaErrChk(cudaStreamEndCapture(solverStream_, &gpre));
            cudaErrChk(cudaGraphInstantiate(&hatPreExec_[is], gpre, NULL, NULL, 0));
            cudaErrChk(cudaGraphDestroy(gpre));
        }
        nvtxRangePush("g_hat_pre");
        cudaErrChk(cudaGraphLaunch(hatPreExec_[is], solverStream_));
        nvtxRangePop();

#ifdef HALO_OVERLAP
        // ---- halo_begin: pack + post MPI for 3 centre fields (eager, uncapturable) ----
        cudaSolverType *hatPtrs[3] = {d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr()};
        nvtxRangePush("hat_halo_begin");
        gpuBatchedHaloBeginExchange(hatPtrs, 3, nxc, nyc, nzc,
                                    /*isCenter=*/true, /*faceOnly=*/false,
                                    /*needInterp=*/false, /*isParticle=*/true, solverStream_);
        nvtxRangePop();

        if (hatInteriorExec_ == nullptr)
        {
            cudaGraph_t gi;
            cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
            gpuInterpC2N_interior(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_interior(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_interior(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, solverStream_);
            cudaErrChk(cudaStreamEndCapture(solverStream_, &gi));
            cudaErrChk(cudaGraphInstantiate(&hatInteriorExec_, gi, NULL, NULL, 0));
            cudaErrChk(cudaGraphDestroy(gi));
        }
        nvtxRangePush("g_hat_interior");
        cudaErrChk(cudaGraphLaunch(hatInteriorExec_, solverStream_));
        nvtxRangePop();

        // ---- halo_end: MPI_Waitall + unpack (eager, uncapturable) ----
        nvtxRangePush("hat_halo_end");
        gpuBatchedHaloEndExchange(hatPtrs, 3, nxc, nyc, nzc,
                                  /*isCenter=*/true, /*faceOnly=*/false,
                                  /*needInterp=*/false, /*isParticle=*/true, solverStream_);
        nvtxRangePop();

        if (hatBoundaryExec_ == nullptr)
        {
            cudaGraph_t gb;
            cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
            gpuBCface_P(nxc, nyc, nzc, d_tempXC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
            gpuBCface_P(nxc, nyc, nzc, d_tempYC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
            gpuBCface_P(nxc, nyc, nzc, d_tempZC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
            gpuInterpC2N_boundary(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_boundary(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N_boundary(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, solverStream_);
            cudaErrChk(cudaStreamEndCapture(solverStream_, &gb));
            cudaErrChk(cudaGraphInstantiate(&hatBoundaryExec_, gb, NULL, NULL, 0));
            cudaErrChk(cudaGraphDestroy(gb));
        }
        nvtxRangePush("g_hat_boundary");
        cudaErrChk(cudaGraphLaunch(hatBoundaryExec_, solverStream_));
        nvtxRangePop();

#else // ---------- blocking fallback ----------
        nvtxRangePush("hat_halo_blocking");
        gpuCommunicateCenterBC_P_3(nxc, nyc, nzc, d_tempXC, d_tempYC, d_tempZC, 2, 2, 2, 2, 2, 2);
        nvtxRangePop();

        if (hatBlockingExec_ == nullptr)
        {
            cudaGraph_t gblk;
            cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
            gpuInterpC2N(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, solverStream_);
            gpuInterpC2N(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, solverStream_);
            cudaErrChk(cudaStreamEndCapture(solverStream_, &gblk));
            cudaErrChk(cudaGraphInstantiate(&hatBlockingExec_, gblk, NULL, NULL, 0));
            cudaErrChk(cudaGraphDestroy(gblk));
        }
        nvtxRangePush("g_hat_interp_blocking");
        cudaErrChk(cudaGraphLaunch(hatBlockingExec_, solverStream_));
        nvtxRangePop();
#endif

        if (hatPostExec_[is] == nullptr)
        {
            cudaGraph_t gpost;
            cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
            gpuSum3(d_tempXN.devPtr(), d_Jxs.speciesPtr(is),
                    d_tempYN.devPtr(), d_Jys.speciesPtr(is),
                    d_tempZN.devPtr(), d_Jzs.speciesPtr(is), nodeSize, solverStream_);
            gpuPIdot(d_Jxh, d_Jyh, d_Jzh, d_tempXN, d_tempYN, d_tempZN, is);
            cudaErrChk(cudaStreamEndCapture(solverStream_, &gpost));
            cudaErrChk(cudaGraphInstantiate(&hatPostExec_[is], gpost, NULL, NULL, 0));
            cudaErrChk(cudaGraphDestroy(gpost));
        }
        nvtxRangePush("g_hat_post");
        cudaErrChk(cudaGraphLaunch(hatPostExec_[is], solverStream_));
        nvtxRangePop();
    }

    gpuSmooth3(d_Jxh, d_Jyh, d_Jzh, 1);

    if (hatRhohatExec_ == nullptr)
    {
        cudaGraph_t grh;
        cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
        gpuDivN2C(d_tempXC.devPtr(),
                  d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr(),
                  nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
        gpuScale(d_tempXC.devPtr(), -dt * th, centSize, solverStream_);
        gpuSum(d_tempXC.devPtr(), d_rhoc.devPtr(), centSize, solverStream_);
        gpuEq(d_rhoh.devPtr(), d_tempXC.devPtr(), centSize, solverStream_);
        cudaErrChk(cudaStreamEndCapture(solverStream_, &grh));
        cudaErrChk(cudaGraphInstantiate(&hatRhohatExec_, grh, NULL, NULL, 0));
        cudaErrChk(cudaGraphDestroy(grh));
    }
    nvtxRangePush("g_hat_rhohat");
    cudaErrChk(cudaGraphLaunch(hatRhohatExec_, solverStream_));
    nvtxRangePop();
    // Communicate rhoh
    gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);
    g_hatFunctionsTimer.end(solverStream_);
#endif
}

#if defined(CUDA_GRAPH) && defined(USE_NCCL)
// =========================================================================
//  GPU calculateHatFunctions (NCCL): same overall structure as
//  gpuCalculateHatFunctions_cuda_graph(), but the per-species tensor halo
//  exchange uses NCCL point-to-point (gpuBatchedHaloExchangeNCCL) instead of
//  MPI. Since NCCL calls are stream-ordered and graph-capturable, the whole
//  per-species step (pre + halo + interior/boundary interp + post) is
//  captured as a SINGLE graph per species, replayed every call — no
//  eager/uncapturable begin/end split is needed the way HALO_OVERLAP's MPI
//  path requires.
// =========================================================================
void EMfields3D::gpuBuildHatNcclGraph(int is)
{
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

    size_t nodeSize = (size_t)nxn * nyn * nzn;
    size_t centSize = (size_t)nxc * nyc * nzc;

    // Immutable device pointer array for the captured halo exchange (same
    // tempXC/YC/ZC arrays for every species, so one array serves all graphs).
    if (d_hatNcclPtrs_ == nullptr)
    {
        cudaSolverType *h3[3] = {d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr()};
        d_hatNcclPtrs_ = gpuMakeNcclPtrArray(h3, 3);
    }

    cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));

    // ---- pre: divSymmTensorN2C + scale3 ----
    gpuDivSymmTensorN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                        d_pXXsn.speciesPtr(is), d_pXYsn.speciesPtr(is), d_pXZsn.speciesPtr(is),
                        d_pYYsn.speciesPtr(is), d_pYZsn.speciesPtr(is), d_pZZsn.speciesPtr(is),
                        nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
    gpuScale3(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
              -dt / 2.0, centSize, solverStream_);

    // ---- Fork: interior interpC2N on interiorStream_ overlaps the NCCL
    //      send/recv on solverStream_. Both forks land in the same graph
    //      because both streams are actively capturing (joined to
    //      solverStream_'s capture via ncclForkEvent_/ncclJoinEvent_). ----
    cudaErrChk(cudaEventRecord(ncclForkEvent_, solverStream_));
    cudaErrChk(cudaStreamWaitEvent(interiorStream_, ncclForkEvent_, 0));

    // ---- Branch A (solverStream_): NCCL halo exchange, captured ----
    gpuBatchedHaloExchangeNCCL(d_hatNcclPtrs_, 3, nxc, nyc, nzc,
                               /*isCenterFlag=*/true, /*isFaceOnlyFlag=*/false, solverStream_);

    // ---- Branch B (interiorStream_): interior interpC2N, overlaps the NCCL
    //      send/recv above ----
    gpuInterpC2N_interior(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, interiorStream_);
    gpuInterpC2N_interior(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, interiorStream_);
    gpuInterpC2N_interior(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, interiorStream_);

    // ---- Join: both branches must finish before boundary compute ----
    cudaErrChk(cudaEventRecord(ncclJoinEvent_, interiorStream_));
    cudaErrChk(cudaStreamWaitEvent(solverStream_, ncclJoinEvent_, 0));

    // ---- boundary: BC faces + boundary interpC2N ----
    gpuBCface_P(nxc, nyc, nzc, d_tempXC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
    gpuBCface_P(nxc, nyc, nzc, d_tempYC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
    gpuBCface_P(nxc, nyc, nzc, d_tempZC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
    gpuInterpC2N_boundary(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_boundary(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_boundary(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, solverStream_);

    // ---- post: sum3 + PIdot ----
    gpuSum3(d_tempXN.devPtr(), d_Jxs.speciesPtr(is),
            d_tempYN.devPtr(), d_Jys.speciesPtr(is),
            d_tempZN.devPtr(), d_Jzs.speciesPtr(is), nodeSize, solverStream_);
    gpuPIdot(d_Jxh, d_Jyh, d_Jzh, d_tempXN, d_tempYN, d_tempZN, is);

    cudaGraph_t g;
    cudaErrChk(cudaStreamEndCapture(solverStream_, &g));
    cudaErrChk(cudaGraphInstantiate(&hatNcclExec_[is], g, NULL, NULL, 0));
    cudaErrChk(cudaGraphDestroy(g));
}

void EMfields3D::gpuCalculateHatFunctions_nccl()
{
    g_hatFunctionsTimer.begin(solverStream_);
    const Grid *grid = &get_grid();
    double _invdx = grid->get_invdx();
    double _invdy = grid->get_invdy();
    double _invdz = grid->get_invdz();

    size_t nodeSize = (size_t)nxn * nyn * nzn;
    size_t centSize = (size_t)nxc * nyc * nzc;

    // Lazy (re)size per-species graph slots -- addresses returned by
    // speciesPtr(is) are fixed for the lifetime of the run, so once a
    // slot is captured it is valid to replay on every subsequent call.
    if ((int)hatNcclExec_.size() != ns)
        hatNcclExec_.assign(ns, nullptr);

    // Smooth rhoc (eager, MPI-based -- unrelated to the NCCL tensor halo below)
    gpuSmooth(d_rhoc, 0);

    // Initialise Jxh/Jyh/Jzh = 0
    cudaSolverType *jptrs[3] = {d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr()};
    gpuSetAll0_N(jptrs, 3, nodeSize, solverStream_);

    for (int is = 0; is < ns; is++)
    {
        if (hatNcclExec_[is] == nullptr)
            gpuBuildHatNcclGraph(is);

        nvtxRangePush("g_hat_nccl");
        cudaErrChk(cudaGraphLaunch(hatNcclExec_[is], solverStream_));
        nvtxRangePop();
    }

    gpuSmooth3(d_Jxh, d_Jyh, d_Jzh, 1);

    if (hatRhohatExec_ == nullptr)
    {
        cudaGraph_t grh;
        cudaErrChk(cudaStreamBeginCapture(solverStream_, cudaStreamCaptureModeThreadLocal));
        gpuDivN2C(d_tempXC.devPtr(),
                  d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr(),
                  nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
        gpuScale(d_tempXC.devPtr(), -dt * th, centSize, solverStream_);
        gpuSum(d_tempXC.devPtr(), d_rhoc.devPtr(), centSize, solverStream_);
        gpuEq(d_rhoh.devPtr(), d_tempXC.devPtr(), centSize, solverStream_);
        cudaErrChk(cudaStreamEndCapture(solverStream_, &grh));
        cudaErrChk(cudaGraphInstantiate(&hatRhohatExec_, grh, NULL, NULL, 0));
        cudaErrChk(cudaGraphDestroy(grh));
    }
    nvtxRangePush("g_hat_rhohat");
    cudaErrChk(cudaGraphLaunch(hatRhohatExec_, solverStream_));
    nvtxRangePop();

    // Communicate rhoh
    gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);
    g_hatFunctionsTimer.end(solverStream_);
}
#endif // CUDA_GRAPH && USE_NCCL

#endif // GPU_SOLVER
