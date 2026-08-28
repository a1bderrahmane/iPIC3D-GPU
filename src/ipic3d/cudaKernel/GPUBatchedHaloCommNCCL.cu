/**
 * @file GPUBatchedHaloCommNCCL.cu
 * @brief Fully stream-ordered, CUDA-Graph-capturable batched GPU halo
 *        exchange using NCCL point-to-point send/recv instead of MPI.
 *
 * Structural twin of GPUBatchedHaloComm.cu's gpuBatchedHaloExchange(), with
 * every MPI_Isend/Irecv/Waitall triplet replaced by
 * ncclGroupStart()/ncclRecv()/ncclSend()/ncclGroupEnd(), and every
 * cudaStreamSynchronize() removed (NCCL ops are stream-ordered, so CUDA's
 * own stream ordering guarantees pack-before-send and send-before-unpack
 * without a host sync). 

 * Reuses the persistent send/recv buffers (d_haloBuf_send_[6],
 * d_haloBuf_recv_[6]) allocated by gpuAllocateHaloBuffers() in
 * EMfields3DGPU.cpp. It deliberately does NOT use the shared
 * h_ptrArray_/d_ptrArray_ staging pair: a captured H2D copy from that
 * buffer would re-read it at every graph replay, and every eager MPI
 * batched exchange rewrites it between replays. Callers instead pass a
 * per-graph immutable DEVICE pointer array (see gpuMakeNcclPtrArray),
 * prepared before capture.
 *
 *
 * Requires: -rdc=true (relocatable device code) at compile time, since the
 * pack/unpack kernels (k_batchPack2D, k_batchUnpack2D, k_batchPackCorners4,
 * k_batchUnpackCorners4) are defined in GPUBatchedHaloComm.cu and launched
 * from this separate translation unit.
 *
 * Compile with -DGPU_SOLVER -DCUDA_GRAPH -DUSE_NCCL to activate.
 */

#ifdef GPU_SOLVER
#if defined(CUDA_GRAPH) && defined(USE_NCCL)


#include "GPUHaloComm.cuh"      // single-array self-copy kernels, BC kernels,
                                // and the batched self-copy kernels
                                // (gpuBatchSelfCopyFaceX/Y/Z,
                                //  gpuBatchSelfCopyEdgeX/Y/Z,
                                //  gpuBatchSelfCopyCornerX/Y/Z)
#include "GPUNcclTypeDef.cuh"
#include "GPUSolverMPITypes.h"
#include "EMfields3D.h"
#include "VCtopology3D.h"
#include <cassert>
#include <cstring>

// =========================================================================
//  Forward declarations of the batch pack/unpack kernels defined in
//  GPUBatchedHaloComm.cu. These have external linkage (not `static`) in
//  that file, so they can be launched from here provided the project is
//  built with relocatable device code (-rdc=true), which a multi-.cu-file
//  CUDA project like this one already requires for its existing kernel
//  cross-references.
// =========================================================================
__global__ void k_batchPack2D(
    cudaSolverType* __restrict__ buf,
    cudaSolverType* const* __restrict__ fields,
    int nFields, int base, int outerStride, int innerStride,
    int outerCount, int innerCount);

__global__ void k_batchUnpack2D(
    const cudaSolverType* __restrict__ buf,
    cudaSolverType* const* __restrict__ fields,
    int nFields, int base, int outerStride, int innerStride,
    int outerCount, int innerCount);

__global__ void k_batchPackCorners4(
    cudaSolverType* __restrict__ buf,
    cudaSolverType* const* __restrict__ fields,
    int nFields, int base, int ny, int nz);

__global__ void k_batchUnpackCorners4(
    const cudaSolverType* __restrict__ buf,
    cudaSolverType* const* __restrict__ fields,
    int nFields, int base, int ny, int nz);

static constexpr int BATCH_BLK = 256;

// ---- Local pack/unpack launch helpers (identical to the MPI path's) ----
static inline void launchPack2D_(
    cudaSolverType* buf, cudaSolverType* const* fields, int nFields,
    int base, int outerStride, int innerStride,
    int outerCount, int innerCount, cudaStream_t s)
{
    int total = outerCount * innerCount * nFields;
    if (total <= 0) return;
    k_batchPack2D<<<(total + BATCH_BLK - 1) / BATCH_BLK, BATCH_BLK, 0, s>>>(
        buf, fields, nFields, base, outerStride, innerStride, outerCount, innerCount);
}

static inline void launchUnpack2D_(
    const cudaSolverType* buf, cudaSolverType* const* fields, int nFields,
    int base, int outerStride, int innerStride,
    int outerCount, int innerCount, cudaStream_t s)
{
    int total = outerCount * innerCount * nFields;
    if (total <= 0) return;
    k_batchUnpack2D<<<(total + BATCH_BLK - 1) / BATCH_BLK, BATCH_BLK, 0, s>>>(
        buf, fields, nFields, base, outerStride, innerStride, outerCount, innerCount);
}

// =========================================================================
//  gpuBatchedHaloExchangeNCCL
//
//  Graph-capturable replacement for gpuBatchedHaloExchange(). Same 3-phase
//  structure (face, edge, corner), same persistent buffers, but:
//    - no cudaStreamSynchronize between pack and communication
//    - ncclSend/ncclRecv (wrapped in ncclGroupStart/End) instead of
//      MPI_Isend/Irecv/Waitall
// =========================================================================
// Allocate + fill a persistent device array of field pointers. Synchronous;
// must be called BEFORE stream capture begins. The returned array must stay
// alive for as long as any graph captured around a gpuBatchedHaloExchangeNCCL
// call that uses it exists (the pack/unpack kernels dereference it at every
// replay).
cudaSolverType** EMfields3D::gpuMakeNcclPtrArray(cudaSolverType* const* h_fieldPtrs, int nFields)
{
    assert(nFields > 0 && nFields <= HALO_MAX_BATCH);
    cudaSolverType** d_arr = nullptr;
    cudaErrChk(cudaMalloc(&d_arr, nFields * sizeof(cudaSolverType*)));
    cudaErrChk(cudaMemcpy(d_arr, h_fieldPtrs, nFields * sizeof(cudaSolverType*),
                          cudaMemcpyHostToDevice));
    return d_arr;
}

void EMfields3D::gpuBatchedHaloExchangeNCCL(
    cudaSolverType* const* d_fieldPtrs,
    int nFields,
    int nx, int ny, int nz,
    bool isCenterFlag,
    bool isFaceOnlyFlag,
    cudaStream_t stream)
{
    assert(nFields > 0 && nFields <= HALO_MAX_BATCH);
    const VirtualTopology3D* vct = &get_vct();
    const int myrank = vct->getCartesian_rank();

    const int xlN = ncclPeerXL_, xrN = ncclPeerXR_;
    const int ylN = ncclPeerYL_, yrN = ncclPeerYR_;
    const int zlN = ncclPeerZL_, zrN = ncclPeerZR_;

    int cc[6];
    cc[0] = (xlN != MPI_PROC_NULL && xlN != myrank) ? 1 : 0;
    cc[1] = (xrN != MPI_PROC_NULL && xrN != myrank) ? 1 : 0;
    cc[2] = (ylN != MPI_PROC_NULL && ylN != myrank) ? 1 : 0;
    cc[3] = (yrN != MPI_PROC_NULL && yrN != myrank) ? 1 : 0;
    cc[4] = (zlN != MPI_PROC_NULL && zlN != myrank) ? 1 : 0;
    cc[5] = (zrN != MPI_PROC_NULL && zrN != myrank) ? 1 : 0;

   
    const bool xRing2 = cc[0] && cc[1] && (xlN == xrN);
    const bool yRing2 = cc[2] && cc[3] && (ylN == yrN);
    const bool zRing2 = cc[4] && cc[5] && (zlN == zrN);
    const int rx0 = xRing2 ? 1 : 0, rx1 = xRing2 ? 0 : 1;
    const int ry0 = yRing2 ? 3 : 2, ry1 = yRing2 ? 2 : 3;
    const int rz0 = zRing2 ? 5 : 4, rz1 = zRing2 ? 4 : 5;

    const int offset = isCenterFlag ? 0 : 1;
    cudaSolverType* const* d_ptrs = d_fieldPtrs;

    const int nyzF = (ny - 2) * (nz - 2);   // YZ face element count per field
    const int nxzF = (nx - 2) * (nz - 2);   // XZ face
    const int nxyF = (nx - 2) * (ny - 2);   // XY face
    const ncclDataType_t nt = ncclTypeOf<cudaSolverType>();

    // =====================================================================
    //  PHASE 1: Face exchange
    // =====================================================================

    // ---- Pack faces into contiguous send buffers ----
    if (cc[0]) { int ix = 1 + offset;    launchPack2D_(d_haloBuf_send_[0], d_ptrs, nFields, ix*ny*nz + 1*nz + 1,    nz, 1, ny-2, nz-2, stream); }
    if (cc[1]) { int ix = nx - 2 - offset; launchPack2D_(d_haloBuf_send_[1], d_ptrs, nFields, ix*ny*nz + 1*nz + 1,    nz, 1, ny-2, nz-2, stream); }
    if (cc[2]) { int iy = 1 + offset;    launchPack2D_(d_haloBuf_send_[2], d_ptrs, nFields, 1*ny*nz + iy*nz + 1, ny*nz, 1, nx-2, nz-2, stream); }
    if (cc[3]) { int iy = ny - 2 - offset; launchPack2D_(d_haloBuf_send_[3], d_ptrs, nFields, 1*ny*nz + iy*nz + 1, ny*nz, 1, nx-2, nz-2, stream); }
    if (cc[4]) { int iz = 1 + offset;    launchPack2D_(d_haloBuf_send_[4], d_ptrs, nFields, 1*ny*nz + 1*nz + iz, ny*nz, nz, nx-2, ny-2, stream); }
    if (cc[5]) { int iz = nz - 2 - offset; launchPack2D_(d_haloBuf_send_[5], d_ptrs, nFields, 1*ny*nz + 1*nz + iz, ny*nz, nz, nx-2, ny-2, stream); }

    
    NCCLCHECK(ncclGroupStart());
    if (cc[0]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[0], nyzF*nFields, nt, xlN, fieldNcclComm_, stream));
                 NCCLCHECK(ncclSend(d_haloBuf_send_[0], nyzF*nFields, nt, xlN, fieldNcclComm_, stream)); }
    if (cc[1]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[1], nyzF*nFields, nt, xrN, fieldNcclComm_, stream));
                 NCCLCHECK(ncclSend(d_haloBuf_send_[1], nyzF*nFields, nt, xrN, fieldNcclComm_, stream)); }
    if (cc[2]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[2], nxzF*nFields, nt, ylN, fieldNcclComm_, stream));
                 NCCLCHECK(ncclSend(d_haloBuf_send_[2], nxzF*nFields, nt, ylN, fieldNcclComm_, stream)); }
    if (cc[3]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[3], nxzF*nFields, nt, yrN, fieldNcclComm_, stream));
                 NCCLCHECK(ncclSend(d_haloBuf_send_[3], nxzF*nFields, nt, yrN, fieldNcclComm_, stream)); }
    if (cc[4]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[4], nxyF*nFields, nt, zlN, fieldNcclComm_, stream));
                 NCCLCHECK(ncclSend(d_haloBuf_send_[4], nxyF*nFields, nt, zlN, fieldNcclComm_, stream)); }
    if (cc[5]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[5], nxyF*nFields, nt, zrN, fieldNcclComm_, stream));
                 NCCLCHECK(ncclSend(d_haloBuf_send_[5], nxyF*nFields, nt, zrN, fieldNcclComm_, stream)); }
    NCCLCHECK(ncclGroupEnd());

    // ---- Self-copy faces (periodic self-neighbour, batched) ----
    {
        constexpr int BLK = 16;
        dim3 block(BLK, BLK);
        if (xlN == myrank && xrN == myrank) {
            dim3 grid(((ny-2)+BLK-1)/BLK, ((nz-2)+BLK-1)/BLK, nFields);
            gpuBatchSelfCopyFaceX<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz, offset);
        }
        if (ylN == myrank && yrN == myrank) {
            dim3 grid(((nx-2)+BLK-1)/BLK, ((nz-2)+BLK-1)/BLK, nFields);
            gpuBatchSelfCopyFaceY<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz, offset);
        }
        if (zlN == myrank && zrN == myrank) {
            dim3 grid(((nx-2)+BLK-1)/BLK, ((ny-2)+BLK-1)/BLK, nFields);
            gpuBatchSelfCopyFaceZ<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz, offset);
        }
        // No sync needed: self-copy and unpack share the same stream, so
        // CUDA stream ordering guarantees self-copy completes first.
    }

    // ---- Unpack face recv buffers into ghost faces ----
    if (cc[0])
        launchUnpack2D_(d_haloBuf_recv_[rx0], d_ptrs, nFields,
                        0*ny*nz + 1*nz + 1, nz, 1, ny-2, nz-2, stream);
    if (cc[1])
        launchUnpack2D_(d_haloBuf_recv_[rx1], d_ptrs, nFields,
                        (nx-1)*ny*nz + 1*nz + 1, nz, 1, ny-2, nz-2, stream);
    if (cc[2])
        launchUnpack2D_(d_haloBuf_recv_[ry0], d_ptrs, nFields,
                        1*ny*nz + 0*nz + 1, ny*nz, 1, nx-2, nz-2, stream);
    if (cc[3])
        launchUnpack2D_(d_haloBuf_recv_[ry1], d_ptrs, nFields,
                        1*ny*nz + (ny-1)*nz + 1, ny*nz, 1, nx-2, nz-2, stream);
    if (cc[4])
        launchUnpack2D_(d_haloBuf_recv_[rz0], d_ptrs, nFields,
                        1*ny*nz + 1*nz + 0, ny*nz, nz, nx-2, ny-2, stream);
    if (cc[5])
        launchUnpack2D_(d_haloBuf_recv_[rz1], d_ptrs, nFields,
                        1*ny*nz + 1*nz + (nz-1), ny*nz, nz, nx-2, ny-2, stream);

    if (isFaceOnlyFlag) return;

    // =====================================================================
    //  PHASE 2: Edge exchange
    // =====================================================================
    {
        const int edgeYlen = ny - 2, edgeZlen = nz - 2, edgeXlen = nx - 2;

        // ---- Pack X-direction edges (Y-edges sent to X neighbours) ----
        for (int dir = 0; dir < 2; ++dir) {
            if (!cc[dir]) continue;
            int ix_send = (dir == 0) ? 1 : (nx - 2);
            cudaSolverType* buf = d_haloBuf_send_[dir];
            int off = 0;
            if (cc[4]) { launchPack2D_(buf + off, d_ptrs, nFields,
                                       ix_send*ny*nz + 1*nz + 0, nz, 1, edgeYlen, 1, stream);
                         off += edgeYlen * nFields; }
            if (cc[5]) { launchPack2D_(buf + off, d_ptrs, nFields,
                                       ix_send*ny*nz + 1*nz + (nz-1), nz, 1, edgeYlen, 1, stream);
                         off += edgeYlen * nFields; }
        }

        // ---- Pack Y-direction edges (Z-edges sent to Y neighbours) ----
        for (int dir = 2; dir < 4; ++dir) {
            if (!cc[dir]) continue;
            int iy_send = (dir == 2) ? 1 : (ny - 2);
            cudaSolverType* buf = d_haloBuf_send_[dir];
            int off = 0;
            if (cc[0]) { launchPack2D_(buf + off, d_ptrs, nFields,
                                       0*ny*nz + iy_send*nz + 1, 1, 1, edgeZlen, 1, stream);
                         off += edgeZlen * nFields; }
            if (cc[1]) { launchPack2D_(buf + off, d_ptrs, nFields,
                                       (nx-1)*ny*nz + iy_send*nz + 1, 1, 1, edgeZlen, 1, stream);
                         off += edgeZlen * nFields; }
        }

        // ---- Pack Z-direction edges (X-edges sent to Z neighbours) ----
        for (int dir = 4; dir < 6; ++dir) {
            if (!cc[dir]) continue;
            int iz_send = (dir == 4) ? 1 : (nz - 2);
            cudaSolverType* buf = d_haloBuf_send_[dir];
            int off = 0;
            if (cc[2]) { launchPack2D_(buf + off, d_ptrs, nFields,
                                       1*ny*nz + 0*nz + iz_send, ny*nz, 1, edgeXlen, 1, stream);
                         off += edgeXlen * nFields; }
            if (cc[3]) { launchPack2D_(buf + off, d_ptrs, nFields,
                                       1*ny*nz + (ny-1)*nz + iz_send, ny*nz, 1, edgeXlen, 1, stream);
                         off += edgeXlen * nFields; }
        }

        // ---- NCCL edge exchange: one group covering all active directions ----
        NCCLCHECK(ncclGroupStart());
        // Y-edges <-> X neighbours
        for (int dir = 0; dir < 2; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[4] ? 1 : 0) + (cc[5] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 0) ? xlN : xrN;
            NCCLCHECK(ncclRecv(d_haloBuf_recv_[dir], nEdges*edgeYlen*nFields, nt, neighbor, fieldNcclComm_, stream));
            NCCLCHECK(ncclSend(d_haloBuf_send_[dir], nEdges*edgeYlen*nFields, nt, neighbor, fieldNcclComm_, stream));
        }
        // Z-edges <-> Y neighbours
        for (int dir = 2; dir < 4; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[0] ? 1 : 0) + (cc[1] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 2) ? ylN : yrN;
            NCCLCHECK(ncclRecv(d_haloBuf_recv_[dir], nEdges*edgeZlen*nFields, nt, neighbor, fieldNcclComm_, stream));
            NCCLCHECK(ncclSend(d_haloBuf_send_[dir], nEdges*edgeZlen*nFields, nt, neighbor, fieldNcclComm_, stream));
        }
        // X-edges <-> Z neighbours
        for (int dir = 4; dir < 6; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[2] ? 1 : 0) + (cc[3] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 4) ? zlN : zrN;
            NCCLCHECK(ncclRecv(d_haloBuf_recv_[dir], nEdges*edgeXlen*nFields, nt, neighbor, fieldNcclComm_, stream));
            NCCLCHECK(ncclSend(d_haloBuf_send_[dir], nEdges*edgeXlen*nFields, nt, neighbor, fieldNcclComm_, stream));
        }
        NCCLCHECK(ncclGroupEnd());

        // ---- Self-copy edges (periodic, batched) ----
        {
            int maxDim = (nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz));
            int nblks = (maxDim + 255) / 256;
            if (xlN == myrank && xrN == myrank) {
                dim3 grid(nblks, nFields);
                gpuBatchSelfCopyEdgeX<<<grid, 256, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                    zrN != MPI_PROC_NULL, zlN != MPI_PROC_NULL,
                    yrN != MPI_PROC_NULL, ylN != MPI_PROC_NULL);
            }
            if (ylN == myrank && yrN == myrank) {
                dim3 grid(nblks, nFields);
                gpuBatchSelfCopyEdgeY<<<grid, 256, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                    xrN != MPI_PROC_NULL, xlN != MPI_PROC_NULL,
                    zrN != MPI_PROC_NULL, zlN != MPI_PROC_NULL);
            }
            if (zlN == myrank && zrN == myrank) {
                dim3 grid(nblks, nFields);
                gpuBatchSelfCopyEdgeZ<<<grid, 256, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                    yrN != MPI_PROC_NULL, ylN != MPI_PROC_NULL,
                    xrN != MPI_PROC_NULL, xlN != MPI_PROC_NULL);
            }
        }

        // ---- Unpack edge recv buffers ----
        // Y-edges from X neighbours -> ghost ix=0 or nx-1
        for (int dir = 0; dir < 2; ++dir) {
            if (!cc[dir]) continue;
            int ix_recv = (dir == 0) ? 0 : (nx - 1);
            const cudaSolverType* buf = d_haloBuf_recv_[dir == 0 ? rx0 : rx1];
            int off = 0;
            if (cc[4]) { launchUnpack2D_(buf + off, d_ptrs, nFields,
                                         ix_recv*ny*nz + 1*nz + 0, nz, 1, edgeYlen, 1, stream);
                         off += edgeYlen * nFields; }
            if (cc[5]) { launchUnpack2D_(buf + off, d_ptrs, nFields,
                                         ix_recv*ny*nz + 1*nz + (nz-1), nz, 1, edgeYlen, 1, stream);
                         off += edgeYlen * nFields; }
        }
        // Z-edges from Y neighbours -> ghost iy=0 or ny-1
        for (int dir = 2; dir < 4; ++dir) {
            if (!cc[dir]) continue;
            int iy_recv = (dir == 2) ? 0 : (ny - 1);
            const cudaSolverType* buf = d_haloBuf_recv_[dir == 2 ? ry0 : ry1];
            int off = 0;
            if (cc[0]) { launchUnpack2D_(buf + off, d_ptrs, nFields,
                                         0*ny*nz + iy_recv*nz + 1, 1, 1, edgeZlen, 1, stream);
                         off += edgeZlen * nFields; }
            if (cc[1]) { launchUnpack2D_(buf + off, d_ptrs, nFields,
                                         (nx-1)*ny*nz + iy_recv*nz + 1, 1, 1, edgeZlen, 1, stream);
                         off += edgeZlen * nFields; }
        }
        // X-edges from Z neighbours -> ghost iz=0 or nz-1
        for (int dir = 4; dir < 6; ++dir) {
            if (!cc[dir]) continue;
            int iz_recv = (dir == 4) ? 0 : (nz - 1);
            const cudaSolverType* buf = d_haloBuf_recv_[dir == 4 ? rz0 : rz1];
            int off = 0;
            if (cc[2]) { launchUnpack2D_(buf + off, d_ptrs, nFields,
                                         1*ny*nz + 0*nz + iz_recv, ny*nz, 1, edgeXlen, 1, stream);
                         off += edgeXlen * nFields; }
            if (cc[3]) { launchUnpack2D_(buf + off, d_ptrs, nFields,
                                         1*ny*nz + (ny-1)*nz + iz_recv, ny*nz, 1, edgeXlen, 1, stream);
                         off += edgeXlen * nFields; }
        }

        // =================================================================
        //  PHASE 3: Corner exchange
        // =================================================================
        if ((cc[2] || cc[3]) && (cc[4] || cc[5])) {
            // ---- Pack corners: 4 elements per field, per X direction ----
            if (cc[0]) {
                int total = nFields * 4;
                k_batchPackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_send_[0], d_ptrs, nFields, 1*ny*nz, ny, nz);
            }
            if (cc[1]) {
                int total = nFields * 4;
                k_batchPackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_send_[1], d_ptrs, nFields, (nx-2)*ny*nz, ny, nz);
            }

            // ---- NCCL corner exchange ----
            NCCLCHECK(ncclGroupStart());
            if (cc[0]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[0], 4*nFields, nt, xlN, fieldNcclComm_, stream));
                         NCCLCHECK(ncclSend(d_haloBuf_send_[0], 4*nFields, nt, xlN, fieldNcclComm_, stream)); }
            if (cc[1]) { NCCLCHECK(ncclRecv(d_haloBuf_recv_[1], 4*nFields, nt, xrN, fieldNcclComm_, stream));
                         NCCLCHECK(ncclSend(d_haloBuf_send_[1], 4*nFields, nt, xrN, fieldNcclComm_, stream)); }
            NCCLCHECK(ncclGroupEnd());

            // ---- Corner self-copy (periodic) ----
            {
                if (xlN == myrank && xrN == myrank) {
                    gpuBatchSelfCopyCornerX<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                        ylN != MPI_PROC_NULL, yrN != MPI_PROC_NULL,
                        zlN != MPI_PROC_NULL, zrN != MPI_PROC_NULL);
                } else if (ylN == myrank && yrN == myrank) {
                    gpuBatchSelfCopyCornerY<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                        xlN != MPI_PROC_NULL, xrN != MPI_PROC_NULL,
                        zlN != MPI_PROC_NULL, zrN != MPI_PROC_NULL);
                } else if (zlN == myrank && zrN == myrank) {
                    gpuBatchSelfCopyCornerZ<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                        ylN != MPI_PROC_NULL, yrN != MPI_PROC_NULL,
                        xlN != MPI_PROC_NULL, xrN != MPI_PROC_NULL);
                }
            }

            // ---- Unpack corners ----
            if (cc[0]) {
                int total = nFields * 4;
                k_batchUnpackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_recv_[rx0], d_ptrs, nFields, 0*ny*nz, ny, nz);
            }
            if (cc[1]) {
                int total = nFields * 4;
                k_batchUnpackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_recv_[rx1], d_ptrs, nFields, (nx-1)*ny*nz, ny, nz);
            }
        } else {
            // Match CPU NBDerivedHaloComm: local periodic corner copies are
            // still required even when no non-self corner exchange exists.
            if (xlN == myrank && xrN == myrank) {
                gpuBatchSelfCopyCornerX<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                    ylN != MPI_PROC_NULL, yrN != MPI_PROC_NULL,
                    zlN != MPI_PROC_NULL, zrN != MPI_PROC_NULL);
            } else if (ylN == myrank && yrN == myrank) {
                gpuBatchSelfCopyCornerY<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                    xlN != MPI_PROC_NULL, xrN != MPI_PROC_NULL,
                    zlN != MPI_PROC_NULL, zrN != MPI_PROC_NULL);
            } else if (zlN == myrank && zrN == myrank) {
                gpuBatchSelfCopyCornerZ<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz, offset,
                    ylN != MPI_PROC_NULL, yrN != MPI_PROC_NULL,
                    xlN != MPI_PROC_NULL, xrN != MPI_PROC_NULL);
            }
        }
    } // end PHASE 2 + PHASE 3 block
}


#endif // CUDA_GRAPH && USE_NCCL
#endif // GPU_SOLVER
