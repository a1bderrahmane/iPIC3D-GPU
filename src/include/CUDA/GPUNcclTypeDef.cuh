#pragma once

#if defined(CUDA_GRAPH) && defined(USE_NCCL)
#ifdef HIPIFLY
#include <rccl.h>
#else
#include <nccl.h>
#endif
#include "cudaTypeDef.cuh" // for cudaSolverType

#define NCCLCHECK(cmd) do {                                  \
  ncclResult_t r = (cmd);                                    \
  if (r != ncclSuccess) {                                    \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",                \
            __FILE__, __LINE__, ncclGetErrorString(r));       \
    MPI_Abort(MPI_COMM_WORLD, 1);                             \
  }                                                            \
} while (0)

// Maps cudaSolverType (float or double, see cudaTypeDef.cuh) to ncclDataType_t
template <typename T> inline ncclDataType_t ncclTypeOf();
template <> inline ncclDataType_t ncclTypeOf<float>()  { return ncclFloat;  }
template <> inline ncclDataType_t ncclTypeOf<double>() { return ncclDouble; }

#endif // CUDA_GRAPH && USE_NCCL
