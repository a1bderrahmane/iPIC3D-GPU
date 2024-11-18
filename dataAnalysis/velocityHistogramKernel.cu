
#include "cudaTypeDef.cuh"
#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

namespace velocityHistogram
{

using namespace particleArraySoA;

/**
/**
 * @brief Kernel function to compute velocity histograms.
 * @details launched for each particle
 *
 * @param nop Number of particles.
 * @param u Pointer to the array of u velocity components.
 * @param v Pointer to the array of v velocity components.
 * @param w Pointer to the array of w velocity components.
 * @param histogramCUDAPtr Pointer to the array of 3 velocityHistogramCUDA objects.
 */
__global__ void velocityHistogramKernel(int nop, cudaCommonType* u, cudaCommonType* v, cudaCommonType* w,
                                        velocityHistogramCUDA* histogramCUDAPtr){

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    if(pidx >= nop)return;

    const cudaCommonType uvw[3] = {u[pidx], v[pidx], w[pidx]};
    const cudaCommonType uv[2] = {uvw[0], uvw[1]};
    const cudaCommonType vw[2] = {uvw[1], uvw[2]};
    const cudaCommonType uw[2] = {uvw[0], uvw[2]};

    histogramCUDAPtr[0].addData(uv, 1);
    histogramCUDAPtr[1].addData(vw, 1);
    histogramCUDAPtr[2].addData(uw, 1);

}

/**
 * @brief calculate the center of each histogram bin
 * @details this kernel is launched once for each histogram bin
 */
__global__ void scaleMarkKernel(velocityHistogramCUDA* histogramCUDAPtr, cudaCommonType* dim0, cudaCommonType* dim1){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= histogramCUDAPtr->getLogicSize())return;

    cudaCommonType center[2];
    histogramCUDAPtr->centerOfBin(idx, center);

    dim0[idx] = center[0];
    dim1[idx] = center[1];

}



} // namespace velocityHistogram







