
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

__global__ void velocityHistogramKernel(int nop, cudaCommonType* u, cudaCommonType* v, cudaCommonType* w, cudaCommonType* q,
                                        velocityHistogramCUDA* histogramCUDAPtr){

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    if(pidx >= nop)return;

    const cudaCommonType uvw[3] = {u[pidx], v[pidx], w[pidx]};
    const cudaCommonType uv[2] = {uvw[0], uvw[1]};
    const cudaCommonType vw[2] = {uvw[1], uvw[2]};
    const cudaCommonType uw[2] = {uvw[0], uvw[2]};

    const auto qAbs = abs(q[pidx] * 10e5);
    //const int qAbs = 1;

    histogramCUDAPtr[0].addData(uv, qAbs);
    histogramCUDAPtr[1].addData(vw, qAbs);
    histogramCUDAPtr[2].addData(uw, qAbs);

}

/**
 * @brief reset and calculate the center of each histogram bin
 * @details this kernel is launched once for each histogram bin for all 3 histograms
 */
__global__ void resetBinScaleMarkKernel(velocityHistogramCUDA* histogramCUDAPtr){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= histogramCUDAPtr->getLogicSize())return;

    
    histogramCUDAPtr[0].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[0].centerOfBin(idx);
    histogramCUDAPtr[1].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[1].centerOfBin(idx);
    histogramCUDAPtr[2].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[2].centerOfBin(idx);

}



} // namespace velocityHistogram







