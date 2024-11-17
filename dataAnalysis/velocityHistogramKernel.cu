
#include "cudaTypeDef.cuh"
#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

namespace velocityHistogram
{

using namespace particleArraySoA;

/**
 * @brief kernel function to calculate the velocity histogram for one species
 * @details the 3 histograms have been updated in advance
 * 
 * @param pclArray the particle array
 * @param histogramCUDAPtrUV the histogram for UV
 * @param histogramCUDAPtrVW the histogram for VW
 * @param histogramCUDAPtrUW the histogram for UW
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



} // namespace velocityHistogram







