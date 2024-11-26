#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

#include <iostream>
#include "cudaTypeDef.cuh"
#include "cudaReduction.cuh"


namespace velocityHistogram
{

__global__ void histogramUpdateKernel(cudaCommonType* minUVW, cudaCommonType* maxUVW, int binDim0, int binDim1, velocityHistogramCUDA* histogramCUDAPtr){
    int idx = threadIdx.x;
    if(idx >= 3)return;

    cudaCommonType min[2];
    cudaCommonType max[2];
    int binDim[2] = {binDim0, binDim1};

    if(idx == 0){ // UV
        min[0] = minUVW[0];
        min[1] = minUVW[1];
        max[0] = maxUVW[0];
        max[1] = maxUVW[1];
    }else if(idx == 1){ // VW
        min[0] = minUVW[1];
        min[1] = minUVW[2];
        max[0] = maxUVW[1];
        max[1] = maxUVW[2];
    }else{ // UW
        min[0] = minUVW[0];
        min[1] = minUVW[2];
        max[0] = maxUVW[0];
        max[1] = maxUVW[2];
    }


    histogramCUDAPtr[idx].setHistogramDevice(min, max, binDim);


}


__global__ void velocityHistogramKernel(int nop, cudaCommonType* u, cudaCommonType* v, cudaCommonType* w, cudaCommonType* q,
                                        velocityHistogramCUDA* histogramCUDAPtr);

__global__ void resetBinScaleMarkKernel(velocityHistogramCUDA* histogramCUDAPtr);


__host__ void velocityHistogram::init(velocitySoA* pclArray, int cycleNum, cudaStream_t stream){
    using namespace particleArraySoA;
    
    getRange(pclArray, stream);
    histogramUpdateKernel<<<1, 3, 0, stream>>>(reductionMinResultCUDA, reductionMaxResultCUDA, 
                                                binThisDim[0], binThisDim[1], 
                                                histogramCUDAPtr);

    // reset the histogram buffer, set the scalMark
    resetBinScaleMarkKernel<<<getGridSize(binThisDim[0] * binThisDim[1], 256), 256, 0, stream>>>(histogramCUDAPtr);

    velocityHistogramKernel<<<getGridSize((int)pclArray->getNOP(), 256), 256, 0, stream>>>
        (pclArray->getNOP(), pclArray->getElement(U), pclArray->getElement(V), pclArray->getElement(W), pclArray->getElement(Q),
        histogramCUDAPtr);
    // copy the histogram object to host
    cudaErrChk(cudaMemcpyAsync(histogramHostPtr, histogramCUDAPtr, 3 * sizeof(velocityHistogramCUDA), cudaMemcpyDefault, stream));

    this->cycleNum = cycleNum;
}

__host__ int velocityHistogram::getRange(velocitySoA* pclArray, cudaStream_t stream){
    using namespace cudaReduction;

    constexpr int blockSize = 256;
    auto blockNum = reduceBlockNum(pclArray->getNOP(), blockSize);

    // std::cout << "nop: " << pclArray->getNOP() << std::endl;
    for(int i=0; i<3; i++){ // UVW
        reduceMin<cudaCommonType, blockSize><<<blockNum, blockSize, blockSize * sizeof(cudaCommonType), stream>>>
            (pclArray->getElement(i), reductionTempArrayCUDA + i * reductionTempArraySize, pclArray->getNOP());
        reduceMinWarp<cudaCommonType><<<1, 32, 0, stream>>>
            (reductionTempArrayCUDA + i * reductionTempArraySize, reductionMinResultCUDA + i, blockNum);

        reduceMax<cudaCommonType, blockSize><<<blockNum, blockSize, blockSize * sizeof(cudaCommonType), stream>>>
            (pclArray->getElement(i), reductionTempArrayCUDA + (i+3) * reductionTempArraySize, pclArray->getNOP());
        reduceMaxWarp<cudaCommonType><<<1, 32, 0, stream>>>
            (reductionTempArrayCUDA + (i+3) * reductionTempArraySize, reductionMaxResultCUDA + i, blockNum);
    }

    return 0;

}



}















