#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

#include <iostream>
#include <cuda.h>
#include "cudaReduction.cuh"


namespace velocityHistogram
{

__global__ void histogramUpdateKernel(cudaCommonType* minUVW, cudaCommonType* maxUVW, int binDim0, int binDim1, velocityHistogramCUDA* histogramCUDAPtr){
    int idx = threadIdx.x;
    if(idx >= 3)return;

    cudaCommonType min[2];
    cudaCommonType max[2];
    int binDim[2] = {binDim0, binDim1};

    if(idx == 0){
        min[0] = minUVW[0];
        min[1] = minUVW[1];
        max[0] = maxUVW[0];
        max[1] = maxUVW[1];
    }else if(idx == 1){
        min[0] = minUVW[0];
        min[1] = minUVW[2];
        max[0] = maxUVW[0];
        max[1] = maxUVW[2];
        
    }else{
        min[0] = minUVW[1];
        min[1] = minUVW[2];
        max[0] = maxUVW[1];
        max[1] = maxUVW[2];
    }


    histogramCUDAPtr[idx].setHistogramDevice(min, max, binDim);


}



__host__ void velocityHistogram::init(velocitySoA* pclArray, int cycleNum, cudaStream_t stream){
    using namespace particleArraySoA;
    
    // reset the histogram buffer, assuming enough size
    for(int i=0; i<3; i++){
        histogramHostPtr[i]->resetBufferAsync(stream);   
    }
    getRange(pclArray, stream);
    
    histogramUpdateKernel<<<1, 3, 0, stream>>>(reductionMinResultCUDA, reductionMaxResultCUDA, 
                                                binThisDim[0], binThisDim[1], 
                                                histogramCUDAPtr[0]);

    velocityHistogramKernel<<<getGridSize((int)pclArray->getNOP(), 256), 256, 0, stream>>>
        (pclArray->getNOP(), pclArray->getElement(U), pclArray->getElement(V), pclArray->getElement(W),
        histogramCUDAPtr[0], histogramCUDAPtr[1], histogramCUDAPtr[2]);
    // copy the histogram object to host
    for(int i=0; i<3; i++){
        cudaErrChk(cudaMemcpyAsync(histogramHostPtr[i], histogramCUDAPtr[i], 
                                    sizeof(velocityHistogramCUDA), cudaMemcpyDeviceToHost, stream));
    }

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















