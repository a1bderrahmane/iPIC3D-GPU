#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

#include <iostream>
#include <cuda.h>

template <typename T, unsigned int blockSize>
__global__ void reduceMax(T* g_idata, T* g_odata, unsigned int n);

template <typename T, unsigned int blockSize>
__global__ void reduceMin(T* g_idata, T* g_odata, unsigned int n);


namespace velocityHistogram
{

__host__ void velocityHistogram::init(velocitySoA* pclArray, int cycleNum, cudaStream_t stream){
    
    getRange(pclArray, stream);
    prepareHistogram();
    velocityHistogramKernel<<<getGridSize((int)pclArray->getNOP(), 256), 256, 0, stream>>>
                            (pclArray, histogramCUDAPtr[0], histogramCUDAPtr[1], histogramCUDAPtr[2]);
    // copy the histogram data to host
    for(int i=0; i<3; i++){
        histogramHostPtr[i]->copyHistogramAsync(stream);
    }

    this->cycleNum = cycleNum;
}

__host__ int velocityHistogram::getRange(velocitySoA* pclArray, cudaStream_t stream){
    constexpr int blockSize = 256;
    int gridSize = getGridSize((int)pclArray->getNOP(), blockSize * 2) / 8; // 2 * 8 data per thread, the total ratio is 4096

    // output host array
    cudaCommonType* minOutput = nullptr; cudaCommonType* maxOutput = nullptr;
    cudaErrChk(cudaMallocHost( &minOutput, sizeof(cudaCommonType) * gridSize * 3 * 2));
    maxOutput = minOutput + gridSize * 3;
    // output device array
    cudaCommonType* minOutputDevice = nullptr; cudaCommonType* maxOutputDevice = nullptr;
    cudaErrChk(cudaMalloc(&minOutputDevice, sizeof(cudaCommonType) * gridSize * 3 * 2));
    maxOutputDevice = minOutputDevice + gridSize * 3;

    // std::cout << "nop: " << pclArray->getNOP() << std::endl;
    for(int i=0; i<3; i++){ // UVW
        reduceMin<cudaCommonType, blockSize><<<gridSize, blockSize, blockSize * sizeof(cudaCommonType), stream>>>
            (pclArray->getElement(i), minOutputDevice + i * gridSize, pclArray->getNOP());
        reduceMax<cudaCommonType, blockSize><<<gridSize, blockSize, blockSize * sizeof(cudaCommonType), stream>>>
            (pclArray->getElement(i), maxOutputDevice + i * gridSize, pclArray->getNOP());
    }
    // copy the min and max value to host
    cudaErrChk(cudaMemcpyAsync(minOutput, minOutputDevice, sizeof(cudaCommonType) * gridSize * 3 * 2, cudaMemcpyDeviceToHost, stream));

    cudaErrChk(cudaStreamSynchronize(stream));
    
    cudaCommonType min[3] = {minOutput[0], minOutput[gridSize], minOutput[gridSize * 2]};
    cudaCommonType max[3] = {maxOutput[0], maxOutput[gridSize], maxOutput[gridSize * 2]};
    cudaCommonType resolution[3] = {0};

    for(int i=0; i<3; i++){ // 3 directions
        // get the min and max value from the output array
        for(int j=0; j<gridSize; j++){
            min[i] = min[i] > minOutput[i * gridSize + j] ? minOutput[i * gridSize + j] : min[i];
            max[i] = max[i] < maxOutput[i * gridSize + j] ? maxOutput[i * gridSize + j] : max[i];
        }
        resolution[i] = (max[i] - min[i]) / 100;
    }

    // fill the minArray, maxArray, resolutionArray
    // this is ugly, but required for the 1-3D compatibility
    minArray[0][0] = min[0];
    minArray[0][1] = min[1];
    maxArray[0][0] = max[0];
    maxArray[0][1] = max[1];
    resolutionArray[0][0] = resolution[0];
    resolutionArray[0][1] = resolution[1];

    minArray[1][0] = min[1];
    minArray[1][1] = min[2];
    maxArray[1][0] = max[1];
    maxArray[1][1] = max[2];
    resolutionArray[1][0] = resolution[1];
    resolutionArray[1][1] = resolution[2];

    minArray[2][0] = min[0];
    minArray[2][1] = min[2];
    maxArray[2][0] = max[0];
    maxArray[2][1] = max[2];
    resolutionArray[2][0] = resolution[0];
    resolutionArray[2][1] = resolution[2];



    return 0;

}



}



template <typename T, unsigned int blockSize>
__device__ void warpReduceMax(volatile T* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}



template <typename T, unsigned int blockSize>
__global__ void reduceMax(T* g_idata, T* g_odata, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = g_idata[i];

    while (i < n) {
        sdata[tid] = max(sdata[tid], g_idata[i]);
        if (i + blockSize < n)
            sdata[tid] = max(sdata[tid], g_idata[i + blockSize]);
        i += gridSize;
    }

    __syncthreads();


    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32) warpReduceMax<T, blockSize>(sdata, tid);


    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}




template <typename T, unsigned int blockSize>
__device__ void warpReduceMin(volatile T *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <typename T, unsigned int blockSize>
__global__ void reduceMin(T* g_idata, T* g_odata, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = g_idata[i];

    while (i < n) {
        sdata[tid] = min(sdata[tid], g_idata[i]);
        if (i + blockSize < n)
            sdata[tid] = min(sdata[tid], g_idata[i + blockSize]);
        i += gridSize;
    }

    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < 32) warpReduceMin<T, blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}











