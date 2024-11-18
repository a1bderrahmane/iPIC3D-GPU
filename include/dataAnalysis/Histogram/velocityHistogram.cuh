#ifndef _VELOCITY_HISTOGRAM_
#define _VELOCITY_HISTOGRAM_

#include "cuda.h"
#include "cudaTypeDef.cuh"


#include "Particle.h"
#include "particleArrayCUDA.cuh"
#include "particleArraySoACUDA.cuh"

#include <iostream>
#include <fstream>


namespace histogram{

template <typename U, int dim, typename T = int>
class histogramCUDA {

private:
    T* hostPtr;
    T* cudaPtr;

    int bufferSize; // the physical size of current buffer, in elements
public:
    int size[dim];  // the logic size of each dimension, in elements
private:
    int logicSize;  // the logic size of the whole histogram

    U min[dim], max[dim], resolution[dim];

public:

    /**
     * @param bufferSize the physical size of the buffer, in elements
     */
    __host__ histogramCUDA(int bufferSize): bufferSize(bufferSize){
        allocate();
        resetBuffer();
    }

    /**
     * @param min the minimum value of each dimension
     * @param max the maximum value of each dimension
     * @param resolution the resolution of each dimension
     */
    __host__ void setHistogram(U* min, U* max, int* binThisDim){
        
        for(int i=0; i<dim; i++){
            if(min[i] >= max[i] || binThisDim[i] <= 0){
                std::cerr << "[!]Invalid histogram range or binThisDim" << std::endl;
                std::cerr << "[!]min: " << min[i] << " max: " << max[i] << " binThisDim: " << binThisDim[i] << std::endl;
                return;
            }
        }

        logicSize = 1;
        for(int i=0; i<dim; i++){
            this->min[i] = min[i];
            this->max[i] = max[i];

            size[i] = binThisDim[i];
            this->resolution[i] = (max[i] - min[i]) / size[i];
            logicSize *= size[i];
        }

        if(bufferSize < logicSize){
            cudaErrChk(cudaFreeHost(hostPtr));
            cudaErrChk(cudaFree(cudaPtr));
            bufferSize = logicSize;
            allocate();
        }
        
        resetBuffer();
    }

    __device__ void setHistogramDevice(U* min, U* max, int* binThisDim){
        
        for(int i=0; i<dim; i++){
            if(min[i] >= max[i] || binThisDim[i] <= 0){
                assert(0);
                return;
            }
        }

        logicSize = 1;
        for(int i=0; i<dim; i++){
            this->min[i] = min[i];
            this->max[i] = max[i];

            size[i] = binThisDim[i];
            this->resolution[i] = (max[i] - min[i]) / size[i];
            logicSize *= size[i];
        }

        if(bufferSize < logicSize){
            assert(0);
        }
        
    }

    __host__ void copyHistogramAsync(cudaStream_t stream = 0){
        cudaErrChk(cudaMemcpyAsync(hostPtr, cudaPtr, logicSize * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    __host__ T* getHistogram(){
        return hostPtr;
    }

    __host__ T* getHistogramCUDA(){
        return cudaPtr;
    }

    __host__ __device__ int getLogicSize(){
        return logicSize;
    }

    __host__ void getSize(int* size){
        for(int i=0; i<dim; i++){
            size[i] = this->size[i];
        }
    }

    __host__ U getMin(int index){
        return min[index];
    }

    __host__ U getMax(int index){
        return max[index];
    }

    __host__ U getResolution(int index){
        return resolution[index];
    }
    
    __device__ void addData(const U* data, const int count = 1){
        int index = 0;
        
        for(int i=dim-1; i>=0; i--){
            // check the range
            if(data[i] < min[i] || data[i] > max[i]){assert(0); return;}

            auto tmp = (int)((data[i] - min[i]) / resolution[i]);
            if(tmp == size[i])tmp--; // the max value
            index +=  tmp;
            if(i != 0)index *= size[i-1];
        }

        if(index >= logicSize)return;
        atomicAdd(&cudaPtr[index], count);
    }

    /**
     * @brief get the center of the bin
     * @param index the index of the bin, in the buffer
     */
    __device__ void centerOfBin(int index, U* center){
        int tmp = index;
        for(int i=0; i<dim; i++){
            center[i] = min[i] + (tmp % size[i] + 0.5) * resolution[i];
            tmp /= size[i];
        }
    }
private:

    __host__ void allocate(){

        cudaErrChk(cudaMallocHost((void**)&hostPtr, bufferSize * sizeof(T)));
        cudaErrChk(cudaMalloc((void**)&cudaPtr, bufferSize * sizeof(T)));
    }

    __host__ void resetBuffer(){
        cudaErrChk(cudaMemset(cudaPtr, 0, bufferSize * sizeof(T)));
    }

public:

    __host__ void resetBufferAsync(cudaStream_t stream = 0){
        cudaErrChk(cudaMemsetAsync(cudaPtr, 0, bufferSize * sizeof(T), stream));
    }

    __host__ ~histogramCUDA(){
        cudaErrChk(cudaFreeHost(hostPtr));
        cudaErrChk(cudaFree(cudaPtr));
    }

};

}


namespace velocityHistogram{

using velocityHistogramCUDA = histogram::histogramCUDA<cudaCommonType, 2, int>;

using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaCommonType, 0, 2>;

__global__ void scaleMarkKernel(velocityHistogramCUDA* histogramCUDAPtr, cudaCommonType* dim0, cudaCommonType* dim1);


/**
 * @brief Histogram for one species
 */
class velocityHistogram
{
private:
    // UVW
    velocityHistogramCUDA* histogramHostPtr;

    velocityHistogramCUDA* histogramCUDAPtr; // one buffer for 3 objects

    int binThisDim[2] = {100, 100};

    int reductionTempArraySize = 0;
    cudaCommonType* reductionTempArrayCUDA;
    cudaCommonType* reductionMinResultCUDA;
    cudaCommonType* reductionMaxResultCUDA;

    int cycleNum;


    bool bigEndian;

    int reduceBlockNum(int dataSize, int blockSize){
        if(dataSize < 4096)dataSize = 4096;
        auto blockNum = getGridSize(dataSize / 4096, blockSize); // 4096 elements per thread
        blockNum = blockNum > 1024 ? 1024 : blockNum;

        if(reductionTempArraySize < blockNum){
            cudaErrChk(cudaFree(reductionTempArrayCUDA));
            cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(cudaCommonType)*blockNum * 6));
            reductionTempArraySize = blockNum;
        }

        return blockNum;
    }


    /**
     * @brief get the Max and Min value of the given value set
     */
    __host__ int getRange(velocitySoA* pclArray, cudaStream_t stream);

public:

    /**
     * @param initSize the initial size of the histogram buffer, in elements
     * @param path the path to store the output file, directory
     */
    __host__ velocityHistogram(int initSize) {

        reductionTempArraySize = 1024;
        cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(cudaCommonType)*reductionTempArraySize * 6));

        cudaErrChk(cudaMalloc(&reductionMinResultCUDA, sizeof(cudaCommonType)*6));
        reductionMaxResultCUDA = reductionMinResultCUDA + 3;

        histogramHostPtr = newHostPinnedObjectArray<velocityHistogramCUDA>(3, initSize);
        cudaErrChk(cudaMalloc(&histogramCUDAPtr, sizeof(velocityHistogramCUDA) * 3));
        cudaErrChk(cudaMemcpy(histogramCUDAPtr, histogramHostPtr, sizeof(velocityHistogramCUDA) * 3, cudaMemcpyDefault));
        
        { // check the endian
            int test = 1;
            char* ptr = reinterpret_cast<char*>(&test);
            if (*ptr == 1) {
                bigEndian = false;
            } else {
                bigEndian = true;
            }
        }
    }

    /**
     * @brief Initiate the kernels for histograming, launch the kernels
     * @details It can be invoked after Moment in the main loop, for the output and solver are on CPU
     */
    __host__ void init(velocitySoA* pclArray, int cycleNum, cudaStream_t stream = 0);


    /**
     * @brief Wait for the histogram data to be ready, write the output to file
     * @details Output the 3 velocity histograms to file, for this species in this subdomain
     *          It should be invoked after a previous Init, can be after the B, for it's on CPU
     */
    __host__ void writeToFile(std::string filePath, cudaStream_t stream = 0){
        cudaErrChk(cudaStreamSynchronize(stream));
        
        for(int i=0; i<3; i++){
            histogramHostPtr[i].copyHistogramAsync(stream);
        }

        cudaErrChk(cudaStreamSynchronize(stream));
        
        std::string items[3] = {"UV", "VW", "UW"};

        for(int i=0; i<3; i++){ // UV, VW, UW
            std::ostringstream ossFileName;
            ossFileName << filePath << "/velocityHistogram_" << MPIdata::get_rank() << "_" << items[i] << "_" << cycleNum << ".vtk";

            std::ofstream vtkFile(ossFileName.str(), std::ios::binary);

            vtkFile << "# vtk DataFile Version 3.0\n";
            vtkFile << "Velocity Histogram\n";
            vtkFile << "BINARY\n";  
            vtkFile << "DATASET STRUCTURED_POINTS\n";
            vtkFile << "DIMENSIONS " << histogramHostPtr[i].size[0] << " " << histogramHostPtr[i].size[1] << " 1\n";
            vtkFile << "ORIGIN " << histogramHostPtr[i].getMin(0) << " " << histogramHostPtr[i].getMin(1) << " 0\n"; 
            vtkFile << "SPACING " << histogramHostPtr[i].getResolution(0) << " " << histogramHostPtr[i].getResolution(1) << " 1\n";  
            vtkFile << "POINT_DATA " << histogramHostPtr[i].getLogicSize() << "\n";  
            vtkFile << "SCALARS scalars int 1\n";  
            vtkFile << "LOOKUP_TABLE default\n";  

            auto histogramBuffer = histogramHostPtr[i].getHistogram();
            for (int j = 0; j < histogramHostPtr[i].getLogicSize(); j++) {
                int value = histogramBuffer[j];
                
                value = __builtin_bswap32(value);

                vtkFile.write(reinterpret_cast<char*>(&value), sizeof(int));
            }

            vtkFile.close();
        }


    }

    __host__ velocityHistogramCUDA* getVelocityHistogramResult(cudaStream_t stream = 0){
        cudaErrChk(cudaStreamSynchronize(stream));
        // now the histogram results on the device buffer

        return histogramHostPtr;
    }

    __host__ void computeScaleMark(int i, cudaCommonType* scaleMark0, cudaCommonType* scaleMark1, cudaStream_t stream = 0){
        scaleMarkKernel<<<getGridSize(histogramHostPtr[i].getLogicSize(), 256), 256, 0, stream>>>(histogramCUDAPtr + i, scaleMark0, scaleMark1);
    }

    ~velocityHistogram(){

        cudaErrChk(cudaFree(reductionTempArrayCUDA));
        cudaErrChk(cudaFree(reductionMinResultCUDA));

        cudaErrChk(cudaFree(histogramCUDAPtr));
        deleteHostPinnedObjectArray(histogramHostPtr, 3);
    }
};





    
}






#endif