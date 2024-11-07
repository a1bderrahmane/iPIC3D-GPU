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
    __host__ void setHistogram(U* min, U* max, U* resolution){
        
        for(int i=0; i<dim; i++){
            if(min[i] >= max[i] || resolution[i] <= 0){
                std::cerr << "[!]Invalid histogram range or resolution" << std::endl;
                std::cerr << "[!]min: " << min[i] << " max: " << max[i] << " res: " << resolution[i] << std::endl;
                return;
            }
        }

        logicSize = 1;
        for(int i=0; i<dim; i++){
            this->min[i] = min[i];
            this->max[i] = max[i];
            this->resolution[i] = resolution[i];

            size[i] = (max[i] - min[i]) / resolution[i];
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

    __host__ void copyHistogramAsync(cudaStream_t stream = 0){
        cudaErrChk(cudaMemcpyAsync(hostPtr, cudaPtr, logicSize * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    __host__ T* getHistogram(){
        return hostPtr;
    }

    __host__ T* getHistogramCUDA(){
        return cudaPtr;
    }

    __host__ int getLogicSize(){
        return logicSize;
    }

    __host__ void getSize(int* size){
        for(int i=0; i<dim; i++){
            size[i] = this->size[i];
        }
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

private:

    __host__ void allocate(){

        cudaErrChk(cudaMallocHost((void**)&hostPtr, bufferSize * sizeof(T)));
        cudaErrChk(cudaMalloc((void**)&cudaPtr, bufferSize * sizeof(T)));
    }

    __host__ void resetBuffer(){
        cudaErrChk(cudaMemset(cudaPtr, 0, bufferSize * sizeof(T)));
    }

public:

    __host__ ~histogramCUDA(){
        cudaErrChk(cudaFreeHost(hostPtr));
        cudaErrChk(cudaFree(cudaPtr));
    }

};

}


namespace velocityHistogram{

using velocityHistogramCUDA = histogram::histogramCUDA<cudaCommonType, 2, int>;

using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaCommonType, 0, 2>;

__global__ void velocityHistogramKernel(velocitySoA* pclArray, velocityHistogramCUDA* histogramCUDAPtrUV, 
                                                        velocityHistogramCUDA* histogramCUDAPtrVW, 
                                                        velocityHistogramCUDA* histogramCUDAPtrUW);

struct getU {
    __host__ __device__
    cudaCommonType operator()(const SpeciesParticle& pcl) const {
        return pcl.get_u();
    }
};

struct getV {
    __host__ __device__
    cudaCommonType operator()(const SpeciesParticle& pcl) const {
        return pcl.get_v();
    }
};

struct getW {
    __host__ __device__
    cudaCommonType operator()(const SpeciesParticle& pcl) const {
        return pcl.get_w();
    }
};

/**
 * @brief Histogram for one species
 */
class velocityHistogram
{
private:
    // UVW
    velocityHistogramCUDA* histogramHostPtr[3];

    velocityHistogramCUDA* histogramCUDAPtr[3];

    cudaCommonType min[3], max[3], resolution[3];

    // UV, VW, UW
    cudaCommonType minArray[3][2];
    cudaCommonType maxArray[3][2];
    cudaCommonType resolutionArray[3][2];

    int cycleNum;

    std::string filePath;

    bool bigEndian;


    /**
     * @brief get the Max and Min value of the given value set
     */
    __host__ int getRange(velocitySoA* pclArray, cudaStream_t stream);

    /**
     * @brief update the histogram range and resolution, reset the buffer
     */
    __host__ int prepareHistogram(){
        
        for(int i=0; i<3; i++){
            histogramHostPtr[i]->setHistogram(minArray[i], maxArray[i], resolutionArray[i]);
            // copy the histogram object to GPU
            cudaErrChk(cudaMemcpy(histogramCUDAPtr[i], histogramHostPtr[i], sizeof(velocityHistogramCUDA), cudaMemcpyHostToDevice));
        }
        return 0;
    }
public:

    /**
     * @param initSize the initial size of the histogram buffer, in elements
     * @param path the path to store the output file, directory
     */
    __host__ velocityHistogram(int initSize, std::string path): filePath(path){
        for(int i=0; i<3; i++){
            histogramHostPtr[i] = newHostPinnedObject<velocityHistogramCUDA>(initSize);
            cudaErrChk(cudaMalloc((void**)&histogramCUDAPtr[i], sizeof(velocityHistogramCUDA)));
        }

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
    __host__ void collect(cudaStream_t stream = 0){
        cudaErrChk(cudaStreamSynchronize(stream));
        // now, all the 3 histograms data are in host buffers
        
        
        std::string items[3] = {"UV", "VW", "UW"};

        for(int i=0; i<3; i++){ // UV, VW, UW
            std::ostringstream ossFileName;
            ossFileName << filePath << "/velocityHistogram_" << MPIdata::get_rank() << "_" << items[i] << "_" << cycleNum << ".vtk";

            std::ofstream vtkFile(ossFileName.str(), std::ios::binary);

            vtkFile << "# vtk DataFile Version 3.0\n";
            vtkFile << "Velocity Histogram\n";
            vtkFile << "BINARY\n";  
            vtkFile << "DATASET STRUCTURED_POINTS\n";
            vtkFile << "DIMENSIONS " << histogramHostPtr[i]->size[0] << " " << histogramHostPtr[i]->size[1] << " 1\n";
            vtkFile << "ORIGIN " << minArray[i][0] << " " << minArray[i][1] << " 0\n"; 
            vtkFile << "SPACING " << resolutionArray[i][0] << " " << resolutionArray[i][1] << " 1\n";  
            vtkFile << "POINT_DATA " << histogramHostPtr[i]->getLogicSize() << "\n";  
            vtkFile << "SCALARS scalars int 1\n";  
            vtkFile << "LOOKUP_TABLE default\n";  

            auto histogramBuffer = histogramHostPtr[i]->getHistogram();
            for (int j = 0; j < histogramHostPtr[i]->getLogicSize(); j++) {
                int value = histogramBuffer[j];
                
                value = __builtin_bswap32(value);

                vtkFile.write(reinterpret_cast<char*>(&value), sizeof(int));
            }

            vtkFile.close();
        }


    }

    ~velocityHistogram(){
        for(int i=0; i<3; i++){
            deleteHostPinnedObject(histogramHostPtr[i]);
            cudaErrChk(cudaFree(histogramCUDAPtr[i]));
        }
    }
};





    
}






#endif