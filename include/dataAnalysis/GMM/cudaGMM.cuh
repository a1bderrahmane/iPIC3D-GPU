#ifndef _CUDA_GMM_H_
#define _CUDA_GMM_H_

#include "cudaTypeDef.cuh"
#include "cudaGMMkernel.cuh"
#include "cudaReduction.cuh"

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>


namespace cudaGMMWeight
{

template <typename T, int dataDim, typename U = int>
class GMMDataMultiDim{

private:
    int dim = dataDim;
    int numData;
    T* data[dataDim]; // pointers to the dimensions of the data points
    U* weight;
public:

    // all dim in one array
    __host__ __device__ GMMDataMultiDim(int numData, T* data, U* weight){ 
        this->numData = numData;
        for(int i = 0; i < dataDim; i++){
            this->data[i] = data + i*numData;
        }
        this->weight = weight;
    }

    // all dim in separate arrays
    __host__ __device__ GMMDataMultiDim(int numData, T** data, U* weight){
        this->numData = numData;
        for(int i = 0; i < dataDim; i++){
            this->data[i] = data[i];
        }
        this->weight = weight;
    }

    __device__ __host__ T* getDim(int dim)const {
        return data[dim];
    }

    __device__ __host__ int getNumData()const {
        return numData;
    }

    __device__ __host__ int getDim()const {
        return dim;
    }

    __device__ __host__ U* getWeight()const {
        return weight;
    }

};

template <typename T>
struct GMMParam_s{
    int numComponents;
    int maxIteration;
    T threshold; // the threshold for the log likelihood
    
    // these 3 are optional, if not set, they will be initialized with the internal init functions
    T* weightInit;
    T* meanInit;
    T* coVarianceInit;
    
};

template <typename T>
using GMMParam_t = GMMParam_s<T>;


template <typename T, int dataDim, typename U = int>
class GMM{

private:
    cudaStream_t GMMStream;

    GMMDataMultiDim<T, dataDim, U>* dataHostPtr; // ogject on host, data pointers pointing to the data on device
    GMMDataMultiDim<T, dataDim, U>* dataDevicePtr = nullptr; // object on device, data pointers pointing to the data on device

    GMMParam_t<T>* paramHostPtr; // object on host, the parameters for the GMM
    // GMMParam_t<T>* paramDevicePtr; // object on device, the parameters for the GMM

    int sizeNumComponents = 0; // the size of the buffers below, for space checking
    int sizeNumData = 0; // the size of the buffers below, for space checking

    // the arrays on device
    T* weightCUDA;                  // numComponents, it will be log(weight) during the iteration
    T* meanCUDA;                    // numComponents * dataDim
    T* coVarianceCUDA;              // numComponents * dataDim * dataDim
    T* coVarianceDecomposedCUDA;    // numComponents * dataDim * dataDim, only the lower triangle
    T* normalizerCUDA;              // numComponents
    T* PosteriorCUDA;               // numComponents, for each component, the sum of the posterior of all data points

    T* posteriorCUDA;               // numComponents * numData
    T* tempArrayCUDA;               // numComponents * numData * dataDim * dataDim, for temporary storage

    int reductionTempArraySize = 0;
    T* reductionTempArrayCUDA;      // dynamic size, for reduction, between reduction and redunctionWarp


    // the arrays on host, results and init values
    T* weight;                  // numComponents
    T* mean;                    // numComponents * dataDim
    T* coVariance;              // numComponents * dataDim * dataDim
    T* coVarianceDecomposed;    // numComponents * dataDim * dataDim, only the lower triangle
    T* normalizer;              // numComponents

    // runtime variables
    T* logLikelihoodCUDA = nullptr;
    T logLikelihood = - INFINITY;
    T logLikelihoodOld = - INFINITY;

    

public:

    //allocate or check if the arrays are allocated adequately, then replace the param
    __host__ int config(GMMParam_t<T>* GMMParam, GMMDataMultiDim<T, dataDim, U>* data){
        // check if the arrays are allocated adequately
        if(GMMParam->numComponents > sizeNumComponents){
            // deallocate the old arrays
            if(sizeNumComponents > 0){
                // device
                cudaErrChk(cudaFree(weightCUDA));
                cudaErrChk(cudaFree(meanCUDA));
                cudaErrChk(cudaFree(coVarianceCUDA));
                cudaErrChk(cudaFree(coVarianceDecomposedCUDA));
                cudaErrChk(cudaFree(normalizerCUDA));
                cudaErrChk(cudaFree(PosteriorCUDA));

                // host
                delete[] weight;
                delete[] mean;
                delete[] coVariance;
                delete[] coVarianceDecomposed;
                delete[] normalizer;

            }

            auto& numCompo = GMMParam->numComponents; 

            // allocate the new arrays
            cudaErrChk(cudaMalloc(&weightCUDA, sizeof(T)*numCompo));
            cudaErrChk(cudaMalloc(&meanCUDA, sizeof(T)*numCompo*dataDim));
            cudaErrChk(cudaMalloc(&coVarianceCUDA, sizeof(T)*numCompo*dataDim*dataDim));
            cudaErrChk(cudaMalloc(&coVarianceDecomposedCUDA, sizeof(T)*numCompo*dataDim*dataDim));
            cudaErrChk(cudaMalloc(&normalizerCUDA, sizeof(T)*numCompo));
            cudaErrChk(cudaMalloc(&PosteriorCUDA, sizeof(T)*numCompo));

            weight = new T[numCompo];
            mean = new T[numCompo*dataDim];
            coVariance = new T[numCompo*dataDim*dataDim];
            coVarianceDecomposed = new T[numCompo*dataDim*dataDim];
            normalizer = new T[numCompo];
        }

        // check the numPoint related arrays
        if(GMMParam->numComponents * data->getNumData() > sizeNumComponents * sizeNumData){
            // deallocate the old arrays
            if(sizeNumData > 0){
                // device
                cudaErrChk(cudaFree(posteriorCUDA));
                cudaErrChk(cudaFree(tempArrayCUDA));
            }

            auto numCompo = GMMParam->numComponents;
            auto numData = data->getNumData();

            // allocate the new arrays
            cudaErrChk(cudaMalloc(&posteriorCUDA, sizeof(T)*numCompo*numData));
            cudaErrChk(cudaMalloc(&tempArrayCUDA, sizeof(T)*numCompo*numData*dataDim*dataDim));
        }

        sizeNumComponents = GMMParam->numComponents;
        sizeNumData = data->getNumData();

        if(logLikelihoodCUDA == nullptr){
            cudaErrChk(cudaMalloc(&logLikelihoodCUDA, sizeof(T)));
        }


        // load the init values, if needed
        if(GMMParam->weightInit != nullptr && GMMParam->meanInit != nullptr && GMMParam->coVarianceInit != nullptr){
            memcpy(weight, GMMParam->weightInit, sizeof(T)*GMMParam->numComponents);
            memcpy(mean, GMMParam->meanInit, sizeof(T)*GMMParam->numComponents*dataDim);
            memcpy(coVariance, GMMParam->coVarianceInit, sizeof(T)*GMMParam->numComponents*dataDim*dataDim);
        }else{ // init with internal initiator
            // TODO
        }

        // precompute log of the mean vector
        for(int i = 0; i < GMMParam->numComponents; i++){
            weight[i] = log(weight[i]);
        }

        // copy to device
        cudaErrChk(cudaMemcpy(weightCUDA, weight, sizeof(T)*GMMParam->numComponents, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(meanCUDA, mean, sizeof(T)*GMMParam->numComponents*dataDim, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(coVarianceCUDA, coVariance, sizeof(T)*GMMParam->numComponents*dataDim*dataDim, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(coVarianceDecomposedCUDA, coVariance, sizeof(T)*GMMParam->numComponents*dataDim*dataDim,cudaMemcpyHostToDevice));
        // cudaErrChk(cudaMemcpy(normalizerCUDA, normalizer, sizeof(T)*GMMParam->numComponents, cudaMemcpyHostToDevice));

        // replace the param
        paramHostPtr = GMMParam;
        dataHostPtr = data;

        if(dataDevicePtr != nullptr)cudaErrChk(cudaFree(dataDevicePtr));
        dataDevicePtr = copyToDevice(dataHostPtr);


        return 0;
    }

    __host__ GMM(){
        cudaErrChk(cudaStreamCreate(&GMMStream));

        reductionTempArraySize = 1024;
        cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(T)*reductionTempArraySize));
    }

    __host__ int initGMM(std::string outputPath){

        // do the GMR
        int step = 0;
        while(step < paramHostPtr->maxIteration){
            
            // E
            calcPxAtMeanAndCoVariance();
            calcLogLikelihoodPxAndposterior();
            logLikelihood = sumLogLikelihood();

            // compare the log likelihood increament with the threshold, if the increament is smaller than the threshold, or the log likelihood is smaller than the previous one, output the GMM
            if(fabs(logLikelihood - logLikelihoodOld) < paramHostPtr->threshold || logLikelihood < logLikelihoodOld){
                // std::cout << "Converged at step " << step << std::endl;
                break;
            }
            // std::cout << "Step " << step << " log likelihood: " << logLikelihood << std::endl;
            logLikelihoodOld = logLikelihood;

            // M
            calcPosterior();
            updateMean();
            updateWeight();
            updateCoVarianceAndDecomposition();

            step++;
        }

        return outputGMM(step, outputPath);

    }

    __host__ int outputGMM(int convergeStep, std::string outputPath){
        auto numComponents = paramHostPtr->numComponents;
        auto numData = dataHostPtr->getNumData();
        // copy the results to host and post process
        cudaErrChk(cudaMemcpy(weight, weightCUDA, sizeof(T)*numComponents, cudaMemcpyDefault));
        cudaErrChk(cudaMemcpy(mean, meanCUDA, sizeof(T)*numComponents*dataDim, cudaMemcpyDefault));
        cudaErrChk(cudaMemcpy(coVariance, coVarianceCUDA, sizeof(T)*numComponents*dataDim*dataDim, cudaMemcpyDefault));
        cudaErrChk(cudaMemcpy(coVarianceDecomposed, coVarianceDecomposedCUDA, sizeof(T)*numComponents*dataDim*dataDim, cudaMemcpyDefault));
        cudaErrChk(cudaMemcpy(normalizer, normalizerCUDA, sizeof(T)*numComponents, cudaMemcpyDefault));

        for(int i = 0; i < numComponents; i++){
            weight[i] = exp(weight[i]);
        }

        {  // output the GMM to json
            std::ofstream file(outputPath);
            if (!file.is_open()) {
                std::cerr << "[!]Could not open file " << outputPath << std::endl;
                return -1;
            }

            file << "{\n";
            file << "\"convergeStep\": " << convergeStep << ",\n";
            file << "\"logLikelyHood\": " << logLikelihood << ",\n";
            file << "\"model\": {\n";
            file << "\"dataDim\": " << dataDim << ",\n";
            file << "\"numComponent\": " << numComponents << ",\n";
            file << "\"components\": [\n";

            for (size_t k = 0; k < numComponents; ++k) {
                file << "{\n";
                file << "\"weight\": " << weight[k] << ",\n";

                file << "\"mean\": [ ";
                for (size_t dim = 0; dim < dataDim; ++dim) {
                    file << mean[k * dataDim + dim];
                    if (dim + 1 < dataDim) {
                        file << ", ";
                    }
                }
                file << " ],\n";

                file << "\"coVariance\": [ ";
                for (size_t dim = 0; dim < dataDim * dataDim; ++dim) {
                    file << coVariance[k * dataDim * dataDim + dim];
                    if (dim + 1 < dataDim * dataDim) {
                        file << ", ";
                    }
                }
                file << " ]\n";
                file << "}";

                if (k + 1 < numComponents) {
                    file << ", ";
                }
            }

            file << "]\n";
            file << "}\n";
            file << "}\n";
        }

        return 0;

    }

    __host__ ~GMM(){
        // deallocate the old arrays
        if(sizeNumComponents > 0){
            // device
            cudaErrChk(cudaFree(weightCUDA));
            cudaErrChk(cudaFree(meanCUDA));
            cudaErrChk(cudaFree(coVarianceCUDA));
            cudaErrChk(cudaFree(coVarianceDecomposedCUDA));
            cudaErrChk(cudaFree(normalizerCUDA));
            cudaErrChk(cudaFree(PosteriorCUDA));

            cudaErrChk(cudaFree(posteriorCUDA));
            cudaErrChk(cudaFree(tempArrayCUDA));

            cudaErrChk(cudaFree(logLikelihoodCUDA));

            cudaErrChk(cudaFree(reductionTempArrayCUDA));

            // host
            delete[] weight;
            delete[] mean;
            delete[] coVariance;
            delete[] coVarianceDecomposed;
            delete[] normalizer;

        }
    }

private:

    int reduceBlockNum(int dataSize, int blockSize){
        if(dataSize < 4096)dataSize = 4096;
        auto blockNum = getGridSize(dataSize / 4096, blockSize); // 4096 elements per thread
        blockNum = blockNum > 1024 ? 1024 : blockNum;

        if(reductionTempArraySize < blockNum){
            cudaErrChk(cudaFree(reductionTempArrayCUDA));
            cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(T)*blockNum));
            reductionTempArraySize = blockNum;
        }

        return blockNum;
    }


    
    void calcPxAtMeanAndCoVariance(){

        // launch kernel
        cudaGMMWeightKernel::calcLogLikelihoodForPointsKernel<<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
                                (dataDevicePtr, meanCUDA, coVarianceDecomposedCUDA, posteriorCUDA, paramHostPtr->numComponents);
        
        // posterior_nk holds log p(x_i|mean,coVariance) for each data point i and each component k, temporary storage


    }




    void calcLogLikelihoodPxAndposterior(){
        // launch kernel, the first posterior_nk is the log p(x_i|mean,coVariance)
        cudaGMMWeightKernel::calcLogLikelihoodPxAndposteriorKernel<<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
                                (dataDevicePtr, weightCUDA, posteriorCUDA, tempArrayCUDA, posteriorCUDA, paramHostPtr->numComponents);
        
        // now the posterior_nk is the log posterior_nk
        // the tempArrayCUDA is the log Px for each data point 

    }

    T sumLogLikelihood(){ // its sync now, but can be async with the M step
        // sum the log likelihood Px with reduction, then return the sum
        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::none, void, U, true> // weighted sum
            <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
            (tempArrayCUDA, reductionTempArrayCUDA, dataHostPtr->getNumData(), nullptr, dataHostPtr->getWeight());
        cudaReduction::reduceSumWarp<T><<<1, 32, 0, GMMStream>>>(reductionTempArrayCUDA, logLikelihoodCUDA, blockNum);

        static T* logResult = nullptr;

        if(logResult == nullptr){
            cudaErrChk(cudaMallocHost(&logResult, sizeof(T)));
        }

        cudaErrChk(cudaMemcpyAsync(logResult, logLikelihoodCUDA, sizeof(T), cudaMemcpyDefault, GMMStream));

        cudaErrChk(cudaStreamSynchronize(GMMStream)); // The M step can be preLaunched for most of the time

        return *logResult;

    }

    void calcPosterior(){ // the Big Gamma, it can be direct sum then log

        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        auto maxValueArray = tempArrayCUDA; // maxValues of posterior_nk for each component
        for(int component = 0; component < paramHostPtr->numComponents; component++){
            // get the max value of the posterior_nk(little gamma), with reduction
            auto posteriorComponent = posteriorCUDA + component*dataHostPtr->getNumData();
            
            cudaReduction::reduceMax<T, blockSize>
                <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                (posteriorComponent, reductionTempArrayCUDA, dataHostPtr->getNumData());

            cudaReduction::reduceMaxWarp<T>
                <<<1, 32, 0, GMMStream>>>
                (reductionTempArrayCUDA, maxValueArray + component, blockNum);

            // reduction sum with pre-process and post-process to get the log Posterior_k 
            cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::minusConstThenEXP, T, U, true>
                <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                (posteriorComponent, reductionTempArrayCUDA, dataHostPtr->getNumData(), maxValueArray + component, dataHostPtr->getWeight());
            
            cudaReduction::reduceSumWarpPostProcess<T, cudaReduction::PostProcessType::logAdd, T>
                <<<1, 32, 0, GMMStream>>>
                (reductionTempArrayCUDA, PosteriorCUDA + component, blockNum, maxValueArray + component);

        }

        // now(after kernel execution) we have the log Posterior_k for each component

    }

    void updateMean(){
        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        // for each component, for each dimension
        for(int component = 0; component < paramHostPtr->numComponents; component++){
            for(int dim = 0; dim < dataHostPtr->getDim(); dim++){
                // calc x_i * posterior_nk, could be merged with the reduction sum
                // sum the x_i * posterior_nk with reduction
                cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::multiplyEXP, T, U, true>
                    <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                    (dataHostPtr->getDim(dim), reductionTempArrayCUDA, dataHostPtr->getNumData(), posteriorCUDA + component*dataHostPtr->getNumData(), dataHostPtr->getWeight());

                // divide by the Posterior_k, can be post processed with the reduction sum warp
                cudaReduction::reduceSumWarpPostProcess<T, cudaReduction::PostProcessType::divideEXP, T>
                    <<<1, 32, 0, GMMStream>>>
                    (reductionTempArrayCUDA, meanCUDA + component*dataHostPtr->getDim() + dim, blockNum, PosteriorCUDA + component);
            }
        }

    }

    void updateWeight(){
        // calc the new weight for components
        cudaGMMWeightKernel::updateWeightKernel
            <<<1, paramHostPtr->numComponents, paramHostPtr->numComponents * sizeof(T), GMMStream>>>
            (weightCUDA, PosteriorCUDA, paramHostPtr->numComponents);
    }

    void updateCoVarianceAndDecomposition(){
        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        // calc the new coVariance for components
        cudaGMMWeightKernel::updateCoVarianceKernel
            <<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
            (dataDevicePtr, posteriorCUDA, PosteriorCUDA, meanCUDA, tempArrayCUDA, paramHostPtr->numComponents);

        // sum the coVariance with reduction, then divide by the Posterior_k
        for(int component = 0; component < paramHostPtr->numComponents; component++){
            auto coVarianceComponent = tempArrayCUDA + component*dataHostPtr->getNumData()*dataDim*dataDim;
            for(int element = 0; element < dataDim * dataDim; element++){
                cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::none, void, U, true>
                    <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                    (coVarianceComponent + element*dataHostPtr->getNumData(), reductionTempArrayCUDA, dataHostPtr->getNumData(), nullptr, dataHostPtr->getWeight());
                    
                cudaReduction::reduceSumWarpPostProcess<T, cudaReduction::PostProcessType::divideEXP, T>
                    <<<1, 32, 0, GMMStream>>>
                    (reductionTempArrayCUDA, coVarianceCUDA + component*dataDim*dataDim + element, blockNum, PosteriorCUDA + component);
            }
        }

        // decompose the coVariance
        cudaGMMWeightKernel::decomposeCoVarianceKernel<T, dataDim>
            <<<1, paramHostPtr->numComponents, 0, GMMStream>>>
            (coVarianceCUDA, coVarianceDecomposedCUDA, normalizerCUDA, paramHostPtr->numComponents);
    }

};

}

#endif // _CUDA_GMM_H_



