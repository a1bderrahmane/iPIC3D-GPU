
#include <thread>
#include <future>
#include "iPic3D.h"
#include "dataAnalysis.cuh"

#include "GMM/cudaGMM.cuh"
#include "particleArraySoACUDA.cuh"
#include "velocityHistogram.cuh"

#include <string>


namespace dataAnalysis
{

using namespace iPic3D;

using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaCommonType, 0, 2>;


/**
 * @brief analysis function for each species, uv, uw, vw
 * @details It launches 3 threads for uv uw vw analysis in parallel
 * 
 * @param outputPath the output path for the species 
 */
int GMMAnalysisSpecies(velocitySoA* velocitySoACUDAPtr, int cycle, std::string outputPath, int device){

    
    using namespace particleArraySoA;

    cudaCommonType* uvwPtr[6] = {
                                velocitySoACUDAPtr->getElement(U), velocitySoACUDAPtr->getElement(V),
                                velocitySoACUDAPtr->getElement(U), velocitySoACUDAPtr->getElement(W),
                                velocitySoACUDAPtr->getElement(V), velocitySoACUDAPtr->getElement(W)
                                };

    std::future<int> future[3];

    auto GMMLambda = [=](int i) mutable {
        using namespace cudaGMM;

        cudaErrChk(cudaSetDevice(device));

        constexpr auto numComponent = 1;

        cudaCommonType weightVector[numComponent];
        cudaCommonType meanVector[numComponent * 2];
        cudaCommonType coVarianceMatrix[numComponent * 4];

        for(int j = 0; j < numComponent; j++){
            weightVector[j] = 1.0 / numComponent;
            meanVector[j * 2] = 0.0;
            meanVector[j * 2 + 1] = 0.0;
            coVarianceMatrix[j * 4] = 1.0;
            coVarianceMatrix[j * 4 + 1] = 0.0;
            coVarianceMatrix[j * 4 + 2] = 0.0;
            coVarianceMatrix[j * 4 + 3] = 1.0;
        }

        GMMParam_t<cudaCommonType> GMMParam = {
            .numComponents = numComponent,
            .maxIteration = 100,
            .threshold = 1e-6,

            .weightInit = weightVector,
            .meanInit = meanVector,
            .coVarianceInit = coVarianceMatrix
        };

        GMMDataMultiDim<cudaCommonType, 2> GMMData(velocitySoACUDAPtr->getNOP(), &uvwPtr[i * 2]);

        // generate exact output file path
        std::string uvw[3] = {"/uv_", "/uw_", "/vw_"};
        auto fileOutputPath = outputPath + uvw[i] + std::to_string(cycle) + ".json";

        GMM<cudaCommonType, 2> gmm;
        gmm.config(&GMMParam, &GMMData);
        return gmm.initGMM(fileOutputPath); // the exact output file name
    };

    for(int i = 0; i < 3; i++){
        // launch 3 async threads for uv, uw, vw
        future[i] = std::async(std::launch::async, GMMLambda, i);
    }

    for(int i = 0; i < 3; i++){
        future[i].wait();
    }

    return 0;
}

/**
 * @brief analysis function, called by startAnalysis
 * @details procesures in this function should be executed in sequence, the order of the analysis should be defined here
 *          But the procedures can launch other threads to do the analysis
 *          Also this function is a friend function of c_Solver, resources in the c_Slover should be dispatched here
 */
int analysisEntre(c_Solver& KCode, int cycle){
    cudaErrChk(cudaSetDevice(KCode.cudaDeviceOnNode));

    // ./velocityGMM/subDomain0/species0/uv_1000.json , like this
    auto GMMSubDomainOutputPath = "./velocityGMM/subDomain" + std::to_string(KCode.myrank) + "/";
    auto HistogramSubDomainOutputPath = "./velocityHistogram/subDomain" + std::to_string(KCode.myrank) + "/";

    // species by species to save VRAM
    for(int i = 0; i < KCode.ns; i++){
        // to SoA
        velocitySoA velocitySoACUDA(KCode.pclsArrayHostPtr[i], KCode.streams[i]);

        // histogram
        auto histogramSpeciesOutputPath = HistogramSubDomainOutputPath + "species" + std::to_string(i) + "/";
        auto velocityHistogram = velocityHistogram::velocityHistogram(12000, histogramSpeciesOutputPath);
        velocityHistogram.init(&velocitySoACUDA, cycle, KCode.streams[i]);
        velocityHistogram.collect(KCode.streams[i]);

        // GMM
        auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "/";
        GMMAnalysisSpecies(&velocitySoACUDA, cycle, GMMSpeciesOutputPath, KCode.cudaDeviceOnNode);
    }

    return 0;
}


/**
 * @brief start all the analysis registered here
 */
std::future<int> startAnalysis(c_Solver& KCode, int cycle){

    if(cycle % 50 != 0){
        return std::future<int>();
    }

    std::future<int> analysisFuture = std::async(analysisEntre, std::ref(KCode), cycle);

    return analysisFuture;
}

/**
 * @brief check if the analysis is done, non-blocking
 * 
 * @return 0 if the analysis is done, 1 if it is not done
 */
int checkAnalysis(std::future<int>& analysisFuture){

    if(analysisFuture.valid() == false){
        return 0;
    }

    if(analysisFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
        return 0;
    }else{
        return 1;
    }

    return 0;
}

/**
 * @brief wait for the analysis to be done, blocking
 */
int waitForAnalysis(std::future<int>& analysisFuture){

    if(analysisFuture.valid() == false){
        return 0;
    }

    analysisFuture.wait();

    return 0;
}


    
} // namespace dataAnalysis







