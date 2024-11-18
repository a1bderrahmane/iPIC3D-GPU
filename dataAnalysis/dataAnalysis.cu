
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

using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaCommonType, 0, 3>;


/**
 * @brief analysis function for each species, uv, uw, vw
 * @details It launches 3 threads for uv uw vw analysis in parallel
 * 
 * @param outputPath the output path for the species 
 */
int GMMAnalysisSpecies(velocityHistogram::velocityHistogram* velocityHistogram, int cycle, std::string outputPath, int device){

    std::future<int> future[3];
    // static cudaGMMWeight::GMM<cudaCommonType, 2, int> gmmArray[3];

    auto GMMLambda = [=](int i) mutable {
        using namespace cudaGMMWeight;

        cudaErrChk(cudaSetDevice(device));

        // GMM config
        constexpr auto numComponent = 2;

        cudaCommonType weightVector[numComponent];
        cudaCommonType meanVector[numComponent * 2];
        cudaCommonType coVarianceMatrix[numComponent * 4];

        for(int j = 0; j < numComponent; j++){
            weightVector[j] = j == 0 ? 0.75 : 0.25;
            meanVector[j * 2] = 0.0;
            meanVector[j * 2 + 1] = 0.0;
            coVarianceMatrix[j * 4] = 0.0001;
            coVarianceMatrix[j * 4 + 1] = 0.0;
            coVarianceMatrix[j * 4 + 2] = 0.0;
            coVarianceMatrix[j * 4 + 3] = 0.0001;
        }

        GMMParam_t<cudaCommonType> GMMParam = {
            .numComponents = numComponent,
            .maxIteration = 50,
            .threshold = 1e-6,

            .weightInit = weightVector,
            .meanInit = meanVector,
            .coVarianceInit = coVarianceMatrix
        };

        // data
        GMMDataMultiDim<cudaCommonType, 2, cudaCommonType> GMMData
            (10000, velocityHistogram->getHistogramScaleMark(i), velocityHistogram->getVelocityHistogramCUDAArray(i));


        // generate exact output file path
        std::string uvw[3] = {"/uv_", "/uw_", "/vw_"};
        auto fileOutputPath = outputPath + uvw[i] + std::to_string(cycle) + ".json";

        //auto& gmm = gmmArray[i];
        cudaGMMWeight::GMM<cudaCommonType, 2, cudaCommonType> gmm;
        gmm.config(&GMMParam, &GMMData);
        auto ret =  gmm.initGMM(fileOutputPath); // the exact output file name

        return ret;
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
    static auto GMMSubDomainOutputPath = "./velocityGMM/subDomain" + std::to_string(KCode.myrank) + "/";
    static auto HistogramSubDomainOutputPath = "./velocityHistogram/subDomain" + std::to_string(KCode.myrank) + "/";

    static auto velocityHistogram = velocityHistogram::velocityHistogram(12000);

    // species by species to save VRAM
    for(int i = 0; i < KCode.ns; i++){
        // to SoA
        velocitySoA velocitySoACUDA(KCode.pclsArrayHostPtr[i], KCode.streams[i]);

        // histogram
        auto histogramSpeciesOutputPath = HistogramSubDomainOutputPath + "species" + std::to_string(i) + "/";
        velocityHistogram.init(&velocitySoACUDA, cycle, KCode.streams[i]);
        velocityHistogram.writeToFileDouble(histogramSpeciesOutputPath, KCode.streams[i]);

        // GMM
        auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "/";
        GMMAnalysisSpecies(&velocityHistogram, cycle, GMMSpeciesOutputPath, KCode.cudaDeviceOnNode);
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







