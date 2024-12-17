#ifndef _DATA_ANALYSIS_CUH_
#define _DATA_ANALYSIS_CUH_

#include <thread>
#include <future>
#include "iPic3D.h"

namespace dataAnalysis
{

void createOutputDirectory(int myrank, int ns, VirtualTopology3D* vct);

std::future<int> startAnalysis(iPic3D::c_Solver& KCode, int cycle);

int checkAnalysis(std::future<int>& analysisFuture);

int waitForAnalysis(std::future<int>& analysisFuture);


    
} // namespace dataAnalysis






#endif





