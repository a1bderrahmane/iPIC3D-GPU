#ifndef _DATA_ANALYSIS_CONFIG_H_
#define _DATA_ANALYSIS_CONFIG_H_

#include <string>

// General configuration
inline const std::string DATA_ANALYSIS_OUTPUT_DIR = "./";
inline constexpr int DATA_ANALYSIS_EVERY_CYCLE = 50; // 0 to disable

// Histogram configuration
inline const std::string HISTOGRAM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityHistogram/";
inline constexpr bool HISTOGRAM_FIXED_RANGE = true;

// GMM configuration
inline const std::string GMM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityGMM/";






#endif