#include "TreeApplication.h"

void TreeApplication::IterateSelectedTreeInSelectedAttractorPointCloud() {
    if (currentlySelectedTreeIndex != -1 && currentlySelectedAttractorPointCloudIndex != -1) {
        #ifdef ENABLE_DEBUG_OUTPUT
        auto start = std::chrono::system_clock::now();
        #endif
        AttractorPointCloud& currentAttrPtCloud = sceneAttractorPointClouds[currentlySelectedAttractorPointCloudIndex];
        sceneTrees[currentlySelectedTreeIndex].IterateGrowth(currentAttrPtCloud.GetPointsCopy(), currentAttrPtCloud.GetMinPoint(), currentAttrPtCloud.GetMaxPoint(), treeParameters, true);
        #ifdef ENABLE_DEBUG_OUTPUT
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";
        #endif
        sceneTrees[currentlySelectedTreeIndex].create();
    }
}

void TreeApplication::RegrowSelectedTreeInSelectedAttractorPointCloud() {
    if (currentlySelectedTreeIndex != -1 && currentlySelectedAttractorPointCloudIndex != -1) {
        #ifdef ENABLE_DEBUG_OUTPUT
        auto start = std::chrono::system_clock::now();
        #endif
        Tree& currentTree = sceneTrees[currentlySelectedTreeIndex];
        AttractorPointCloud& currentAttrPtCloud = sceneAttractorPointClouds[currentlySelectedAttractorPointCloudIndex];
        currentTree.ResetTree();
        currentTree.IterateGrowth(currentAttrPtCloud.GetPointsCopy(), currentAttrPtCloud.GetMinPoint(), currentAttrPtCloud.GetMaxPoint(), treeParameters, true);
        #ifdef ENABLE_DEBUG_OUTPUT
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";
        #endif
        sceneTrees[currentlySelectedTreeIndex].create();
    }
}