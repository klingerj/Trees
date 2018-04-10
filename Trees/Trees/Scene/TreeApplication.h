#pragma once

#include "Globals.h"
#include "Tree.h"
#include "AttractorPointCloud.h"

class TreeApplication {
private:
    TreeParameters treeParameters;
    std::vector<Tree> sceneTrees;
    std::vector<AttractorPointCloud> sceneAttractorPointClouds;

    // App management variables
    unsigned int currentlySelectedTreeIndex;
    unsigned int currentlySelectedAttractorPointCloudIndex;

public:
    TreeApplication() : currentlySelectedTreeIndex(-1), currentlySelectedAttractorPointCloudIndex(-1) {
        treeParameters = TreeParameters();
        std::vector<Tree> sceneTrees = std::vector<Tree>();
        std::vector<AttractorPointCloud> sceneAttractorPointClouds = std::vector<AttractorPointCloud>();
    }
    void AddTreeToScene() {
        sceneTrees.emplace_back(Tree());
        currentlySelectedTreeIndex = sceneTrees.size() - 1;
    }
    void AddAttractorPointCloudToScene() {
        sceneAttractorPointClouds.emplace_back(AttractorPointCloud());
        currentlySelectedAttractorPointCloudIndex = sceneAttractorPointClouds.size() - 1;
    }
    AttractorPointCloud& GetSelectedAttractorPointCloud() {
        if (currentlySelectedAttractorPointCloudIndex != -1) {
            return sceneAttractorPointClouds[currentlySelectedAttractorPointCloudIndex];
        }
    }
    Tree& GetSelectedTree() {
        if (currentlySelectedTreeIndex != -1) {
            return sceneTrees[currentlySelectedTreeIndex];
        }
    }
    const Tree& GetSelectedTreeConst() const {
        if (currentlySelectedTreeIndex != -1) {
            return sceneTrees[currentlySelectedTreeIndex];
        }
    }
    void GrowSelectedTreeIntoSelectedAttractorPointCloud() {
        if (currentlySelectedTreeIndex != -1 && currentlySelectedAttractorPointCloudIndex != -1) {
            #ifdef ENABLE_DEBUG_OUTPUT
            auto start = std::chrono::system_clock::now();
            #endif
            sceneTrees[currentlySelectedTreeIndex].IterateGrowth(sceneAttractorPointClouds[currentlySelectedAttractorPointCloudIndex].GetPointsCopy(), treeParameters, true);
            #ifdef ENABLE_DEBUG_OUTPUT
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::time_t end_time = std::chrono::system_clock::to_time_t(end);
            std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";
            #endif
        }
    }
    TreeParameters& GetTreeParameters() { return treeParameters; }
    const TreeParameters& GetTreeParametersConst() const { return treeParameters; }
};
