#pragma once

#include "Tree.h"
#include "AttractorPointCloud.h"

class TreeApplication {
private:
    TreeParameters treeParameters;
    std::vector<Tree> sceneTrees;
    std::vector<AttractorPointCloud> sceneAttractorPointsCloud;
public:
    TreeApplication() {
        treeParameters = TreeParameters();
        std::vector<Tree> sceneTrees = std::vector<Tree>();
        std::vector<AttractorPointCloud> sceneAttractorPointsCloud = std::vector<AttractorPointCloud>();
    }
    inline TreeParameters& GetTreeParameters() { return treeParameters; }
};
