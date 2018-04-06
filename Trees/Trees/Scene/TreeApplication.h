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

    }
};
