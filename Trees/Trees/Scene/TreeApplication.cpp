#include "TreeApplication.h"

void TreeApplication::IterateSelectedTreeInSelectedAttractorPointCloud() {
    if (currentlySelectedTreeIndex != -1 && currentlySelectedAttractorPointCloudIndex != -1) {
        auto start = std::chrono::system_clock::now();
        /*#ifdef ENABLE_DEBUG_OUTPUT
        auto start = std::chrono::system_clock::now();
        #endif*/
        bool prevState = treeParameters.resetAttractorPointState;
        treeParameters.resetAttractorPointState = false;
        AttractorPointCloud& currentAttrPtCloud = sceneAttractorPointClouds[currentlySelectedAttractorPointCloudIndex];
        sceneTrees[currentlySelectedTreeIndex].IterateGrowth(currentAttrPtCloud.GetPointsCopy(), currentAttrPtCloud.GetMinPoint(), currentAttrPtCloud.GetMaxPoint(), treeParameters, true);
        treeParameters.resetAttractorPointState = prevState;
        /*#ifdef ENABLE_DEBUG_OUTPUT
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";
        #endif*/
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";

        sceneTrees[currentlySelectedTreeIndex].create();
    }
}

void TreeApplication::RegrowSelectedTreeInSelectedAttractorPointCloud() {
    if (currentlySelectedTreeIndex != -1 && currentlySelectedAttractorPointCloudIndex != -1) {
        auto start = std::chrono::system_clock::now();
        /*#ifdef ENABLE_DEBUG_OUTPUT
        auto start = std::chrono::system_clock::now();
        #endif*/
        Tree& currentTree = sceneTrees[currentlySelectedTreeIndex];
        currentTree.ResetTree();
        AttractorPointCloud& currentAttrPtCloud = sceneAttractorPointClouds[currentlySelectedAttractorPointCloudIndex];
        treeParameters.resetAttractorPointState = true;
        currentTree.IterateGrowth(currentAttrPtCloud.GetPointsCopy(), currentAttrPtCloud.GetMinPoint(), currentAttrPtCloud.GetMaxPoint(), treeParameters, true);
        /*#ifdef ENABLE_DEBUG_OUTPUT
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";
        #endif*/
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";
        sceneTrees[currentlySelectedTreeIndex].create();
    }
}

void TreeApplication::ComputeWorldSpaceSketchPoints(const Camera& camera) {
    const glm::mat4 viewMat = camera.GetView();
    const glm::mat4 invViewMat = glm::inverse(viewMat);
    const glm::mat4 invProjMat = glm::inverse(camera.GetProj());

    const glm::vec3& rootBudPoint = sceneTrees[currentlySelectedTreeIndex].branches[0].GetBuds()[0].point;
    const glm::vec3 budPointView = (viewMat * glm::vec4(rootBudPoint, 1.0f));
    const float tanFovy = std::tan(camera.GetFovy() * 0.5f);
    const float len = glm::length(rootBudPoint - camera.GetEye());
    const glm::vec3 refPoint = rootBudPoint;//glm::vec3(camera.GetEye() + glm::vec3(viewMat[2]));

    for (unsigned int sp = 0; sp < (unsigned int)currentSketchPoints.size(); ++sp) {
        glm::vec3& currentSketchPoint = currentSketchPoints[sp];
        // do math similar to raycasting - want the position position in the frustum using len = |viewSpaceBudPoint - eye|
        const glm::vec3 V = tanFovy * len * glm::vec3(viewMat[1]);
        const glm::vec3 H = tanFovy * len * camera.GetAspect() * glm::vec3(viewMat[0]);
        //currentSketchPoint = glm::vec3(invViewMat * glm::vec4(currentSketchPoint, 1.0f));
        currentSketchPoint = refPoint + V * currentSketchPoint.y + H * currentSketchPoint.x;

        /*currentSketchPoint = glm::vec3(invProjMat * glm::vec4(currentSketchPoint, 1.0f)); // From ndc to view space
        //currentSketchPoint.z = budViewZ; // set view space z value
        //currentSketchPoint = glm::vec3(invViewMat * glm::vec4(currentSketchPoint, 1.0f)); // view to world space*/
    }
}

// Generates an attractor point cloud from the given sketch points and adds it to the scene
void TreeApplication::GenerateSketchAttractorPointCloud() {
    AddAttractorPointCloudToScene();
    GetSelectedAttractorPointCloud().GeneratePointsGivenSketchPoints(treeParameters.numAttractorPointsToGenerate, currentSketchPoints, treeParameters.brushRadius);
}