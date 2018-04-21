#pragma once

#include "Globals.h"
#include "../OpenGL/ShaderProgram.h"
#include "Tree.h"
#include "AttractorPointCloud.h"

class TreeApplication {
private:
    TreeParameters treeParameters;
    std::vector<Tree> sceneTrees; // trees in the scene
    std::vector<AttractorPointCloud> sceneAttractorPointClouds; // attractor point clouds in the scene
    std::vector<glm::vec2> currentSketchPoints; // the sketch points in screen space of the current sketch stroke

    // App management variables
    int currentlySelectedTreeIndex;
    int currentlySelectedAttractorPointCloudIndex;

public:
    TreeApplication() : currentlySelectedTreeIndex(-1), currentlySelectedAttractorPointCloudIndex(-1) {
        treeParameters = TreeParameters();
        std::vector<Tree> sceneTrees = std::vector<Tree>();
        std::vector<AttractorPointCloud> sceneAttractorPointClouds = std::vector<AttractorPointCloud>();
    }

    void DestroyTrees() {
        for (unsigned int t = 0; t < (unsigned int)sceneTrees.size(); ++t) {
            sceneTrees[t].DestroyMeshes();
        }
    }
    void DestroyAttractorPointClouds() {
        for (unsigned int ap = 0; ap < (unsigned int)sceneAttractorPointClouds.size(); ++ap) {
            sceneAttractorPointClouds[ap].destroy();
        }
    }

    // Scene Edition Functions
    void AddTreeToScene() {
        sceneTrees.emplace_back(Tree());
        currentlySelectedTreeIndex = (int)(sceneTrees.size()) - 1;
    }
    void AddAttractorPointCloudToScene() {
        sceneAttractorPointClouds.emplace_back(AttractorPointCloud());
        currentlySelectedAttractorPointCloudIndex = (int)(sceneAttractorPointClouds.size()) - 1;
    }

    AttractorPointCloud& GetSelectedAttractorPointCloud() {
        if (currentlySelectedAttractorPointCloudIndex != -1) {
            return sceneAttractorPointClouds[currentlySelectedAttractorPointCloudIndex];
        }
        return AttractorPointCloud(); // a bad temporary attractor point cloud! TODO replace this functionality
    }
    Tree& GetSelectedTree() {
        if (currentlySelectedTreeIndex != -1) {
            return sceneTrees[currentlySelectedTreeIndex];
        }
        return Tree(); // a bad temporary tree!
    }
    const Tree& GetSelectedTreeConst() const {
        if (currentlySelectedTreeIndex != -1) {
            return sceneTrees[currentlySelectedTreeIndex];
        }
        return Tree(); // a bad temporary tree!
    }

    void IterateSelectedTreeInSelectedAttractorPointCloud();
    void RegrowSelectedTreeInSelectedAttractorPointCloud();


    TreeParameters& GetTreeParameters() { return treeParameters; }
    const TreeParameters& GetTreeParametersConst() const { return treeParameters; }

    // Functions for drawing the scene
    void DrawAttractorPointClouds(ShaderProgram& sp) {
        for (unsigned int ap = 0; ap < (unsigned int)sceneAttractorPointClouds.size(); ++ap) {
            AttractorPointCloud& currentAPC = sceneAttractorPointClouds[ap];
            if (currentAPC.ShouldDisplay()) {
                sp.Draw(sceneAttractorPointClouds[ap]);
            }
        }
    }
    void DrawTrees(ShaderProgram& sp) {
        for (unsigned int t = 0; t < (unsigned int)sceneTrees.size(); ++t) {
            Tree& currentTree = sceneTrees[t];
            if (currentTree.HasBeenCreated()) {
                sp.setUniformColor("u_color", currentTree.GetBranchColor());
                sp.Draw(currentTree.GetTreeMesh());
                sp.setUniformColor("u_color", currentTree.GetLeafColor());
                sp.Draw(currentTree.GetLeavesMesh());
            }
        }
    }
};
