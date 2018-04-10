#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "pcg_random.hpp"
#include <chrono>
#include <ctime>
#include <random>

#include "../OpenGL/Drawable.h"
#include "Mesh.h"

struct AttractorPoint {
    glm::vec3 point; // Point in world space
    float nearestBudDist2; // how close the nearest bud is that has this point in its perception volume, squared
    int nearestBudBranchIdx; // index in the array of the branch of that bud ^^
    int nearestBudIdx; // index in the array of the bud of that branch ^^

    AttractorPoint(const glm::vec3& p) : point(p), nearestBudDist2(9999999.0f), nearestBudBranchIdx(-1), nearestBudIdx(-1) {}
};

class AttractorPointCloud : public Drawable {
protected:
    std::vector<AttractorPoint> points;
private:
    pcg32 rng;
    std::uniform_real_distribution<float> dis;
public:
    AttractorPointCloud() {
        points = std::vector<AttractorPoint>();
        rng(101); // Any seed
        dis = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }
    const std::vector<AttractorPoint>& GetPoints() { return points; }
    std::vector<AttractorPoint> GetPointsCopy() { return points; }
    void GeneratePoints(unsigned int numPoints);
    void GeneratePoints(const Mesh& m, unsigned int numPoints);
    void AddPoints(const std::vector<AttractorPoint>& p) {
        points.insert(points.begin(), p.begin(), p.end());
    }
    static AttractorPointCloud UnionAttractorPointClouds(const AttractorPointCloud& ap1, const AttractorPointCloud& ap2) {
        AttractorPointCloud unionCloud;
        unionCloud.AddPoints(ap1.points);
        unionCloud.AddPoints(ap2.points);
        return unionCloud;
    }

    // Inherited functions from Drawable
    void create() override;
    GLenum drawMode() { return GL_POINTS; }
};
