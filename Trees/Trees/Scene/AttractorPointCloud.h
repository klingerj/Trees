#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
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
    bool removed;

    AttractorPoint() : AttractorPoint(glm::vec3(0.0f)) {}
    AttractorPoint(const glm::vec3& p) : point(p), nearestBudDist2(9999999.0f), nearestBudBranchIdx(-1), nearestBudIdx(-1), removed(false) {}
};

class AttractorPointCloud : public Drawable {
protected:
    
private:
    std::vector<AttractorPoint> points;
    glm::vec3 minPoint;
    glm::vec3 maxPoint;
    pcg32 rng;
    std::uniform_real_distribution<float> dis;
    Mesh boundingMesh;
    bool shouldDisplay;
public:
    AttractorPointCloud() : shouldDisplay(true), minPoint(glm::vec3(999999.0f)), maxPoint(glm::vec3(-999999.0f)) {
        points = std::vector<AttractorPoint>();
        rng(101); // Any seed
        dis = std::uniform_real_distribution<float>(-1.0f, 1.0f);
        boundingMesh = Mesh();
    }
    bool ShouldDisplay() const { return shouldDisplay && points.size() > 0; }
    void ToggleDisplay() { shouldDisplay = !shouldDisplay; }
    const std::vector<AttractorPoint>& GetPointsConst() const { return points; }
    std::vector<AttractorPoint>& GetPoints() { return points; }
    std::vector<AttractorPoint> GetPointsCopy() const { return points; }
    glm::vec3& GetMinPoint() { return minPoint; }
    glm::vec3& GetMaxPoint() { return maxPoint; }
    void GeneratePointsInUnitCube(unsigned int numPoints);
    void GeneratePoints(unsigned int numPoints);
    void GeneratePoints(unsigned int numPoints, const std::vector<glm::vec3>& sketchPoints, const float brushRadius);
    void AddPoints(const std::vector<AttractorPoint>& p) { points.insert(points.begin(), p.begin(), p.end()); }
    static AttractorPointCloud UnionAttractorPointClouds(const AttractorPointCloud& ap1, const AttractorPointCloud& ap2) {
        AttractorPointCloud unionCloud;
        unionCloud.AddPoints(ap1.points);
        unionCloud.AddPoints(ap2.points);
        return unionCloud;
    }

    // Inherited functions from Drawable
    void create() override;
    GLenum drawMode() override { return GL_POINTS; }
};
