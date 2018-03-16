#pragma once
#include "glm/glm.hpp"
#include "../Raytracing/Raytracing.h"

#include <vector>

// Note: UVs and colors aren't necessary to draw (for this project)
struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    Vertex() : pos(glm::vec3(0.f)), nor(glm::vec3(0.f)) {}
};

class Triangle {
private:
    glm::vec3 planeNormal;
    std::vector<glm::vec3> points;
public:
    Triangle() {
        points = std::vector<glm::vec3>();
        points.reserve(3);
    }
    void AppendVertex(const glm::vec3& p) {
        points.emplace_back(p);
    }
    Intersection Intersect(const Ray& r) const;
    inline void ComputePlaneNormal() { planeNormal = glm::normalize(glm::cross(points[1] - points[0], points[2] - points[1])); }
};

// TODO: refactor, remove the excess storage of vertices/indices
// TODO: currently assumes a triangulated mesh. No triangulation occurs here right now.
class Mesh {
private:
    // Lists of vertices and indices
    // Note: Given the way TinyObj presents the vertex information, the indices list just ends up being a
    // list of the numbers [0, 1, 2, ..., n] for n vertices. The vertices are already arranged in the proper order.
    std::vector<Vertex> vertices; 
    std::vector<unsigned int> indices;
    std::vector<Triangle> triangles;
public:
    Mesh();
    ~Mesh();
    void LoadFromFile(const char* filepath);
    inline const std::vector<Vertex>& GetVertices() const { return vertices; }
    inline const std::vector<unsigned int>& GetIndices() const { return indices; }
    Intersection Intersect(const Ray& r) const; // Intersect a single ray with this mesh
    const bool Contains(const glm::vec3& p) const; // Check if a point intersects this mesh an odd number of times
};
