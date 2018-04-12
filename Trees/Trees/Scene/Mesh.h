#pragma once
#include <vector>
#include "glm/glm.hpp"
#include "../Raytracing/Raytracing.h"
#include "../OpenGL/Drawable.h"

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
class Mesh : public Drawable {
private:
    // Lists of vertices and indices
    std::vector<Triangle> triangles;
    std::vector<Vertex> vertices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;
public:
    Mesh() {
        triangles = std::vector<Triangle>();
        vertices = std::vector<Vertex>();
        positions = std::vector<glm::vec3>();
        normals = std::vector<glm::vec3>();
        indices = std::vector<unsigned int>();
    }
    void LoadFromFile(const char* filepath);

    // Getters
    const std::vector<Triangle>&     GetTriangles() const { return triangles; }
    const std::vector<Vertex>&       GetVertices()  const { return vertices; }
    const std::vector<glm::vec3>&    GetPositions() const { return positions; }
    const std::vector<glm::vec3>&    GetNormals()   const { return normals; }
    const std::vector<unsigned int>& GetIndices()   const { return indices; }

    // Setters
    void SetPositions(std::vector<glm::vec3>& p) { positions = p; }
    void SetNormals(std::vector<glm::vec3>& n) { normals = n; }
    void SetIndices(std::vector<unsigned int>& i) { indices = i; }

    // Raytracing functions
    Intersection Intersect(const Ray& r) const; // Intersect a single ray with this mesh
    bool Contains(const glm::vec3& p) const; // Check if a point intersects this mesh an odd number of times

    // Mesh manipulation
    void AddPositions(const std::vector<glm::vec3>& p) {
        positions.insert(positions.end(), p.begin(), p.end());
    }
    void AddNormals(const std::vector<glm::vec3>& n) {
        normals.insert(normals.end(), n.begin(), n.end());
    }
    void AddIndices(const std::vector<unsigned int>& i) {
        indices.insert(indices.end(), i.begin(), i.end());
    }

    // Inherited Function(s)
    void create() override;
    GLenum drawMode() override { return GL_TRIANGLES; }
};
