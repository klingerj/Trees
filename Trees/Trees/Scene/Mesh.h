#pragma once
#include "glm/glm.hpp"

#include <vector>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    // Note: UVs and colors aren't necessary to draw (for this project)
};

class Mesh {
private:
    // Lists of vertices and indices
    std::vector<Vertex> vertices;
    /*std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;*/
    std::vector<unsigned int> indices;
public:
    Mesh();
    ~Mesh();
    void LoadFromFile(const char* filepath);
    inline const std::vector<Vertex>& GetVertices() const { return vertices; }
    /*inline const std::vector<glm::vec3>& GetPositions() const { return positions; }
    inline const std::vector<glm::vec3>& GetNormals() const { return normals; }*/
    inline const std::vector<unsigned int>& GetIndices() const { return indices; }
};
