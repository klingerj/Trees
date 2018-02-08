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
    // Note: Given the way TinyObj presents the vertex information, the indices list just ends up being a
    // list of the numbers [0, 1, 2, ..., n] for n vertices. The vertices are already arranged in the proper order.
    std::vector<Vertex> vertices; 
    std::vector<unsigned int> indices;
public:
    Mesh();
    ~Mesh();
    void LoadFromFile(const char* filepath);
    inline const std::vector<Vertex>& GetVertices() const { return vertices; }
    inline const std::vector<unsigned int>& GetIndices() const { return indices; }
};
