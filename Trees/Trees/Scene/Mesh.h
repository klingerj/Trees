#pragma once
#include "glm\glm.hpp"

#include <vector>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    // Note: UVs and colors aren't necessary to draw (for this project)
};

class Mesh {
private:
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

public:
    Mesh();
    ~Mesh();
    void LoadFromFile(const char* filepath);
    inline const std::vector<Vertex>& GetVertices() const {
        return vertices;
    }
    inline const std::vector<unsigned int>& GetIndices() const {
        return indices;
    }
};
