#pragma once
#include "glm\glm.hpp"

#include <vector>

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nor;
    // Note: UVs and colors have to be read with TinyObj, but aren't necessary to draw (for this project)
};

class Mesh {
private:
    std::vector<Vertex> vertices;

public:
    Mesh();
    ~Mesh();
    void LoadFromFile(const char* filepath);
};
