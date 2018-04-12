#pragma once
#include <glad/glad.h>
#include "glm/glm.hpp"
#include <glm/gtc/type_ptr.hpp>

#include "Drawable.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class ShaderProgram {
public:
    unsigned int ID; // GL Shader ID

    // Read in and compile the vert/frag shaders
    ShaderProgram(const GLchar* vertexPath, const GLchar* fragmentPath); // Load + compile vert/frag shaders and required OpenGL calls
    ~ShaderProgram() {}
    const void use() const { glUseProgram(ID); } // Use this ShaderProgram
    void Draw(Drawable& d);
    void setCameraViewProj(const char* uniformName, const glm::mat4& camViewProj); // Set the camera VP Matrix uniform
    void setUniformColor(const char* uniformName, const glm::vec3& color); // Set the uniform color, probably for a tree
};
