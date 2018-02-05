#pragma once
#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class ShaderProgram {
public:
    unsigned int ID;

    ShaderProgram(const GLchar* vertexPath, const GLchar* fragmentPath); // Load + compile vert/frag shaders and required OpenGL calls
    ~ShaderProgram() {}
    void use(); // Use this ShaderProgram
};

