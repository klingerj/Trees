#version 450 core

layout (location = 0) in vec3 fsPos;
layout (location = 1) in vec3 fsNor;
out vec4 FragColor;

void main() {
    FragColor = vec4(fsNor, 1);
    //FragColor = vec4(0, 1, 0, 1);
}