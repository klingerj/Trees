#version 450 core

layout (location = 0) in vec3 fPos;
layout (location = 1) in vec3 fNor;
out vec4 FragColor;

void main() {
    FragColor = vec4(fNor, 1);
}