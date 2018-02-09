#version 450 core

layout (location = 0) in vec3 fPos;
layout (location = 1) in vec3 fNor;
out vec4 FragColor;

void main() {
    float lambert = dot(normalize(vec3(1, 1, -2)), fNor);
    FragColor = vec4(vec3(lambert), 1);
}