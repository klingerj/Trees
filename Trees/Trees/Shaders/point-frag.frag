#version 450 core

layout (location = 0) in vec3 fPos;
out vec4 FragColor;

void main() {
    FragColor = vec4(vec3(0.8, 0.9, 0), 1);
}