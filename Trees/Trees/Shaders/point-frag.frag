#version 460 core

layout (location = 0) in vec3 fPos;
out vec4 FragColor;

void main() {
    FragColor = vec4(fPos, 1);
}