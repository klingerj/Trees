#version 450 core

layout (location = 0) in vec3 fPos;
out vec4 FragColor;

void main() {
    FragColor = vec4(vec3(1.0, 0.66, 0.33), 1);
}