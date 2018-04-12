#version 450 core

layout (location = 0) in vec3 fsPos;
out vec4 FragColor;

void main() {
    FragColor = vec4(vec3(1.0, 0.55, 0.33), 1);
}