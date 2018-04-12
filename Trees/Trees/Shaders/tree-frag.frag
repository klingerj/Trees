#version 450 core

layout (location = 0) in vec3 fsPos;
layout (location = 1) in vec3 fsNor;
out vec4 FragColor;

uniform vec3 u_color;

void main() {
    float lambert = dot(normalize(vec3(1, 1, 1)), fsNor);
    FragColor = vec4(u_color * abs(lambert), 1); // want this for actual 3d geometry
}