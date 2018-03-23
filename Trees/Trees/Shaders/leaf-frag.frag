#version 450 core

layout (location = 0) in vec3 fPos;
layout (location = 1) in vec3 fNor;
out vec4 FragColor;

void main() {
    float lambert = dot(normalize(vec3(1, 1, 1)), fNor);
    //FragColor = vec4(1, 0, 0, 1);
    FragColor = vec4(vec3(0.2, 0.4, 0.2) * abs(lambert), 1); // want this for actual 3d geometry
    //FragColor = vec4(vec3(0, 1, 0), 1);
    //FragColor = vec4(abs(fNor), 1);
}