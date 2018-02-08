#version 450 core

layout (location = 0) in vec3 vPos;
layout (location = 0) out vec3 fPos;

uniform mat4 cameraViewProj;

void main() {
    fPos = vPos;
    gl_Position = cameraViewProj * vec4(vPos, 1);
}