#version 450 core

layout (location = 0) in vec3 vsPos;
layout (location = 0) out vec3 fsPos;

uniform mat4 cameraViewProj;

void main() {
    fsPos = vsPos;
    gl_Position = cameraViewProj * vec4(vsPos, 1);
}