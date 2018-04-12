#version 450 core

layout (location = 0) in vec3 vsPos;
layout (location = 1) in vec3 vsNor;
layout (location = 0) out vec3 fsPos;
layout (location = 1) out vec3 fsNor;

uniform mat4 cameraViewProj;

void main() {
    fsPos = vsPos;
    fsNor = vsNor;
    gl_Position = cameraViewProj * vec4(vsPos, 1);
}