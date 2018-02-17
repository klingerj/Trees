#version 450 core

layout (location = 0) in vec3 vPos;
layout (location = 1) in vec3 vNor;
layout (location = 0) out vec3 fPos;
layout (location = 1) out vec3 fNor;

uniform mat4 cameraViewProj;

void main() {
    fPos = vPos;
    fNor = normalize(vNor);
    gl_Position = cameraViewProj * vec4(vPos, 1);
}