#version 450 core

layout (location = 0) in vec3 vPos;
layout (location = 0) out vec3 fPos;

void main() {
    fPos = vPos;
    gl_Position = vec4(vPos * vec3(600.0 / 800.0, 1, 1), 1); // account for aspect ratio...why height/width though?
}