#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#define WORLD_UP_VECTOR glm::vec3(0.0f, 1.0f, 0.0f)

class Camera {
private:
    glm::vec3 eye; // world space eye position
    glm::vec3 ref; // world space ref position
    float fovy; // field of view in radians
    float nearPlane; // near clip plane
    float farPlane; // far clip plane
    float aspect; // aspect ratio
public:
    Camera(const glm::vec3& e, const glm::vec3& r, const float fov, const float a, const float n, const float f) : eye(e), ref(r), fovy(fov), aspect(a), nearPlane(n), farPlane(f) {}
    ~Camera() {}
    // Set new aspect ratio (e.x. when window is resized
    inline void SetAspect(const float a) { aspect = a; }
    // Return the View-Projection (VP) Matrix
    inline const glm::mat4 GetViewProj() const { return glm::perspective(fovy, aspect, nearPlane, farPlane) * glm::lookAt(eye, ref, WORLD_UP_VECTOR); }
};
