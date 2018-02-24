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
    float theta; // Uses polar camera model. Theta and phi parameterize the sphere.
    float phi;
    float radius;
public:
    Camera(const glm::vec3& r, const float fov, const float a, const float n, const float f, const float t, const float p, const float ra) :
        ref(r), fovy(fov), aspect(a), nearPlane(n), farPlane(f), theta(t), phi(p), radius(ra) { UpdateEye(); }
    ~Camera() {}
    // Set new aspect ratio (e.x. when window is resized
    inline void SetAspect(const float a) { aspect = a; }
    // Return the View-Projection (VP) Matrix
    inline const glm::mat4 GetViewProj() const { return glm::perspective(fovy, aspect, nearPlane, farPlane) * glm::lookAt(eye, ref, WORLD_UP_VECTOR); }
    inline void UpdateEye() { eye = glm::vec3(glm::rotate(glm::mat4(1.0f), theta, glm::vec3(0.0f, 1.0f, 0.0f)) *
                                              glm::rotate(glm::mat4(1.0f), phi, glm::vec3(1.0f, 0.0f, 0.0f)) * glm::vec4(0.0f, 0.0f, radius, 1.0f)); }
    inline void TranslateAlongRadius(const float amt) { radius += amt; UpdateEye(); }
    inline void RotateTheta(const float t) { theta += t;  UpdateEye(); }
    inline void RotatePhi(const float p) { phi += p;  UpdateEye(); }
};
