#pragma once
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "Globals.h"
#include <iostream>

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
    int viewportHeight;
    int viewportWidth;
public:
    Camera(const glm::vec3& r, const float fov, const float a, const float n, const float f, const float t, const float p, const float ra, const int vph, const int vpw) :
        ref(r), fovy(fov), aspect(a), nearPlane(n), farPlane(f), theta(t), phi(p), radius(ra), viewportHeight(vph), viewportWidth(vpw) { UpdateEye(); }
    ~Camera() {}
    // Set new aspect ratio (e.x. when window is resized
    float GetAspect() const { return aspect; }
    void SetAspect(const float a) { aspect = a; }
    void SetViewportHeight(const int h) { viewportHeight = h; }
    void SetViewportWidth(const int w) { viewportWidth = w; }
    int GetViewportHeight() const { return viewportHeight; }
    int GetViewportWidth() const { return viewportWidth; }
    float GetFovy() const { return fovy; }
    glm::vec3 GetEye() const { return eye; }
    // Return the View-Projection (VP) Matrix
    const glm::mat4 GetViewProj() const { return glm::perspective(fovy, aspect, nearPlane, farPlane) * glm::lookAt(eye, ref, WORLD_UP_VECTOR); }
    const glm::mat4 GetView() const { return glm::lookAt(eye, ref, WORLD_UP_VECTOR); }
    const glm::mat4 GetProj() const { return glm::perspective(fovy, aspect, nearPlane, farPlane); }
    const float GetFarPlane() const { return farPlane; }
    void UpdateEye() {
        eye = glm::vec3(glm::rotate(glm::mat4(1.0f), theta, glm::vec3(0.0f, 1.0f, 0.0f)) * 
                        glm::rotate(glm::mat4(1.0f), phi, glm::vec3(1.0f, 0.0f, 0.0f)) * glm::vec4(0.0f, 0.0f, radius, 1.0f));
    }
    void TranslateAlongRadius(const float amt) { radius += amt; UpdateEye(); }
    void RotateTheta(const float t) { theta += t;  UpdateEye(); }
    void RotatePhi(const float p) { phi += p;  UpdateEye(); }
    void TranslateRefAlongWorldY(const float t) { ref.y += t; }
};
