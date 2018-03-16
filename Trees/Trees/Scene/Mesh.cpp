#include "Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION // Define once in a cc/cpp file
#include "tiny_obj_loader.h"

#include <iostream>

Mesh::Mesh() {}

Mesh::~Mesh() {}

// Implementation based on example usage here: https://github.com/syoyo/tinyobjloader
void Mesh::LoadFromFile(const char* filepath) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filepath);

    if (!err.empty()) { // `err` may contain warning message
        std::cerr << err << std::endl;
    }

    if (!ret) {
        exit(EXIT_FAILURE);
    }

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); ++s) {
        // Loop over faces (polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            Triangle t = Triangle();

            // Loop over vertices in the face
            for (size_t v = 0; v < fv; ++v) {
                const unsigned int index = (unsigned int)(index_offset + v);
                tinyobj::index_t idx = shapes[s].mesh.indices[index];
                Vertex newVert;
                newVert.pos = { attrib.vertices[3 * idx.vertex_index], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2] };
                newVert.nor = { attrib.normals[3 * idx.normal_index], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2] };
                vertices.emplace_back(newVert);
                indices.emplace_back(index);
                t.AppendVertex(newVert.pos);
            }
            index_offset += fv;
            t.ComputePlaneNormal();
            triangles.emplace_back(t);
        }
    }
    return;
}

// TODO: implement better tri intersection?
Intersection Triangle::Intersect(const Ray& r) const {

    // 1. Ray-plane intersection
    const float t = glm::dot(planeNormal, (points[0] - r.GetOrigin())) / glm::dot(planeNormal, r.GetDirection());
    if (t < 0) {
        return Intersection();
    }

    const glm::vec3 P = r.GetOrigin() + t * r.GetDirection();

    // 2. Barycentric test
    const float S = 1.0f / (0.5f * glm::length(glm::cross(points[0] - points[1], points[0] - points[2])));
    const float S1 = 0.5f * glm::length(glm::cross(P - points[1], P - points[2]));
    const float S2 = 0.5f * glm::length(glm::cross(P - points[2], P - points[0]));
    const float S3 = 0.5f * glm::length(glm::cross(P - points[0], P - points[1]));
    const float sum = (S1 + S2 + S3) * S;

    if ((S1 >= 0.0f && S1 <= 1.0f) && (S2 >= 0.0f && S2 <= 1.0f) && (S3 >= 0.0f && S3 <= 1.0f) && std::fabsf(sum - 1.0f) < EPSILON) {
        return Intersection(P, planeNormal, t);
    }
    return Intersection();
}

Intersection Mesh::Intersect(const Ray& r) const {
    Intersection finalIsect = triangles[0].Intersect(r);
    for (unsigned int i = 1; i < (unsigned int)triangles.size(); ++i) {
        const Intersection isect = triangles[i].Intersect(r);
        if (isect.IsValid() && (!finalIsect.IsValid() || isect.GetT() < finalIsect.GetT())) {
            finalIsect = isect;
        }
    }
    return finalIsect;
}

const bool Mesh::Contains(const glm::vec3 & p) const {
    Ray r = Ray(p, glm::vec3(1.0f, 0.0f, 0.0f)); // Ray direction is arbitrary. It can be anything
    Intersection isect = Intersect(r);
    unsigned int isectCounter = 0;
    while (isect.IsValid()) {
        ++isectCounter;
        isectCounter %= 2;
        r = isect.SpawnRayAtPoint(r);
        isect = Intersect(r);
    }
    return isectCounter == 1; // There was an odd number of intersections
}
