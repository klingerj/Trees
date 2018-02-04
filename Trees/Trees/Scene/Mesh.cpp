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

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; ++v) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                Vertex newVert;
                newVert.pos = { attrib.vertices[3 * idx.vertex_index], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2] };
                newVert.nor = { attrib.normals[3 * idx.normal_index], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2] };
                vertices.emplace_back(newVert);
                indices.emplace_back(index_offset + v);
                
                /*tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];*/
                //tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
                //tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
            }
            index_offset += fv;

            // per-face material
            //shapes[s].mesh.material_ids[f];
        }
    }
    return;
}
