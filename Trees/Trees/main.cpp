#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "pcg_random.hpp"

#include "OpenGL\ShaderProgram.h"
#include "Scene\Mesh.h"
#include "Scene\Camera.h"
#include "Scene\Tree.h"
#include "Scene\Globals.h"

#include <iostream>
#include <vector>
#include <random>

// For performance analysis / timing
#include <chrono>
#include <ctime>

Camera camera = Camera(glm::vec3(0.0f, 1.63f, 0.0f), 0.7853981634f, // 45 degrees vs 75 degrees
(float)VIEWPORT_WIDTH_INITIAL / VIEWPORT_HEIGHT_INITIAL, 0.01f, 2000.0f, 0.0f, -31.74f, 5.4f);
const float camMoveSensitivity = 0.03f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    camera.SetAspect((float)width / height);
}

// Keyboard controls
void processInput(GLFWwindow *window) {

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    } else if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.TranslateAlongRadius(-camMoveSensitivity);
    } else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera.TranslateAlongRadius(camMoveSensitivity);
    } else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        camera.RotatePhi(-camMoveSensitivity);
    } else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        camera.RotatePhi(camMoveSensitivity);
    } else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        camera.RotateTheta(-camMoveSensitivity);
    } else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        camera.RotateTheta(camMoveSensitivity);
    } else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        camera.TranslateRefAlongWorldY(-camMoveSensitivity);
    } else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        camera.TranslateRefAlongWorldY(camMoveSensitivity);
    }
}

int main() {
    // Test Mesh Loading
    Mesh m = Mesh();
    m.LoadFromFile("OBJs/sphereLowPoly.obj");
    Mesh m2 = Mesh();
    m2.LoadFromFile("OBJs/leaf.obj");

    // GLFW Window Setup
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef ENABLE_MULTISAMPLING
    glfwWindowHint(GLFW_SAMPLES, 4);
    #endif

    GLFWwindow* window = glfwCreateWindow(VIEWPORT_WIDTH_INITIAL, VIEWPORT_HEIGHT_INITIAL, "Trees", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize Glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, VIEWPORT_WIDTH_INITIAL, VIEWPORT_HEIGHT_INITIAL);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Test loop vectorization. Note: using the compiler flags, this stuff only seems to compile in Release Mode
    // Needed flags: /O2 /Qvec-report:1 (can also use report:2)
    // Source: https://software.intel.com/en-us/articles/a-guide-to-auto-vectorization-with-intel-c-compilers

    // Stores the point positions: currently a list of floats. I need to include glm or eigen
    // Is it faster to initialize a vector of points with # and value and then set the values, or to push_back values onto an empty list
    // Answer to that: https://stackoverflow.com/questions/32199388/what-is-better-reserve-vector-capacity-preallocate-to-size-or-push-back-in-loo
    // Best options seem to be preallocate or emplace_back with reserve
    const unsigned int numPoints = 1000000; //300k for heart
    unsigned int numPointsIncluded = 0;
    std::vector<glm::vec3> points = std::vector<glm::vec3>();

    // Using PCG RNG: http://www.pcg-random.org/using-pcg-cpp.html

    // Make a random number engine
    pcg32 rng(101);

    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    auto start = std::chrono::system_clock::now();
    // Unfortunately, we can't really do any memory preallocating because we don't actually know how many points will be included
    for (unsigned int i = 0; i < numPoints; ++i) {
        /*const glm::vec3 p = glm::vec3(dis(rng) * 10.0f, dis(rng) * 10.0f, dis(rng) * 10.0f); // for big cube growth chamber: scales of 10, 20, 10
        if (p.x * p.x + p.z * p.z < 10.0f) {
            points.emplace_back(p + glm::vec3(0.0f, 10.0f, 0.0f));
        }*/ // cylinder sdf

        const glm::vec3 p = glm::vec3(dis(rng) * 2.0f /** -0.6f*/, dis(rng) * 2.0f /*0.5f*/  /** 0.012f*/, dis(rng) * 4.0f /** 0.113f*/);
        
        // Intersect with mesh instead
        if (m.Contains(p)) {
            points.emplace_back(p);
            ++numPointsIncluded;
        }
    }

    // Create the actual AttractorPoints
    std::vector<AttractorPoint> attractorPoints = std::vector<AttractorPoint>();
    attractorPoints.reserve(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        attractorPoints.emplace_back(AttractorPoint(points[i]));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Attractor Point Generation: " << elapsed_seconds.count() << "s\n";
    std::cout << "Number of Attractor Points: " << numPointsIncluded << "\n\n";

    // new tree generation
    Tree tree = Tree(glm::vec3(0.0f, 0.0f, 0.0f));

    start = std::chrono::system_clock::now();
    tree.IterateGrowth(NUM_ITERATIONS, attractorPoints);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";

    // Code for GL stuff

    // Create indices for the attractor points
    std::vector<unsigned int> indices = std::vector<unsigned int>(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        indices[i] = i;
    }

    // Now using cubes / rectangular prisms, will be slightly faster and have less visual issues

    std::vector<glm::vec3> cubePoints;
    std::vector<glm::vec3> cubeNormals;
    std::vector<unsigned int> cubeIndices;

    start = std::chrono::system_clock::now();

    ///Positions
    //Front face
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, 1.0f));

    //Right face
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, 1.0f));

    //Left face
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, -1.0f));

    //Back face
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, -1.0f));

    //Top face
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, -1.0f));

    //Bottom face
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, 1.0f));

    /// Normals
    //Front
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, 0, 1));
    }
    //Right
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(1, 0, 0));
    }
    //Left
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(-1, 0, 0));
    }
    //Back
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, 0, -1));
    }
    //Top
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, 1, 0));
    }
    //Bottom
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, -1, 0));
    }

    /// Indices
    for (int i = 0; i < 6; i++) {
        cubeIndices.emplace_back(i * 4);
        cubeIndices.emplace_back(i * 4 + 1);
        cubeIndices.emplace_back(i * 4 + 2);
        cubeIndices.emplace_back(i * 4);
        cubeIndices.emplace_back(i * 4 + 2);
        cubeIndices.emplace_back(i * 4 + 3);
    }

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Cube Primitive Generation: " << elapsed_seconds.count() << "s\n";

    // End cube generation

    // GL code

    start = std::chrono::system_clock::now();

    // Vectors of stuff to render
    #define INTERNODE_CUBES

    std::vector<glm::vec3> budPoints = std::vector<glm::vec3>();
    std::vector<unsigned int> budIndices = std::vector<unsigned int>();
    std::vector<unsigned int> branchIndices = std::vector<unsigned int>(); // one gl line corresponds to one internode. These indices will use the budPoints vector.

    std::vector<glm::vec3> internodePoints = std::vector<glm::vec3>();
    std::vector<glm::vec3> internodeNormals = std::vector<glm::vec3>();
    std::vector<unsigned int> internodeIndices = std::vector<unsigned int>();

    // Leaves
    // extract the position and normal data from the leaf mesh
    std::vector<glm::vec3> leafGeomPoints = std::vector<glm::vec3>();
    std::vector<glm::vec3> leafGeomNormals = std::vector<glm::vec3>();
    std::vector<unsigned int> leafGeomIndices = m2.GetIndices();
    const std::vector<Vertex>& leafVertices = m2.GetVertices();
    for (int i = 0; i < leafVertices.size(); ++i) {
        leafGeomPoints.emplace_back(leafVertices[i].pos);
        leafGeomNormals.emplace_back(leafVertices[i].nor);
    }

    std::vector<glm::vec3> leafPoints = std::vector<glm::vec3>();
    std::vector<glm::vec3> leafNormals = std::vector<glm::vec3>();
    std::vector<unsigned int> leafIndices = std::vector<unsigned int>();

    // get the points vectors
    const std::vector<TreeBranch>& branches = tree.GetBranches();
    for (int br = 0; br < tree.GetBranches().size(); ++br) {
        const std::vector<Bud>& buds = branches[br].GetBuds();
        #ifndef INTERNODE_CUBES
        const unsigned int idxOffset = budPoints.size();
        #endif
        int bu;
        #ifdef INTERNODE_CUBES
        bu = 1;
        #else
        bu = 0;
        #endif;
        for (; bu < buds.size(); ++bu) {
            #ifdef INTERNODE_CUBES // create internode VBO
            const Bud& currentBud = buds[bu];
            const glm::vec3& internodeEndPoint = currentBud.point; // effectively, just the position of the bud at the end of the current internode
            glm::vec3 branchAxis = glm::normalize(internodeEndPoint - buds[bu - 1].point); // not sure if i store this direction somewhere

            // Back to rotation

            // orientation is messed up in certain cases still...
            //const bool axesAreAligned = std::abs(glm::dot(branchAxis, WORLD_UP_VECTOR)) > 0.99f;
            //glm::vec3 crossVec = axesAreAligned ? glm::vec3(1.0f, 0.0f, 0.0f) : WORLD_UP_VECTOR; // avoid glm::cross returning a nan or 0-vector
            const float angle = std::acos(glm::dot(branchAxis, WORLD_UP_VECTOR));
            glm::mat4 branchTransform;
            if (angle > 0.01f) {
                const glm::vec3 axis = glm::normalize(glm::cross(WORLD_UP_VECTOR, branchAxis));
                const glm::quat branchQuat = glm::angleAxis(angle, axis);
                branchTransform = glm::toMat4(branchQuat); // initially just a rotation matrix, eventually stores the entire transformation
            } else { // if it's pretty much straight up, call it straight up
                branchTransform = glm::mat4(1.0f);
            }

            // Compute the translation component - set to the the base branch point + 0.5 * internodeLength, placing the cylinder at the halfway point
            const glm::vec3 translation = internodeEndPoint - 0.5f * branchAxis * currentBud.internodeLength;

            // Create an overall transformation matrix of translation and rotation
            branchTransform = glm::translate(glm::mat4(1.0f), translation) * branchTransform * glm::scale(glm::mat4(1.0f), glm::vec3(currentBud.branchRadius * 0.02f, currentBud.internodeLength * 0.5f, currentBud.branchRadius * 0.02f));

            std::vector<glm::vec3> cubePointsTrans = std::vector<glm::vec3>();
            std::vector<glm::vec3> cubeNormalsTrans = std::vector<glm::vec3>();
            for (int i = 0; i < cubePoints.size(); ++i) {
                cubePointsTrans.emplace_back(glm::vec3(branchTransform * glm::vec4(cubePoints[i], 1.0f)));
                glm::vec3 transformedNormal = glm::normalize(glm::vec3(glm::inverse(glm::transpose(branchTransform)) * glm::vec4(cubeNormals[i], 0.0f)));
                cubeNormalsTrans.emplace_back(transformedNormal);
            }

            std::vector<unsigned int> cubeIndicesNew = std::vector<unsigned int>();
            for (int i = 0; i < cubeIndices.size(); ++i) {
                const unsigned int size = (unsigned int)internodePoints.size();
                cubeIndicesNew.emplace_back(cubeIndices[i] + size); // offset this set of indices by the # of positions. Divide by two bc it contains positions and normals
            }

            internodePoints.insert(internodePoints.end(), cubePointsTrans.begin(), cubePointsTrans.end());
            internodeNormals.insert(internodeNormals.end(), cubeNormalsTrans.begin(), cubeNormalsTrans.end());
            internodeIndices.insert(internodeIndices.end(), cubeIndicesNew.begin(), cubeIndicesNew.end());

            // Leaves
            if (currentBud.type == AXILLARY && currentBud.fate != FORMED_BRANCH/* && branches[br].GetAxisOrder() > 1*/) {
                const float leafScale = 0.05f * currentBud.internodeLength / currentBud.branchRadius; // Joe's made-up heuristic
                if (leafScale < 0.01) { break; }
                std::vector<glm::vec3> leafPointsTrans = std::vector<glm::vec3>();
                std::vector<glm::vec3> leafNormalsTrans = std::vector<glm::vec3>();
                const glm::mat4 leafTransform = glm::translate(glm::mat4(1.0f), internodeEndPoint) * glm::toMat4(glm::angleAxis(std::acos(glm::dot(currentBud.naturalGrowthDir, WORLD_UP_VECTOR)), glm::normalize(glm::cross(WORLD_UP_VECTOR, currentBud.naturalGrowthDir))));
                for (int i = 0; i < leafGeomPoints.size(); ++i) {
                    leafPointsTrans.emplace_back(glm::vec3(leafTransform * glm::vec4(leafGeomPoints[i] * leafScale, 1.0f)));
                    glm::vec3 transformedNormal = glm::normalize(glm::vec3(glm::inverse(glm::transpose(leafTransform)) * glm::vec4(leafGeomNormals[i], 0.0f)));
                    leafNormalsTrans.emplace_back(transformedNormal);
                }

                std::vector<unsigned int> leafIndicesNew = std::vector<unsigned int>();
                for (int i = 0; i < leafGeomIndices.size(); ++i) {
                    const unsigned int size = (unsigned int)leafPoints.size();
                    leafIndicesNew.emplace_back(leafGeomIndices[i] + size); // offset this set of indices by the # of positions. Divide by two bc it contains positions and normals
                }
                leafPoints.insert(leafPoints.end(), leafPointsTrans.begin(), leafPointsTrans.end());
                leafNormals.insert(leafNormals.end(), leafNormalsTrans.begin(), leafNormalsTrans.end());
                leafIndices.insert(leafIndices.end(), leafIndicesNew.begin(), leafIndicesNew.end());
            }

            #else // old GL_LINES code
            if (bu < buds.size() - 1) { // proper indexing, just go with it
                branchIndices.emplace_back(bu + idxOffset);
                branchIndices.emplace_back(bu + 1 + idxOffset);
            }
            #endif
            budPoints.emplace_back(buds[bu].point);
        }
    }

    // create indices vector
    for (int i = 0; i < budPoints.size(); ++i) {
        budIndices.emplace_back(i);
    }

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Tree VBO Info Creation: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();

    /// GL calls and drawing

    ShaderProgram sp = ShaderProgram("Shaders/point-vert.vert", "Shaders/point-frag.frag");
    ShaderProgram sp2 = ShaderProgram("Shaders/treeNode-vert.vert", "Shaders/treeNode-frag.frag");
    ShaderProgram sp3 = ShaderProgram("Shaders/mesh-vert.vert", "Shaders/mesh-frag.frag");
    ShaderProgram sp4 = ShaderProgram("Shaders/leaf-vert.vert", "Shaders/leaf-frag.frag");

    // Array/Buffer Objects
    unsigned int VAO, VAO2, VAO3, VAO4, VAO5, VAO6;
    unsigned int VBO, VBO2, VBO3, VBO4, VBO5, VBO6, VBO7;
    unsigned int EBO, EBO2, EBO3, EBO4, EBO5, EBO6, EBO7;

    glGenVertexArrays(1, &VAO);
    glGenVertexArrays(1, &VAO2);
    glGenVertexArrays(1, &VAO3);
    glGenVertexArrays(1, &VAO4);
    glGenVertexArrays(1, &VAO5);
    glGenVertexArrays(1, &VAO6);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &VBO3);
    glGenBuffers(1, &VBO4);
    glGenBuffers(1, &VBO5);
    glGenBuffers(1, &VBO6);
    glGenBuffers(1, &VBO7);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &EBO2);
    glGenBuffers(1, &EBO3);
    glGenBuffers(1, &EBO4);
    glGenBuffers(1, &EBO5);
    glGenBuffers(1, &EBO6);
    glGenBuffers(1, &EBO7);

    // VAO Binding
    glBindVertexArray(VAO);

    // VBO Binding
    // Points
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    std::vector<glm::vec3> tempPts = std::vector<glm::vec3>();
    tempPts.reserve(attractorPoints.size());
    for (int i = 0; i < attractorPoints.size(); ++i) {
        tempPts.emplace_back(attractorPoints[i].point);
    }
    std::vector<unsigned int> tempPtsIdx = std::vector<unsigned int>();
    tempPtsIdx.reserve(attractorPoints.size());
    for (int i = 0; i < attractorPoints.size(); ++i) {
        tempPtsIdx.emplace_back(i);
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * tempPts.size(), tempPts.data(), GL_STATIC_DRAW);
    // EBO Binding
    // Points
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * tempPtsIdx.size(), tempPtsIdx.data(), GL_STATIC_DRAW);
    // Attribute linking
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);



    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);

    #ifndef INTERNODE_CUBES
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * budPoints.size(), budPoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * budIndices.size(), budIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0); // pass positions
    glEnableVertexAttribArray(0);

    glBindVertexArray(VAO3);
    glBindBuffer(GL_ARRAY_BUFFER, VBO3);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * budPoints.size(), budPoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO3);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * branchIndices.size(), branchIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0); // pass positions
    glEnableVertexAttribArray(0);

    #else

    // Points
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * internodePoints.size(), internodePoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * internodeIndices.size(), internodeIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Normals
    glBindBuffer(GL_ARRAY_BUFFER, VBO3);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * internodeNormals.size(), internodeNormals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO3);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * internodeIndices.size(), internodeIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(1);

    #endif

    // Bud Points
    glBindVertexArray(VAO4);
    glBindBuffer(GL_ARRAY_BUFFER, VBO4);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * budPoints.size(), budPoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO4);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * budIndices.size(), budIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Leaves
    glBindVertexArray(VAO6);
    // Points
    glBindBuffer(GL_ARRAY_BUFFER, VBO6);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * leafPoints.size(), leafPoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO6);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * leafIndices.size(), leafIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Normals
    glBindBuffer(GL_ARRAY_BUFFER, VBO7);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * leafNormals.size(), leafNormals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO7);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * leafIndices.size(), leafIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(1);

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for GL calls and ShaderPrograms and VAO/VBO/EBO Creation: " << elapsed_seconds.count() << "s\n";

    // Mesh buffers
    std::vector<unsigned int> idx = m.GetIndices();

    glBindVertexArray(VAO5);
    glBindBuffer(GL_ARRAY_BUFFER, VBO5);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * m.GetVertices().size(), m.GetVertices().data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO5);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * idx.size(), idx.data(), GL_STATIC_DRAW);
    // Attribute linking

    // Positions + Normals
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    // Bind the 0th VBO. Set up attribute pointers to location 1 for normals.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)sizeof(glm::vec3)); // skip the first Vertex.pos
    glEnableVertexAttribArray(1);




    // GL Params or whatever
    // TODO Create an init() function
    glPointSize(2);
    glLineWidth(1);
    glEnable(GL_DEPTH_TEST);

    #ifdef ENABLE_MULTISAMPLING
    glEnable(GL_MULTISAMPLE);
    #endif

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Attractor Points
        /*glBindVertexArray(VAO);
        sp.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_POINTS, (GLsizei) tempPtsIdx.size(), GL_UNSIGNED_INT, 0);*/

        // old cubes / new bud points
        glBindVertexArray(VAO2);
        sp2.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        #ifndef INTERNODE_CUBES
        // draws bud points NOPE******
        //glDrawElements(GL_POINTS, (GLsizei)budIndices.size(), GL_UNSIGNED_INT, 0);
        #else
        // draw the internode geometry
        glDrawElements(GL_TRIANGLES, (GLsizei)internodeIndices.size(), GL_UNSIGNED_INT, 0);
        #endif

        glBindVertexArray(VAO3);
        // new tree branches
        //sp3.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        #ifndef INTERNODE_CUBES
        // draw GL_LINES intenodes
        glDrawElements(GL_LINES, (GLsizei)branchIndices.size(), GL_UNSIGNED_INT, 0);
        #else
        // do nothing idk
        #endif

        // Draw whatever mesh
        /*sp3.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glBindVertexArray(VAO5);
        glDrawElements(GL_TRIANGLES, (GLsizei)idx.size(), GL_UNSIGNED_INT, 0);*/

        // draw leaves
        glBindVertexArray(VAO6);
        sp4.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_TRIANGLES, (GLsizei)leafIndices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteVertexArrays(1, &VAO2);
    glDeleteVertexArrays(1, &VAO3);
    glDeleteVertexArrays(1, &VAO4);
    glDeleteVertexArrays(1, &VAO5);
    glDeleteVertexArrays(1, &VAO6);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &VBO2);
    glDeleteBuffers(1, &VBO3);
    glDeleteBuffers(1, &VBO4);
    glDeleteBuffers(1, &VBO5);
    glDeleteBuffers(1, &VBO6);
    glDeleteBuffers(1, &VBO7);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &EBO2);
    glDeleteBuffers(1, &EBO3);
    glDeleteBuffers(1, &EBO4);
    glDeleteBuffers(1, &EBO5);
    glDeleteBuffers(1, &EBO6);
    glDeleteBuffers(1, &EBO7);

    glfwTerminate();
    return 0;
}
