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

// draw branches as 3d geometry vs gl lines for branches
//#define CUBES

// For 5-tree scene, eye and ref: glm::vec3(0.25f, 0.5f, 3.5f), glm::vec3(0.25f, 0.0f, 0.0f
Camera camera = Camera(glm::vec3(0.0f, 3.0f, 0.0f), 0.7853981634f, // 45 degrees vs 75 degrees
(float)VIEWPORT_WIDTH_INITIAL / VIEWPORT_HEIGHT_INITIAL, 0.01f, 2000.0f, 10.0f, 0.0f, 5.0f);
const float camMoveSensitivity = 0.001f;

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
    //Mesh m = Mesh();
    //m.LoadFromFile("OBJs/plane.obj");

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
    const unsigned int numPoints = 100000;
    unsigned int numPointsIncluded = 0;
    std::vector<glm::vec3> points = std::vector<glm::vec3>();

    // Using PCG RNG: http://www.pcg-random.org/using-pcg-cpp.html

    // Make a random number engine
    pcg32 rng(101);

    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    auto start = std::chrono::system_clock::now();
    // Unfortunately, we can't really do any memory preallocating because we don't actually know how many points will be included
    for (unsigned int i = 0; i < numPoints; ++i) {
        const glm::vec3 p = glm::vec3(dis(rng) * 5.0f, dis(rng) * 5.0f, dis(rng) * 5.0f); // for big cube growth chamber: scales of 10, 20, 10
        if (glm::length(p) < 5.0f /*p.y > 0.2f*/ /*&& (p.x * p.x + p.y * p.y) > 0.2f*/) {
            points.emplace_back(p + glm::vec3(0.0f, 2.51f, 0.0f));
            ++numPointsIncluded;
        }
    }
    // Create the actual AttractorPoints
    const float killDist = 0.05f;
    std::vector<AttractorPoint> attractorPoints = std::vector<AttractorPoint>();
    attractorPoints.reserve(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        attractorPoints.emplace_back(AttractorPoint(points[i], killDist));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Attractor Point Generation: " << elapsed_seconds.count() << "s\n";

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

    std::vector<glm::vec3> internodePoints = std::vector <glm::vec3>();
    std::vector<glm::vec3> internodeNormals = std::vector <glm::vec3>();
    std::vector<unsigned int> internodeIndices = std::vector <unsigned int>();

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
            const bool axesAreAligned = std::abs(glm::dot(branchAxis, WORLD_UP_VECTOR)) > 0.99f;
            glm::vec3 crossVec = axesAreAligned ? glm::vec3(1.0f, 0.0f, 0.0f) : WORLD_UP_VECTOR; // avoid glm::cross returning a nan or 0-vector
            const glm::vec3 axis = glm::normalize(glm::cross(crossVec, branchAxis));
            const float angle = std::acos(glm::dot(branchAxis, WORLD_UP_VECTOR));

            const glm::quat branchQuat = glm::angleAxis(angle, axis);
            glm::mat4 branchTransform = glm::toMat4(branchQuat); // initially just a rotation matrix, eventually stores the entire transformation

            // Compute the translation component - set to the the base branch point + 0.5 * internodeLength, placing the cylinder at the halfway point
            const glm::vec3 translation = internodeEndPoint - 0.5f * branchAxis * currentBud.internodeLength;

            // Create an overall transformation matrix of translation and rotation
            branchTransform = glm::translate(glm::mat4(1.0f), translation) * branchTransform * glm::scale(glm::mat4(1.0f), glm::vec3(currentBud.branchRadius * 0.01f, currentBud.internodeLength * 0.5f, currentBud.branchRadius * 0.01f));

            std::vector<glm::vec3> cubePointsTrans = std::vector<glm::vec3>();
            std::vector<glm::vec3> cubeNormalsTrans = std::vector<glm::vec3>();
            // Interleave VBO data or nah
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

    // Array/Buffer Objects
    unsigned int VAO, VAO2, VAO3, VAO4;
    unsigned int VBO, VBO2, VBO3, VBO4;
    unsigned int EBO, EBO2, EBO3, EBO4;
    glGenVertexArrays(1, &VAO);
    glGenVertexArrays(1, &VAO2);
    glGenVertexArrays(1, &VAO3);
    glGenVertexArrays(1, &VAO4);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &VBO3);
    glGenBuffers(1, &VBO4);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &EBO2);
    glGenBuffers(1, &EBO3);
    glGenBuffers(1, &EBO4);

    // VAO Binding
    glBindVertexArray(VAO);

    // VBO Binding
    // Points
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    std::vector<glm::vec3> tempPts = std::vector<glm::vec3>();
    tempPts.reserve(attractorPoints.size());
    for (int i = 0; i < attractorPoints.size(); ++i) {
        tempPts.emplace_back(attractorPoints[i].GetPoint());
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

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for GL calls and ShaderPrograms and VAO/VBO/EBO Creation: " << elapsed_seconds.count() << "s\n";



    // This was TinyOBJ Debugging
    /*for (int i = 0; i < m.GetVertices().size(); i++) {
        std::cout << m.GetVertices()[i].pos.x << m.GetVertices()[i].pos.y << m.GetVertices()[i].pos.z << std::endl;
        std::cout << m.GetVertices()[i].nor.x << m.GetVertices()[i].nor.y << m.GetVertices()[i].nor.z << std::endl;
    }

    for (int i = 0; i < m.GetIndices().size(); i++) {
        std::cout << m.GetIndices()[i] << std::endl;
    }*/

    //std::vector<unsigned int> idx = m.GetIndices();

    // Mesh buffers
    /*glBindVertexArray(VAO3);
    glBindBuffer(GL_ARRAY_BUFFER, VBO3);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * m.GetVertices().size(), m.GetVertices().data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO3);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * idx.size(), idx.data(), GL_STATIC_DRAW);
    // Attribute linking

    // Positions + Normals
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(0);
    // Bind the 0th VBO. Set up attribute pointers to location 1 for normals.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)sizeof(glm::vec3)); // skip the first Vertex.pos
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);*/

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
        sp3.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        #ifndef INTERNODE_CUBES
        // draw GL_LINES intenodes
        glDrawElements(GL_LINES, (GLsizei)branchIndices.size(), GL_UNSIGNED_INT, 0);
        #else
        // do nothing idk
        #endif

        /*glBindVertexArray(VAO4);
        sp.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_POINTS, (GLsizei)budIndices.size(), GL_UNSIGNED_INT, 0);*/

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteVertexArrays(1, &VAO2);
    glDeleteVertexArrays(1, &VAO3);
    glDeleteVertexArrays(1, &VAO4);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &VBO2);
    glDeleteBuffers(1, &VBO3);
    glDeleteBuffers(1, &VBO4);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &EBO2);
    glDeleteBuffers(1, &EBO3);
    glDeleteBuffers(1, &EBO4);

    glfwTerminate();
    return 0;
}
