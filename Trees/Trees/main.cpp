#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "pcg_random.hpp"

#include "OpenGL\ShaderProgram.h"
#include "Scene\Mesh.h"
#include "Scene\Camera.h"

#include <iostream>
#include <vector>
#include <random>

// For performance analysis / timing
//#include <chrono>
//#include <ctime>

#define GLM_FORCE_RADIANS
#define VIEWPORT_WIDTH_INITIAL 800
#define VIEWPORT_HEIGHT_INITIAL 600

// For 5-tree scene, eye and ref: glm::vec3(0.25f, 0.5f, 3.5f), glm::vec3(0.25f, 0.0f, 0.0f
Camera camera = Camera(glm::vec3(0.0f, 0.35f, 0.9f), glm::vec3(0.0f, 0.35f, 0.0f), 0.7853981634f,
                          (float)VIEWPORT_WIDTH_INITIAL / VIEWPORT_HEIGHT_INITIAL, 0.01f, 10.0f);

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    camera.SetAspect((float)width / height);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// Tree growth code - will need to be refactored TO ANOTHER FILE PLS ALSO CANT TURN OFF CAPS LOCK RN SO

class TreeNode {
private:
    glm::vec3 point; // Point in world space
    float influenceDist; // Radius of sphere of influence
    int parentIdx; // index of the parent of this Node, in the array of nodes. Will probably change

public:
    TreeNode(const glm::vec3& p, const float& d, const int& i) : point(p), influenceDist(d), parentIdx(i) {}
    ~TreeNode() {}
    inline bool InfluencesPoint(const glm::vec3& p) const {
        return glm::distance2(p, point) < (influenceDist * influenceDist);
    }
    inline const glm::vec3 GetPoint() const {
        return point;
    }
    inline const int GetParentIndex() const {
        return parentIdx;
    }
};

class AttractorPoint {
private:
    glm::vec3 point; // Point in world space
    float killDist; // Radius for removal

public:
    AttractorPoint(const glm::vec3& p, const float& d) : point(p), killDist(d) {}
    ~AttractorPoint() {}
    inline const glm::vec3 GetPoint() const {
        return point;
    }
    inline bool IsKilledBy(const glm::vec3& p) const {
        return glm::distance2(p, point) < (killDist * killDist);
    }
};

int main() {
    // Test Mesh Loading
    //Mesh m = Mesh();
    //m.LoadFromFile("OBJs/plane.obj");

    // GLFW Window Setup
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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
    const unsigned int numPoints = 4000;
    unsigned int numPointsIncluded = 0;
    std::vector<glm::vec3> points = std::vector<glm::vec3>();

    // Using PCG RNG: http://www.pcg-random.org/using-pcg-cpp.html

    // Seed with a real random value, if available
    //pcg_extras::seed_seq_from<std::random_device> seed_source;

    // Make a random number engine
    pcg32 rng(101);

    // Testing for seeds that will cause the crash and for ones that don't
    // Good seed: 100
    // Bad seed (causes crash): 101

    std::uniform_real_distribution<float> dis(-3.0f, 3.0f);

    // Create points
    // Unfortunately, we can't really do any memory preallocating because we don't actually know how many points will be included
    for (unsigned int i = 0; i < numPoints; ++i) {
        const glm::vec3 p = glm::vec3(dis(rng) * 0.16667f, (dis(rng) + 3.0f) * 0.083333f * 0.75f + 0.2f, dis(rng) * 0.083333f * 0.75f);
        if ((p.x * p.x + p.y * p.y) > 0.1f /*p.y > 0.2f*/ /*&& (p.x * p.x + p.y * p.y) > 0.2f*/) {
            points.emplace_back(p);
            ++numPointsIncluded;
        }
    }

    // Create the AttractorPoints
    const float killDist = 0.25f;
    std::vector<AttractorPoint> attractorPoints = std::vector<AttractorPoint>();
    attractorPoints.reserve(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        attractorPoints.emplace_back(AttractorPoint(points[i], killDist));
    }

    // Create the TreeNode(s)
    const float branchLength = 0.03f;
    const float branchInflDist = 0.26f;
    std::vector<TreeNode> treeNodes = std::vector<TreeNode>();

    // Place the Tree "seeds"
    /*for (int i = 1; i <= 5; ++i) {
        treeNodes.emplace_back(TreeNode(glm::vec3(0.75f * ((float)i) - 2.0f, 0.0f, 0.0f), branchInflDist + 0.05f * i, -1));
    }*/

    treeNodes.emplace_back(TreeNode(glm::vec3(0.01f, 0.1f, 0.0f), branchInflDist, -1));

    // Run the tree algorithm
    //TODO
    //for a certain number of iterations?
    //for each tree node
    // for each point
    //    get each point in reach. add it to a vector of points.
    //       compute the direction of the next treenode, create it a distance of d away
    //for each point
    //  for each tree node
    //    if a tree node is in the kill distance, remove this attractor point

    const unsigned int numIters = 15;
    unsigned int numTreeNodes = (unsigned int)treeNodes.size(); // Update the number of tree nodes to run the algorithm on in the for loop
    bool didUpdate = false; // Flag indicating that none of the tree nodes had any nearby points, so kill the algorithm

    for (unsigned int n = 0; n < numIters && attractorPoints.size() > 0; ++n) {
        // Create TreeNodes
        for (unsigned int ti = 0; ti < numTreeNodes; ++ti) { // Perform the algorithm for each tree node
            glm::vec3 accumDir = glm::vec3(0.0f); // Accumulate the direction of each influencing AttractorPoint
            bool numNearbyPoints = false; // Flag indicating there is at least one nearby attractor point
            const TreeNode& currTreeNode = treeNodes[ti];
            const glm::vec3& treeNodePoint = currTreeNode.GetPoint();

            for (unsigned int pi = 0; pi < attractorPoints.size(); ++pi) {
                const glm::vec3& attrPoint = attractorPoints[pi].GetPoint();

                if (currTreeNode.InfluencesPoint(attrPoint)) {
                    accumDir += attrPoint - treeNodePoint;
                    numNearbyPoints = true;
                }
            }

            // If at least one attractor point is within the sphere of influence of this tree node
            if (numNearbyPoints) {
                didUpdate = true;
                // Normalize the accumulated direction
                accumDir = glm::normalize(accumDir);

                /*const glm::vec3 tropism = glm::vec3(0.0f, -0.01f * n * n, 0.0f);
                accumDir += tropism;*/
                accumDir = glm::normalize(accumDir);

                // Create a new TreeNode
                treeNodes.emplace_back(TreeNode(treeNodePoint + accumDir * branchLength, branchInflDist, ti));
            }
        }

        if (!didUpdate) {
            break; // kill the algorithm
        }

        // Kill attractor points that need to be killed
        // https://stackoverflow.com/questions/347441/erasing-elements-from-a-vector

        auto attrPtIter = attractorPoints.begin();

        //int i = 0; // count where we are in the loop for when we break for debugging reasons
        while (attrPtIter != attractorPoints.end()) {
            bool didRemovePoint = false;
            for (unsigned int ti = 0; ti < numTreeNodes; ++ti) { // size does NOT include the newly created tree nodes
                if (attrPtIter->IsKilledBy(treeNodes[ti].GetPoint())) {
                    attrPtIter = attractorPoints.erase(attrPtIter); // crash here occasionally *** TODO
                    didRemovePoint = true;
                    break;
                }
            }
            if (!didRemovePoint) {
                ++attrPtIter;
            }
        }
        //std::cout << "Num points left: " << attractorPoints.size() << std::endl;
        numTreeNodes = (unsigned int)treeNodes.size();

        /*for (unsigned int pi = 0; pi < attractorPoints.size(); ++pi) {
            const AttractorPoint& currAttrPt = attractorPoints[pi];
            for (unsigned int ti = 0; ti < treeNodes.size(); ++ti) { // size includes the newly created tree nodes
                if (currAttrPt.IsKilledBy(treeNodes[ti].GetPoint())) {
                    attractorPoints.erase(attractorPoints.begin() + pi);
                    // TODO: update the buffers that get drawn
                }
            }
        }*/
    }

    // Print out info
    std::cout << "Number of Iterations: " << numIters << std::endl;
    std::cout << "Branch Length: " << branchLength << std::endl;
    std::cout << "Kill Distance: " << killDist << std::endl;
    std::cout << "Node Influence Distance: " << branchInflDist << std::endl;
    std::cout << "Number of attractor points (initial): " << numPointsIncluded << std::endl;
    std::cout << "Number of attractor points (final): " << attractorPoints.size() << std::endl;
    std::cout << "Number of Tree Nodes: " << treeNodes.size() << std::endl;

    // Create indices for the attractor points
    std::vector<unsigned int> indices = std::vector<unsigned int>(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        indices[i] = i;
    }

    /// Untransformed cylinder code
    std::vector<glm::vec3> cylPoints = std::vector<glm::vec3>();
    std::vector<glm::vec3> cylNormals = std::vector<glm::vec3>();
    std::vector<unsigned int> cylIndices = std::vector<unsigned int>();
    const float radius = 0.005f;
    // Could speed this up by calling .reserve()
    // should support variable LOD branch cylinders? maybe later

    // Generate Positions:
    // Store top cap verts (Indices 0 - 19)
    for (int i = 0; i < 20; ++i) {
        cylPoints.emplace_back(glm::vec3(glm::rotate(glm::mat4(1.0f), i * 18.0f, glm::vec3(0, 1, 0)) * glm::vec4(radius, 1.0f, 0, 1)));
    }

    // Store bottom cap verts (Indices 20 - 39)
    for (int i = 20; i < 40; ++i) {
        cylPoints.emplace_back(glm::vec3(glm::rotate(glm::mat4(1.0f), (i - 20) * 18.0f, glm::vec3(0, 1, 0)) * glm::vec4(radius, -1.0f, 0, 1)));
    }

    // Generate second rings for the barrel of the cylinder

    // Store top cap verts (Indices 40 - 59)
    for (int i = 0; i < 20; ++i) {
        cylPoints.emplace_back(glm::vec3(glm::rotate(glm::mat4(1.0f), i * 18.0f, glm::vec3(0, 1, 0)) * glm::vec4(radius, 1.0f, 0, 1)));
    }
    // Store bottom cap verts (Indices 60 - 79)
    for (int i = 20; i < 40; ++i) {
        cylPoints.emplace_back(glm::vec3(glm::rotate(glm::mat4(1.0f), (i - 20) * 18.0f, glm::vec3(0, 1, 0)) * glm::vec4(radius, -1.0f, 0, 1)));
    }

    // Generate Normals:
    for (int i = 0; i < 20; i++) {
        cylNormals.emplace_back(glm::vec3(0, 1, 0));
    }
    //Store bottom cap normals (IDX 20 - 39)
    for (int i = 20; i < 40; i++) {
        cylNormals.emplace_back(glm::vec3(0, -1, 0));
    }
    //Store top of barrel normals (IDX 40 - 59)
    for (int i = 0; i < 20; i++) {
        cylNormals.emplace_back(glm::vec3(glm::rotate(glm::mat4(1.0f), i*18.0f, glm::vec3(0, 1, 0)) * glm::vec4(1, 0, 0, 0)));
    }
    //Store bottom of barrel normals (IDX 60 - 79)
    for (int i = 20; i < 40; i++) {
        cylNormals.emplace_back(glm::vec3(glm::rotate(glm::mat4(1.0f), (i - 20)*18.0f, glm::vec3(0, 1, 0)) * glm::vec4(1, 0, 0, 0)));
    }

    // Generate Indices:
    //Build indices for the top cap (18 tris, indices 0 - 53, up to vertex 19)
    for (int i = 0; i < 18; ++i) {
        cylIndices.emplace_back(0);
        cylIndices.emplace_back(i + 1);
        cylIndices.emplace_back(i + 2);
    }
    //Build indices for the top cap (18 tris, indices 54 - 107, up to vertex 39)
    for (int i = 18; i < 36; ++i) {
        cylIndices.emplace_back(20);
        cylIndices.emplace_back(i + 3);
        cylIndices.emplace_back(i + 4);
    }
    //Build indices for the barrel of the cylinder
    for (int i = 0; i < 19; ++i) {
        cylIndices.emplace_back(i + 40);
        cylIndices.emplace_back(i + 41);
        cylIndices.emplace_back(i + 60);
        cylIndices.emplace_back(i + 41);
        cylIndices.emplace_back(i + 61);
        cylIndices.emplace_back(i + 60);
    }
    //Build the last quad of the barrel, which has looping indices
    cylIndices.emplace_back(59);
    cylIndices.emplace_back(40);
    cylIndices.emplace_back(79);
    cylIndices.emplace_back(40);
    cylIndices.emplace_back(60);
    cylIndices.emplace_back(79);

    // End Cylinder generation

    // Create points and indices for the tree branches
    std::vector<glm::vec3> pointsTreeBranch = std::vector<glm::vec3>();
    //std::vector<glm::vec3> normalsTreeBranch = std::vector<glm::vec3>();
    std::vector<unsigned int> indicesTreeBranch = std::vector<unsigned int>();
    int idxCounter = 0;
    for (unsigned int i = (unsigned int)treeNodes.size() - 1; i > 0; --i) {
        const TreeNode& currTreeNode = treeNodes[i];
        const int parentIdx = currTreeNode.GetParentIndex();
        if (parentIdx != -1) {

            // Copy over the platonic cylinder positions
            // transform them individually i guess
            // Check if i should emplace_back repeatedly or create a whole new vector and use insert - performance compare after getting them working
            // Could//Should move this to a compute shader bc doing this sequentially is lame

            // Compute the rotation using quaternion
            const glm::vec3 branchBasePoint = treeNodes[parentIdx].GetPoint();
            glm::vec3 branchAxis = currTreeNode.GetPoint() - branchBasePoint;

            // Compute the translation component real quick - set to the the base branch point + 0.5 * D, placing the cylinder at the halfway point
            const glm::vec3 translation = branchBasePoint + 0.5f * branchAxis;
            const float branchAxisLength = length(branchAxis);
            
            // Back to rotation
            branchAxis /= branchAxisLength;
            const glm::vec3 axis = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), branchAxis); // might be negative / TODO *** goes to zero for perfectly vertical trees. find a new way to do this
            const float angle = std::acos(glm::dot(glm::vec3(0.0f, 1.0f, 0.0f), branchAxis));

            const glm::quat branchQuat = glm::angleAxis(angle, axis);
            glm::mat4 branchTrans = glm::toMat4(branchQuat);

            // Create an overall transformation matrix of translation and rotation
            branchTrans = glm::translate(glm::mat4(1.0f), translation) * branchTrans * glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, branchAxisLength * 0.5f, 1.0f));

            std::vector<glm::vec3> cylPointsTrans = std::vector<glm::vec3>();
            std::vector<glm::vec3> cylNormalsTrans = std::vector<glm::vec3>();
            // Interleave VBO data
            for (int i = 0; i < cylPoints.size(); ++i) {
                cylPointsTrans.emplace_back(glm::vec3(branchTrans * glm::vec4(cylPoints[i], 1.0f)));
                cylPointsTrans.emplace_back(glm::vec3(branchTrans * glm::vec4(cylNormals[i], 0.0f)));
            }

            /*for (int i = 0; i < cylNormals.size(); ++i) {
                
            }*/

            std::vector<unsigned int> cylIndicesNew = std::vector<unsigned int>();
            for (int i = 0; i < cylIndices.size(); ++i) {
                cylIndicesNew.emplace_back(cylIndices[i] + pointsTreeBranch.size() / 2);
            }


            // Creation for GL_LINES branches
            /*pointsTreeBranch.push_back(treeNodes[parentIdx].GetPoint()); // base of branch
            pointsTreeBranch.push_back(currTreeNode.GetPoint()); // branch end point
            indicesTreeBranch.emplace_back(idxCounter++);
            indicesTreeBranch.emplace_back(idxCounter++);*/

            pointsTreeBranch.insert(pointsTreeBranch.end(), cylPointsTrans.begin(), cylPointsTrans.end());
            //normalsTreeBranch.insert(normalsTreeBranch.end(), cylNormalsTrans.begin(), cylNormalsTrans.end());
            indicesTreeBranch.insert(indicesTreeBranch.end(), cylIndicesNew.begin(), cylIndicesNew.end());
        }
    }
    
    /// GL calls and drawing
    
    ShaderProgram sp = ShaderProgram("Shaders/point-vert.vert", "Shaders/point-frag.frag");
    ShaderProgram sp2 = ShaderProgram("Shaders/treeNode-vert.vert", "Shaders/treeNode-frag.frag");
    ShaderProgram sp3 = ShaderProgram("Shaders/mesh-vert.vert", "Shaders/mesh-frag.frag");
    
    // Array/Buffer Objects
    unsigned int VAO, VAO2, VAO3;
    unsigned int VBO, VBO2, VBO3;
    unsigned int EBO, EBO2, EBO3;
    glGenVertexArrays(1, &VAO);
    glGenVertexArrays(1, &VAO2);
    glGenVertexArrays(1, &VAO3);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &VBO3);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &EBO2);
    glGenBuffers(1, &EBO3);

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
    glBindVertexArray(0);

    glBindVertexArray(VAO2);
    // Tree Branches VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * pointsTreeBranch.size(), pointsTreeBranch.data(), GL_STATIC_DRAW);
    // Tree Branches EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indicesTreeBranch.size(), indicesTreeBranch.data(), GL_STATIC_DRAW);
    // Attribute linking
    // Pos and Nor
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3) * 2, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3) * 2, (void*)sizeof(glm::vec3));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
    
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

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindVertexArray(VAO);
        sp.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_POINTS, (GLsizei) tempPtsIdx.size(), GL_UNSIGNED_INT, 0);
        
        glBindVertexArray(VAO2);
        sp2.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_TRIANGLES, (GLsizei) indicesTreeBranch.size(), GL_UNSIGNED_INT, 0);

        /*glBindVertexArray(VAO3);
        sp3.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_TRIANGLES, (GLsizei) idx.size(), GL_UNSIGNED_INT, 0);*/

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteVertexArrays(1, &VAO2);
    glDeleteVertexArrays(1, &VAO3);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &VBO2);
    glDeleteBuffers(1, &VBO3);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &EBO2);
    glDeleteBuffers(1, &EBO3);

    glfwTerminate();
    return 0;
}
