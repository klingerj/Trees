#include "glad\glad.h"
#include "GLFW\glfw3.h"
#include "glm\glm.hpp"
#include "pcg_random.hpp"

#include "OpenGL/ShaderProgram.h"
#include "Scene/Mesh.h"

#include <iostream>
#include <vector>
#include <random>

// For performance analysis / timing
//#include <chrono>
//#include <ctime>

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
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
    int parentIdx;

public:
    TreeNode(const glm::vec3& p, const float& d, const int& i) : point(p), influenceDist(d), parentIdx(i) {}
    ~TreeNode() {}
    inline bool InfluencesPoint(const glm::vec3& p) const {
        return glm::length(p - point) < influenceDist;
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
        return glm::length(p - point) < killDist;
    }
};

int main() {
    // Test Mesh Loading
    Mesh m = Mesh();
    m.LoadFromFile("OBJs/plane.obj");

    // GLFW Window Setup
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Trees", NULL, NULL);
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

    glViewport(0, 0, 800, 600);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Test loop vectorization. Note: using the compiler flags, this stuff only seems to compile in Release Mode
    // Needed flags: /O2 /Qvec-report:1 (can also use report:2)
    // Source: https://software.intel.com/en-us/articles/a-guide-to-auto-vectorization-with-intel-c-compilers

    // Stores the point positions: currently a list of floats. I need to include glm or eigen
    // Is it faster to initialize a vector of points with # and value and then set the values, or to push_back values onto an empty list
    // Answer to that: https://stackoverflow.com/questions/32199388/what-is-better-reserve-vector-capacity-preallocate-to-size-or-push-back-in-loo
    // Best options seem to be preallocate or emplace_back with reserve
    const unsigned int numPoints = 200;
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

    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Create points
    // Unfortunately, we can't really do any memory preallocating because we don't actually know how many points will be included
    for (unsigned int i = 0; i < numPoints; ++i) {
        const glm::vec3 p = glm::vec3(dis(rng), dis(rng), /*dis(rng)*/ 0.0f);
        if ((p.x * p.x + p.y * p.y) > 0.15f /*p.y > 0.25f && (p.x * p.x + p.y * p.y) > 0.2f*/) {
            points.emplace_back(p);
            ++numPointsIncluded;
        }
    }

    // Create the AttractorPoints
    const float killDist = 0.3f;
    std::vector<AttractorPoint> attractorPoints = std::vector<AttractorPoint>();
    attractorPoints.reserve(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        attractorPoints.emplace_back(AttractorPoint(points[i], killDist));
    }

    // Create the TreeNode(s)
    const float branchLength = 0.1f;
    const float branchInflDist = 0.4f;
    std::vector<TreeNode> treeNodes = std::vector<TreeNode>();
    treeNodes.emplace_back(TreeNode(glm::vec3(0.0f, 0.0f, 0.0f), branchInflDist, -1));
    //treeNodes.emplace_back(TreeNode(glm::vec3(-0.1f, 0.0f, 0.0f), branchInflDist, -1));

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

    const unsigned int numIters = 8;
    int numTreeNodes = treeNodes.size(); // Update the number of tree nodes to run the algorithm on in the for loop

    for (unsigned int n = 0; n < numIters && attractorPoints.size() > 0; ++n) { // Run the algorithm a certain number of times, or if there are no attractor points
        // Create TreeNodes
        for (int ti = 0; ti < numTreeNodes; ++ti) { // Perform the algorithm for each tree node
            glm::vec3 accumDir = glm::vec3(0.0f); // Accumulate the direction of each influencing AttractorPoint
            bool numNearbyPoints = false; // Count number of nearby attractor points
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
                // Normalize the accumulated direction
                accumDir = glm::normalize(accumDir);

                // Create a new TreeNode
                treeNodes.emplace_back(TreeNode(treeNodePoint + accumDir * branchLength, branchInflDist, ti));
            }
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
        numTreeNodes = treeNodes.size();

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

    // Create points and indices for the tree branches
    std::vector<glm::vec3> pointsTreeBranch = std::vector<glm::vec3>(0);
    std::vector<unsigned int> indicesTreeBranch = std::vector<unsigned int>(0);
    int idxCounter = 0;
    for (int i = treeNodes.size() - 1; i > 0; --i) {
        const TreeNode& currTreeNode = treeNodes[i];
        const int parentIdx = currTreeNode.GetParentIndex();
        if (parentIdx != -1) {
            pointsTreeBranch.push_back(treeNodes[parentIdx].GetPoint()); // base of branch
            pointsTreeBranch.push_back(currTreeNode.GetPoint()); // branch end point
            indicesTreeBranch.emplace_back(idxCounter++);
            indicesTreeBranch.emplace_back(idxCounter++);
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    
    /*for (int i = 0; i < m.GetVertices().size(); i++) {
        std::cout << m.GetVertices()[i].pos.x << m.GetVertices()[i].pos.y << m.GetVertices()[i].pos.z << std::endl;
        std::cout << m.GetVertices()[i].nor.x << m.GetVertices()[i].nor.y << m.GetVertices()[i].nor.z << std::endl;
    }

    for (int i = 0; i < m.GetIndices().size(); i++) {
        std::cout << m.GetIndices()[i] << std::endl;
    }*/

    std::vector<unsigned int> idx = m.GetIndices();

    // Mesh buffers
    glBindVertexArray(VAO3);
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
    glBindVertexArray(0);

    glPointSize(2);
    glLineWidth(1);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindVertexArray(VAO);
        sp.use();
        glDrawElements(GL_POINTS, tempPtsIdx.size(), GL_UNSIGNED_INT, 0);
        
        glBindVertexArray(VAO2);
        sp2.use();
        glDrawElements(GL_LINES, indicesTreeBranch.size(), GL_UNSIGNED_INT, 0);

        /*glBindVertexArray(VAO3);
        sp3.use();
        glDrawElements(GL_TRIANGLES, idx.size(), GL_UNSIGNED_INT, 0);*/

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