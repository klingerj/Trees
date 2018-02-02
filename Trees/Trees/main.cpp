#include "glad\glad.h"
#include "GLFW\glfw3.h"
#include "glm\glm.hpp"
#include <iostream>
#include <vector>
#include <random>
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
    // GLFW Window Setup
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Trees", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Initialize Glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, 800, 600);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Test loop vectorization. Note: using the compiler flags, this stuff only seems to compile in Release Mode
    // Needed flags: /O2 /Qvec-report:1 (can also use report:2)
    // Source: https://software.intel.com/en-us/articles/a-guide-to-auto-vectorization-with-intel-c-compilers

    // first populate the list
    /*auto list = std::vector<int>(1000000000, 0);
    const unsigned int listSize = list.size();
    for (unsigned int i = 0; i < listSize; ++i) {
        list[i] = i;
    }

    // This should vectorize too
    auto start = std::chrono::system_clock::now();
    for (unsigned int i = 0; i < listSize; ++i) {
        list[i] = list[i] + 1;
    }
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";*/

    // Stores the point positions: currently a list of floats. I need to include glm or eigen
    // Is it faster to initialize a vector of points with # and value and then set the values, or to push_back values onto an empty list
    // Answer to that: https://stackoverflow.com/questions/32199388/what-is-better-reserve-vector-capacity-preallocate-to-size-or-push-back-in-loo
    // Best options seem to be preallocate or emplace_back with reserve
    const unsigned int numPoints = 10000;
    unsigned int numPointsIncluded = 0;
    std::vector<glm::vec3> points = std::vector<glm::vec3>();

    // Random number generator. source taken from: http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Create points
    // Unfortunately, we can't really do any memory preallocating because we don't actually know how many points will be included
    for (unsigned int i = 0; i < numPoints; ++i) {
        const glm::vec3 p = glm::vec3(dis(gen) * 2.0f - 1.0f, dis(gen) * 2.0f - 1.0f, dis(gen) * 2.0f - 1.0f);
        if ((p.x * p.x + p.y * p.y) > 0.2f /*p.y > 0.25f && (p.x * p.x + p.y * p.y) > 0.2f*/) {
            points.emplace_back(p);
            ++numPointsIncluded;
        }
    }

    // Create the AttractorPoints
    const float killDist = 0.4f;
    std::vector<AttractorPoint> attractorPoints = std::vector<AttractorPoint>();
    attractorPoints.reserve(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        attractorPoints.emplace_back(AttractorPoint(points[i], killDist));
    }

    // Create the TreeNode(s)
    const float branchLength = 0.1f;
    const float branchInflDist = 0.45f;
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
    const unsigned int numIters = 12;
    int numTreeNodes = treeNodes.size(); // Update the number of tree nodes to run the algorithm on in the for loop

    for (unsigned int n = 0; n < numIters && attractorPoints.size() > 0; ++n) { // Run the algorithm a certain number of times, or if there are no attractor points
        // Create TreeNodes
        for (int ti = 0; ti < numTreeNodes; ++ti) { // Perform the algorithm for each tree node
            glm::vec3 accumDir = glm::vec3(0.0f); // Accumulate the direction of each influencing AttractorPoint
            unsigned int numNearbyPoints = 0; // Count number of nearby attractor points
            const TreeNode& currTreeNode = treeNodes[ti];

            for (unsigned int pi = 0; pi < attractorPoints.size(); ++pi) {
                const glm::vec3& attrPoint = attractorPoints[pi].GetPoint();
                if (currTreeNode.InfluencesPoint(attrPoint)) {
                    accumDir += attrPoint - currTreeNode.GetPoint();
                    ++numNearbyPoints;
                }
            }
            if (numNearbyPoints > 0) {
                // Normalize the accumulated direction
                accumDir = glm::normalize(accumDir);

                // Create a new TreeNode
                treeNodes.emplace_back(TreeNode(treeNodes[ti].GetPoint() + accumDir * branchLength, branchInflDist, ti));
            }
        }

        // Kill attractor points that need to be killed
        // https://stackoverflow.com/questions/347441/erasing-elements-from-a-vector

        std::vector<AttractorPoint>::iterator attrPtIter = attractorPoints.begin();

        while (attrPtIter != attractorPoints.end()) {
            for (unsigned int ti = 0; ti < numTreeNodes; ++ti) { // size does NOT include the newly created tree nodes
                if (attrPtIter->IsKilledBy(treeNodes[ti].GetPoint())) {
                    attrPtIter = attractorPoints.erase(attrPtIter); // crash here occasionally
                    break;
                }
            }
            ++attrPtIter;
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
    std::cout << "Number of attractor points: " << numPointsIncluded << std::endl;
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


    /// First triangle / points

    // Create and compile vert/frag shaders

    // For the AttractorPoints
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char* vertexShaderSource =
        "#version 450 core\n\nlayout(location = 0) in vec3 vPos;\nlayout(location = 0) out vec3 fPos;\n\nvoid main() {\nfPos = vPos;\ngl_Position = vec4(vPos * vec3(600.0 / 800.0, 1, 1), 1);\n}";
                                                                                                                                               // account for aspect ratio...why am i dividing, not multiplying?
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // Shader compilation success check
    int  success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fragmentShaderSource =  "#version 450 core\n\nlayout(location = 0) in vec3 fPos;\nout vec4 FragColor;\n\nvoid main() {\nFragColor = vec4(vec3(1, 0, 0), 1);\n}";
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // Shader compilation success check
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Fragment shader for Tree Branches - just draws as green
    unsigned int fragmentShader2;
    fragmentShader2 = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fragmentShaderSource2 = "#version 450 core\n\nlayout(location = 0) in vec3 fPos;\nout vec4 FragColor;\n\nvoid main() {\nFragColor = vec4(abs(fPos), 1);\n}";
    glShaderSource(fragmentShader2, 1, &fragmentShaderSource2, NULL);
    glCompileShader(fragmentShader2);
    // Shader compilation success check
    glGetShaderiv(fragmentShader2, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader2, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Shader Program
    // For AttractorPoints
    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    // For Tree Branches
    unsigned int shaderProgram2;
    shaderProgram2 = glCreateProgram();
    glAttachShader(shaderProgram2, vertexShader);
    glAttachShader(shaderProgram2, fragmentShader2);
    glLinkProgram(shaderProgram2);
    glGetProgramiv(shaderProgram2, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram2, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glDeleteShader(fragmentShader2);
    
    /// Array/Buffer Objects
    unsigned int VAO, VAO2;
    unsigned int VBO, VBO2;
    unsigned int EBO, EBO2;
    glGenVertexArrays(1, &VAO);
    glGenVertexArrays(1, &VAO2);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &EBO2);

    // VAO Binding
    glBindVertexArray(VAO);

    // VBO Binding
    // Points
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numPointsIncluded, points.data(), GL_STATIC_DRAW);
    // EBO Binding
    // Points
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), indices.data(), GL_STATIC_DRAW);
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

    glPointSize(2);
    glLineWidth(1);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindVertexArray(VAO);
        glUseProgram(shaderProgram);
        glDrawElements(GL_POINTS, indices.size(), GL_UNSIGNED_INT, 0);
        
        glBindVertexArray(VAO2);
        glUseProgram(shaderProgram2);
        glDrawElements(GL_LINES, indicesTreeBranch.size(), GL_UNSIGNED_INT, 0);

        // Temporary but draw the tree node points
        /*glUseProgram(shaderProgram);
        glDrawElements(GL_POINTS, indicesTreeBranch.size(), GL_UNSIGNED_INT, 0);*/

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &VBO2);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &EBO2);

    glfwTerminate();
    return 0;
}