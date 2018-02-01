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
public:
    std::vector<TreeNode> children; // make this private later. Also not very memory coalescent
    TreeNode(const glm::vec3& p, const float& d) : point(p), influenceDist(d) {}
    ~TreeNode() {}
    inline bool InfluencesPoint(const glm::vec3& p) const {
        return glm::length(p - point) < influenceDist;
    }
};

class AttractorPoint {
private:
    glm::vec3 point; // Point in world space
    float killDist; // Radius for removal
public:
    AttractorPoint(const glm::vec3& p, const float& d) : point(p), killDist(d) {}
    ~AttractorPoint() {}
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
    const unsigned int numPoints = 10;
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
        if ((p.x * p.x + p.y * p.y + p.z * p.z) < 0.75f) { // sphere sdf, radius of 0.75f
            points.emplace_back(p);
            ++numPointsIncluded;
        }
    }

    // Create the AttractorPoints
    std::vector<AttractorPoint> attractorPoints = std::vector<AttractorPoint>();
    attractorPoints.reserve(numPointsIncluded);
    for (int i = 0; i < numPointsIncluded; ++i) {
        attractorPoints.emplace_back(AttractorPoint(points[i], 0.15f));
    }

    // Create the TreeNode(s)
    const float branchLength = 0.3f;
    std::vector<TreeNode> treeNodes = std::vector<TreeNode>();
    treeNodes.emplace_back(TreeNode(glm::vec3(0.0f, 0.0f, 0.0f), 0.5f)); // center of screen

    // Run the tree algorithm
    //TODO


    // Create indices for the attractor points
    std::vector<unsigned int> indices = std::vector<unsigned int>(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        indices[i] = i;
    }

    /// First triangle / points

    // Create and compile vert/frag shaders
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
    const char* fragmentShaderSource =  "#version 450 core\n\nlayout(location = 0) in vec3 fPos;\nout vec4 FragColor;\n\nvoid main() {\nFragColor = vec4(vec3(1)/*abs(fPos)*/, 1);\n}";
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // Shader compilation success check
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Shader Program
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
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    /// Array/Buffer Objects
    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // VAO Binding
    glBindVertexArray(VAO);

    // VBO Binding
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * numPointsIncluded, points.data(), GL_STATIC_DRAW);

    // EBO Binding
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), indices.data(), GL_STATIC_DRAW);

    // Attribute linking
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0); // not really sure what this does

    glPointSize(2);

    // Render loop
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_POINTS, indices.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();
    return 0;
}