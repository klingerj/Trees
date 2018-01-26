#include <glad\glad.h>
#include <GLFW\glfw3.h>
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

    // Test loop vectorization. Note: only seems to compile in Release Mode
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
    const unsigned int numPoints = 5000;
    unsigned numPointsIncluded = 0;
    std::vector<float> points = std::vector<float>(); // 1000 points with value 0

    // Random number generator. source taken from: http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Create points
    for (unsigned int i = 0; i < numPoints * 3; i += 3) {
        const float x = dis(gen) * 2.0f - 1.0f; // account for aspect ratio
        const float y = dis(gen) * 2.0f - 1.0f;
        const float z = dis(gen) * 2.0f - 1.0f;
        if (std::sqrt(x * x + y * y + z * z) < 0.75f) {
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
            numPointsIncluded++;
        }
    }

    // Create indices
    std::vector<unsigned int> indices = std::vector<unsigned int>(numPointsIncluded, 0);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        indices[i] = i;
    }

    /// First triangle / points

    // Create and compile vert/frag shaders
    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char* vertexShaderSource =
        "#version 450 core\n\nlayout(location = 0) in vec3 vPos;\nlayout(location = 0) out vec3 fPos;\n\nvoid main() {\nfPos = vPos;\ngl_Position = vec4(vPos * vec3(0.75, 1, 1), 1);\n}";
                                                                                                                                                                  // account for aspect ratio
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
    const char* fragmentShaderSource =  "#version 450 core\n\nlayout(location = 0) in vec3 fPos;\nout vec4 FragColor;\n\nvoid main() {\nFragColor = vec4(abs(fPos), 1);\n}";
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
    
    // VAO Binding
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // VBO Binding
    unsigned int VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * numPoints, points.data(), GL_STATIC_DRAW);

    // EBO Binding
    unsigned int EBO;
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), indices.data(), GL_STATIC_DRAW);

    // Attribute linking
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
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