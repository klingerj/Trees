#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"

#include "OpenGL/ShaderProgram.h"
#include "Scene/Mesh.h"
#include "Scene/Camera.h"
#include "Scene/Tree.h"
#include "Scene/Globals.h"
#include "Scene/TreeApplication.h"
#include "Scene/UIManager.h"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <ctime>

Camera camera = Camera(glm::vec3(0.0f, 1.63f, 0.0f), 0.7853981634f, // 45 degrees vs 75 degrees
(float)VIEWPORT_WIDTH_INITIAL / VIEWPORT_HEIGHT_INITIAL, 0.01f, 2000.0f, 0.0f, -31.74f, 5.4f);
const float camMoveSensitivity = 0.03f;


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    camera.SetAspect((float)width / height);
}

// Based on https://stackoverflow.com/questions/45130391/opengl-get-cursor-coordinate-on-mouse-click-in-c
void mouse_click_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        std::cout << "Cursor Position at (" << xpos << ", " << ypos << ")" << std::endl;
    }
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
    glfwSetMouseButtonCallback(window, mouse_click_callback);

    UIManager uiMgr = UIManager(window);

    // App creation and initialization

    TreeApplication treeApp = TreeApplication();
    treeApp.AddTreeToScene();

    ShaderProgram sp  = ShaderProgram("Shaders/point-vert.vert", "Shaders/point-frag.frag");
    ShaderProgram sp2 = ShaderProgram("Shaders/tree-vert.vert", "Shaders/tree-frag.frag");
    ShaderProgram sp3 = ShaderProgram("Shaders/mesh-vert.vert", "Shaders/mesh-frag.frag");

    // Array/Buffer Objects
    unsigned int VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

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
        uiMgr.ImguiNewFrame();
        uiMgr.HandleInput(treeApp);

        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Draw attractor points with drawable class
        sp.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        treeApp.DrawAttractorPointClouds(sp);
        
        sp2.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        treeApp.DrawTrees(sp2);

        uiMgr.RenderImgui();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    treeApp.DestroyTrees();
    treeApp.DestroyAttractorPointClouds();

    glfwTerminate();
    return 0;
}
