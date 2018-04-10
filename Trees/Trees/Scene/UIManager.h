#pragma once

#include "imgui.h"
#include "imgui_impl_glfw_glad.h"

struct GLFWwindow;
class TreeApplication;

class UIManager {
public:
    UIManager(GLFWwindow* window) { ImguiSetup(window); }
    void ImguiSetup(GLFWwindow* window);
    void ImguiNewFrame() {
        ImGui_ImplGlfwGLAD_NewFrame();
    }
    void RenderImgui() {
        ImGui::Render();
        ImGui_ImplGlfwGLAD_RenderDrawData(ImGui::GetDrawData());
    }
    void DestroyImgui() {
        ImGui_ImplGlfwGLAD_Shutdown();
        ImGui::DestroyContext();
    }
    void HandleInput(TreeApplication& treeApp);
};
