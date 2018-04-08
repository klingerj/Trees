#pragma once

#include "imgui.h"
#include "imgui_impl_glfw_glad.h"

struct GLFWwindow;
struct TreeParameters;

class UIManager {
public:
    UIManager(GLFWwindow* window) { ImguiSetup(window); }
    void ImguiSetup(GLFWwindow* window);
    inline void ImguiNewFrame() {
        ImGui_ImplGlfwGLAD_NewFrame();
    }
    inline void RenderImgui() {
        ImGui::Render();
        ImGui_ImplGlfwGLAD_RenderDrawData(ImGui::GetDrawData());
    }
    inline void DestroyImgui() {
        ImGui_ImplGlfwGLAD_Shutdown();
        ImGui::DestroyContext();
    }
    void HandleInput(TreeParameters& treeParams);
};
