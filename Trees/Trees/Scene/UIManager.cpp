#include "UIManager.h"
#include "imgui.cpp"
#include "imgui_demo.cpp"
#include "imgui_draw.cpp"
#include "imgui_internal.h"
#include "imconfig.h"
#include "TreeApplication.h"

void UIManager::ImguiSetup(GLFWwindow* window) {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui_ImplGlfwGLAD_Init(window, true);
    ImGui::StyleColorsDark();
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
}

void UIManager::HandleInput(TreeApplication& treeApp) {
    ImGui::SliderFloat("Maximum Branch Radius", &treeApp.GetTreeParameters().maximumBranchRadius, 0.0f, 100.0f);
    ImGui::SliderInt("Maximum Branch Radius", &treeApp.GetTreeParameters().numSpaceColonizationIterations, 0, 10000);
    if (ImGui::Button("Regrow Tree")) {
        treeApp.GrowSelectedTreeIntoSelectedAttractorPointCloud();
    }
}
