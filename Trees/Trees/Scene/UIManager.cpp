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
    ImGui::SliderFloat("Internode Scale", &treeApp.GetTreeParameters().internodeScale, 0.0f, 10.0f);
    ImGui::SliderFloat("Minimum Branch Radius", &treeApp.GetTreeParameters().minimumBranchRadius, 0.0f, 100.0f);
    ImGui::SliderFloat("Maximum Branch Radius", &treeApp.GetTreeParameters().maximumBranchRadius, 0.0f, 100.0f);
    ImGui::SliderInt("Num Space Col Iterations", &treeApp.GetTreeParameters().numSpaceColonizationIterations, 0, 10000);
    ImGui::SliderInt("Num Attr Pts to Gen", &treeApp.GetTreeParameters().numAttractorPointsToGenerate, 0, 5000000);
    //ImGui::Checkbox("Enable Debug Output", &treeApp.GetTreeParameters().enableDebugOutput);
    if (ImGui::Button("Iterate Tree")) {
        treeApp.IterateSelectedTreeInSelectedAttractorPointCloud();
    }
    if (ImGui::Button("Regrow Tree")) {
        treeApp.RegrowSelectedTreeInSelectedAttractorPointCloud();
    }
    if (ImGui::Button("Add Attr Pt Cloud")) {
        treeApp.AddAttractorPointCloudToScene();
        treeApp.GetSelectedAttractorPointCloud().GeneratePoints(treeApp.GetTreeParameters().numAttractorPointsToGenerate);
        treeApp.GetTreeParameters().reconstructUniformGridOnGPU = true;
    }
    if (ImGui::Button("Show/Hide Current Attr Pt Cloud")) {
        treeApp.GetSelectedAttractorPointCloud().ToggleDisplay();
    }
}
