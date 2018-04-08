#include "UIManager.h"
#include "imgui.cpp"
#include "imgui_demo.cpp"
#include "imgui_draw.cpp"
#include "imgui_internal.h"
#include "imconfig.h"
#include "Tree.h"

void UIManager::ImguiSetup(GLFWwindow* window) {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui_ImplGlfwGLAD_Init(window, true);
    ImGui::StyleColorsDark();
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
}

void UIManager::HandleInput(TreeParameters& treeParams) {
    ImGui::SliderFloat("Borchert-Honda Model Lambda (0 = Wide growth, 1 = Tall growth)", &treeParams.BHLambda, 0.0f, 1.0f);
    std::cout << treeParams.BHLambda << std::endl;
}
