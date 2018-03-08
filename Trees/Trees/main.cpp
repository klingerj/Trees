#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "pcg_random.hpp"

#include "OpenGL\ShaderProgram.h"
#include "Scene\Mesh.h"
#include "Scene\Camera.h"

#include <iostream>
#include <vector>
#include <random>

// For performance analysis / timing
#include <chrono>
#include <ctime>

#define GLM_FORCE_RADIANS
#define VIEWPORT_WIDTH_INITIAL 800
#define VIEWPORT_HEIGHT_INITIAL 600

// draw branches as 3d geometry vs gl lines for branches
//#define CUBES

#define INITIAL_BRANCH_RADIUS 0.1f
#define BUD_OCCUPANCY_RADIUS 0.5f

// Used in space colonization
#define COS_THETA 0.70710678118f // cos(pi/4)
#define COS_THETA_SMALL 0.86602540378 // cos(pi6)

// For BH Model
#define ALPHA 1.2f // proportionality constant for resource flow computation
#define LAMBDA 0.52f

// For addition of new shoots
#define OPTIMAL_GROWTH_DIR_WEIGHT 0.1f
#define TROPISM_DIR_WEIGHT -0.2f
#define TROPISM_VECTOR glm::vec3(0.0f, -1.0f, 0.0f)

// The radius of the outermost branches on the tree
// this doesnt work!!!!
#define MINIMUM_BRANCH_RADIUS 1.0f
#define PIPE_EXPONENT 1.0f // somewhere between 2 and 3 usually according to the paper

#define NUM_ITERATIONS 3

// For 5-tree scene, eye and ref: glm::vec3(0.25f, 0.5f, 3.5f), glm::vec3(0.25f, 0.0f, 0.0f
Camera camera = Camera(glm::vec3(0.0f, 3.0f, 0.0f), 0.7853981634f, // 45 degrees vs 75 degrees
(float)VIEWPORT_WIDTH_INITIAL / VIEWPORT_HEIGHT_INITIAL, 0.01f, 2000.0f, 10.0f, 0.0f, 5.0f);
const float camMoveSensitivity = 0.001f;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    camera.SetAspect((float)width / height);
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

class AttractorPoint {
private:
    glm::vec3 point; // Point in world space
    float killDist; // Radius for removal

public:
    AttractorPoint(const glm::vec3& p, const float& d) : point(p), killDist(d), nearestBudDist2(9999999.0f), nearestBudBranchIdx(-1), nearestBudIdx(-1) {}
    ~AttractorPoint() {}
    inline const glm::vec3 GetPoint() const {
        return point;
    }
    inline bool IsKilledBy(const glm::vec3& p) const {
        return glm::distance2(p, point) < (killDist * killDist);
    }
    // Make these private
    // Newer paper variables
    float nearestBudDist2; // how close the nearest bud is that has this point in its perception volume, squared
    int nearestBudBranchIdx; // index in the array of the branch of that bud ^^
    int nearestBudIdx; // index in the array the bud of that branch ^^
    // For the indices: -1 indicates, not yet set, -2 indicates the terminal bud of the branch
};

/// New proposed code structure for tree growth simulation

enum BUD_FATE {
    DORMANT,
    FORMED_BRANCH,
    FORMED_FLOWER,
    ABORT // TODO: when does this happen
};

enum BUD_TYPE {
    TERMINAL, // at the end of a branch
    LATERAL // along the sides of a branch
};

// Store any information relevant to a particular bud
struct Bud {
    glm::vec3 point;
    glm::vec3 naturalGrowthDir; // Growth direction of this bud. Use Golden Angle (137.5 degrees) for lateral buds.
    glm::vec3 optimalGrowthDir; // optimal growth direction computing during space colonization
    float occupancyRadius; // Radius about the bud in which attractor points are removed
    float environmentQuality; // In space colonization, this is a binary 0 or 1
    float accumEnvironmentQuality; // Using Borchert-Honda Model, indicates the accumulated amount of resources reaching this bud
    float resourceBH; // amount of available resource reaching this Bud using the BH Model
    int formedBranchIndex; // If this bud's fate is FORMED_BRANCH, this is the index in the Tree's list of branches of that formed branch. -1 o.w.
    float internodeLength;
    float branchRadius;
    int numNearbyAttrPts;
    BUD_TYPE type;
    BUD_FATE fate;

    // Constructor: to allow use with emplace_back() on vectors
    Bud(const glm::vec3& p, const glm::vec3& nd, const glm::vec3& d, const float r, const float q, const float aq, const float re,
        const int i, const float l, const float br, const int n, BUD_TYPE t, BUD_FATE f) :
        point(p), naturalGrowthDir(nd), optimalGrowthDir(d), occupancyRadius(r), environmentQuality(q), accumEnvironmentQuality(aq), resourceBH(re),
        formedBranchIndex(i), internodeLength(l), branchRadius(br), numNearbyAttrPts(n), type(t), fate(f) {}
};

// Wraps up necessary information regarding a tree branch.
class TreeBranch {
    friend class Tree;
private: // TODO: make the terminal bud just be the last bud in the one list of buds. Too complicated differentiating it in the code.
    std::vector<Bud> buds; // List of buds. Last bud is always the terminal bud.
    glm::vec3 growthDirection; // World space direction in which this branch is oriented
    float radius; // Branch radius. Computed using pipe model
    unsigned int axisOrder; // Order n (0, 1, ..., n) of this axis. Original trunk of a tree is 0, each branch supported by this branch has order 1, etc
    int prevBranchIndex; // Index of the branch supporting this one in the 

public:
    TreeBranch(const glm::vec3& p, const glm::vec3& d, const int ao, const int bi) :
        growthDirection(d), radius(INITIAL_BRANCH_RADIUS), axisOrder(ao), prevBranchIndex(bi) {
        buds = std::vector<Bud>();
        buds.reserve(4); // Reserve memory beforehand so we are less likely to have to resize the array later on. Performance test this.
        buds.emplace_back(p, glm::vec3(growthDirection), glm::vec3(0.0f), BUD_OCCUPANCY_RADIUS, 0.0f, 0.0f, 0.0f, -1, 0.1f, 0.0f, 0, TERMINAL, DORMANT); // add the terminal bud for this branch. Applies a prelim internode length (tweak, TODO) ***
        // maybe make the reserve value a function of the iterations, later iterations will probably be shorter than an early branch that has
        // been around for awhile?
    }
    inline const std::vector<Bud>& GetBuds() const {
        return buds;
    }
    // Adds a certain number of axillary buds to the list of buds, starting at the index just before the terminal bud
    void AddAxillaryBuds(const Bud& sourceBud, const int numBuds, const float internodeLength) {
        // Create a temporary list of Buds that will be inserted in this branch's list of buds
        std::vector<Bud> newBuds = std::vector<Bud>();

        // Axillary bud orientation: Golden angle of 137.5 about the growth axis
        glm::vec3 crossVec = (std::abs(glm::dot(growthDirection, WORLD_UP_VECTOR)) > 0.99f) ? glm::vec3(1.0f, 0.0f, 0.0f) : WORLD_UP_VECTOR; // avoid glm::cross returning a 0-vector
        const glm::quat branchQuat = glm::angleAxis(glm::radians(25.0f), glm::normalize(glm::cross(growthDirection, crossVec)));
        const glm::mat4 budRotMat = glm::toMat4(branchQuat);

        // Direction in which the bud itself is oriented
        glm::vec3 budGrowthDir = glm::normalize(glm::vec3(budRotMat * glm::vec4(growthDirection, 0.0f)));

        // Direction in which growth occurs
        const glm::vec3 newBudGrowthDir = glm::normalize(sourceBud.naturalGrowthDir + OPTIMAL_GROWTH_DIR_WEIGHT * sourceBud.optimalGrowthDir + TROPISM_DIR_WEIGHT * TROPISM_VECTOR);

        // buds will be inserted @ current terminal bud pos + (float)b * branchGrowthDir * internodeLength
        Bud& terminalBud = buds[buds.size() - 1]; // last bud is always the terminal bud
        for (int b = 0; b < numBuds; ++b) {
            // Account for golden angle here
            const glm::quat branchQuatGoldenAngle = glm::angleAxis(glm::radians(137.5f * (float)(buds.size() - 1 + b)), growthDirection);
            const glm::mat4 budRotMatGoldenAngle = glm::toMat4(branchQuatGoldenAngle);
            const glm::vec3 budGrowthGoldenAngle = glm::normalize(glm::vec3(budRotMatGoldenAngle * glm::vec4(budGrowthDir, 0.0f)));
            newBuds.emplace_back(terminalBud.point + (float)b * newBudGrowthDir * internodeLength, budGrowthGoldenAngle, glm::vec3(0.0f),
                BUD_OCCUPANCY_RADIUS, 0.0f, 0.0f, 0.0f, -1, internodeLength, 0.0f, 0, LATERAL, DORMANT);
        }
        // Update terminal bud position
        terminalBud.point = terminalBud.point + (float)(newBuds.size()) * growthDirection * internodeLength;
        terminalBud.internodeLength = internodeLength;
        buds.insert(buds.begin() + buds.size() - 1, newBuds.begin(), newBuds.end());
    }
};

// Wrap up branches into one Tree class. This class organizes the simulation functions
class Tree {
private:
    std::vector<TreeBranch> branches;
    inline void InitializeTree(const glm::vec3& p) { branches.emplace_back(TreeBranch(p, glm::vec3(0.0f, 1.0f, 0.0f), 0, -1)); } // Init a tree to be a single branch
public:
    Tree(const glm::vec3& p) {
        branches.reserve(65536); // Reserve a lot so we don't have to resize often. This vector will definitely expand a lot. Also, the code will crash without this due to some contiguous memory issue, probably.
        InitializeTree(p);
    }
    inline const std::vector<TreeBranch>& GetBranches() const {
        return branches;
    }
    void IterateGrowth(const int numIters, std::vector<AttractorPoint>& attractorPoints) {
        for (int n = 0; n < numIters; ++n) {

            std::cout << "Iteration #: " << n << std::endl;

            auto start = std::chrono::system_clock::now();
            PerformSpaceColonization(attractorPoints); // 1. Compute Q (presence of space/light) and optimal growth direction using space colonization
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            std::time_t end_time = std::chrono::system_clock::to_time_t(end);
            std::cout << "Elapsed time for Space Colonization: " << elapsed_seconds.count() << "s\n";
            
            start = std::chrono::system_clock::now();
            ComputeBHModel();                          // 2. Using BH Model, flow resource basipetally and then acropetally
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            end_time = std::chrono::system_clock::to_time_t(end);
            std::cout << "Elapsed time for Computing BH Model (both passes): " << elapsed_seconds.count() << "s\n";

            start = std::chrono::system_clock::now();
            AppendNewShoots();                         // 3. Add new shoots using the resource computed in previous step
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            end_time = std::chrono::system_clock::to_time_t(end);
            std::cout << "Elapsed time for Appending New Shoots: " << elapsed_seconds.count() << "s\n";

            start = std::chrono::system_clock::now();
            ResetState();                              // 4. Prepare all data to be iterated over again, e.g. set accumQ / resrouceBH for all buds back to 0
            end = std::chrono::system_clock::now();
            elapsed_seconds = end - start;
            end_time = std::chrono::system_clock::to_time_t(end);
            std::cout << "Elapsed time for State Resetting: " << elapsed_seconds.count() << "s\n\n";

            if (attractorPoints.size() == 0) { break; }
        }

        auto start = std::chrono::system_clock::now();
        ComputeBranchRadii();
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Elapsed time for Computing Branch Radii: " << elapsed_seconds.count() << "s\n";
    }
    void PerformSpaceColonization(std::vector<AttractorPoint>& attractorPoints) {
        // 1. Remove all attractor points that are too close to any bud
        for (int br = 0; br < branches.size(); ++br) {
            const std::vector<Bud>& buds = branches[br].buds;
            for (int bu = 0; bu < buds.size(); ++bu) {
                auto attrPtIter = attractorPoints.begin();
                while (attrPtIter != attractorPoints.end()) {
                    const Bud& currentBud = buds[bu];
                    const float budToPtDist = glm::length2(attrPtIter->GetPoint() - currentBud.point);
                    if (budToPtDist < 4.0f * currentBud.internodeLength * currentBud.internodeLength) { // 2x internode length - use distance squared
                        attrPtIter = attractorPoints.erase(attrPtIter); // This attractor point is close to the bud, remove it
                    }
                    else {
                        ++attrPtIter;
                    }
                }
            }
        }

        // 2. Compute the optimal growth direction for each bud
        for (int br = 0; br < branches.size(); ++br) {
            std::vector<Bud>& buds = branches[br].buds;
            const int numBuds = buds.size();
            for (int bu = 0; bu < numBuds; ++bu) {
                Bud& currentBud = buds[bu];
                auto attrPtIter = attractorPoints.begin();
                while (attrPtIter != attractorPoints.end()) { // if a bud is FORMED_BRANCH, do i not run the alg on it? probs just test, but need to add geometry correctly
                    if (currentBud.fate == DORMANT) { // early measure, aka buds wont grow after branching once. TODO also keep in mind that branching points store buds twice.
                        glm::vec3 budToPtDir = attrPtIter->GetPoint() - currentBud.point; // Use current lateral or terminal bud
                        const float budToPtDist2 = glm::length2(budToPtDir);
                        budToPtDir = glm::normalize(budToPtDir);
                        const float dotProd = glm::dot(budToPtDir, currentBud.naturalGrowthDir);
                        if (budToPtDist2 < (16.0f * currentBud.internodeLength * currentBud.internodeLength) && dotProd > std::abs(COS_THETA_SMALL)) { // 4x internode length - use distance squared
                            // Any given attractor point can only be perceived by one bud - the nearest one.
                            // If we end up find a bud closer to this attractor point than the previously recorded one,
                            // update the point accordingly and remove this attractor point's contribution from that bud's
                            // growth direction vector.
                            if (budToPtDist2 < attrPtIter->nearestBudDist2) {
                                attrPtIter->nearestBudDist2 = budToPtDist2;
                                if (attrPtIter->nearestBudBranchIdx != -1 && attrPtIter->nearestBudIdx != -1) {
                                    Bud& oldNearestBud = branches[attrPtIter->nearestBudBranchIdx].buds[attrPtIter->nearestBudIdx];
                                    glm::vec3& oldNearestBudDir = oldNearestBud.optimalGrowthDir * (float)oldNearestBud.numNearbyAttrPts;
                                    oldNearestBudDir -= budToPtDir;
                                    if (--oldNearestBud.numNearbyAttrPts > 0) {
                                        oldNearestBudDir = glm::normalize(oldNearestBudDir);
                                    }
                                    else {
                                        oldNearestBudDir = glm::vec3(0.0f);
                                    }
                                }
                                attrPtIter->nearestBudBranchIdx = br;
                                attrPtIter->nearestBudIdx = bu;
                                currentBud.optimalGrowthDir += budToPtDir;
                                ++currentBud.numNearbyAttrPts;
                            }
                        }
                    }
                    ++attrPtIter;
                }
                if (currentBud.numNearbyAttrPts > 0) {
                    currentBud.optimalGrowthDir = glm::normalize(currentBud.optimalGrowthDir);
                    currentBud.environmentQuality = 1.0f;
                }
            }
        }
    }
    void ComputeBHModel() { // Perform each pass of the BH Model for resource flow
        ComputeBHModelBasipetalPass(); // don't wrap this so much, just call the recursive functions. they're one liners anyway. TODO
        ComputeBHModelAcropetalPass();
    }

    // make this a non-member helper function in the cpp file
    float ComputeQAccumRecursive(TreeBranch& branch) {
        float accumQ = 0.0f;
        for (int bu = branch.buds.size() - 1; bu >= 0; --bu) { // iterate in reverse
            Bud& currentBud = branch.buds[bu];
            switch (currentBud.type) {
            case TERMINAL:
                accumQ += currentBud.environmentQuality;
                break;
            case LATERAL:
            {
                // Need to check whether this lateral bud has actually formed a branch
                switch (currentBud.fate) {
                case DORMANT:
                    accumQ += currentBud.environmentQuality;
                    break;
                case FORMED_BRANCH:
                    accumQ += ComputeQAccumRecursive(branches[currentBud.formedBranchIndex]);
                    break;
                case FORMED_FLOWER: // double check if we include the resource in this case TODO
                    accumQ += ComputeQAccumRecursive(branches[currentBud.formedBranchIndex]);
                    break;
                default: // includes ABORT case - ignore this bud
                    break;
                }
                break;
            }
            }
            currentBud.accumEnvironmentQuality = accumQ;
        }
        return accumQ;
    }
    void ComputeBHModelBasipetalPass() { // Compute the amount of resource reaching each internode (actually stored in bud above that internode)
        // Way 1:
        // Make a recursive function that takes in a particular branch, it should return the incoming Q at the base of the branch.
        // Should just be a for loop with one recursive call.
        // This is a little inefficient and I should eventually memoize the information so we don't have to recompute branches.
        ComputeQAccumRecursive(branches[0]); // ignore return value
    }

    // make this a non-member helper function in the cpp file
    void ComputeResourceFlowRecursive(TreeBranch& branch, float resource) {
        for (int bu = 0; bu < branch.buds.size(); ++bu) {
            Bud& currentBud = branch.buds[bu];
            // this all only applies to lateral buds. if terminal, just set the amount of resource
            // if this bud is dormant, set this bud's amount of resource to whatever the current value is
            // if this bud is formed flower or abort, set the amount of resource to 0
            // else, aka if this bud is formed branch, then set the resource amount for this branch to the main axis part of the
            // formula, and make an additional call on the branch at the appropriate index with the other portion of the eqn
            // in the formula: v is resource, Qm is the accumQ stored in buds[bu+1]. Ql is accumQ stored branches[currentBud.formedBranchIndex].buds[1].
            // Yes that is a hard coded index but there should be an invariant that any lateral bud that has no consecutive lateral buds is followed by
            // a terminal bud.
            // Note, deciding now that at branching points, a bud will be stored in both branches. It's a little more memory intensive, and I might
            // change this later, but right now it simplifies things a bit.
            switch (currentBud.type) {
            case TERMINAL:
                currentBud.resourceBH = resource;
                break;
            case LATERAL:
                switch (currentBud.fate) {
                case DORMANT:
                    currentBud.resourceBH = resource;
                    break;
                case FORMED_BRANCH: { // have to scope for nontrivial cases, apparently: https://stackoverflow.com/questions/10381144/error-c2361-initialization-of-found-is-skipped-by-default-label
                    TreeBranch& lateralBranch = branches[currentBud.formedBranchIndex];
                    const float Qm = branch.buds[bu + 1].accumEnvironmentQuality; // Q on main axis
                    const float Ql = lateralBranch.buds[1].accumEnvironmentQuality; // Q on lateral axis
                    const float denom = LAMBDA * Qm + (1.0f - LAMBDA) * Ql;
                    currentBud.resourceBH = resource * (LAMBDA * Qm) / denom; // formula for main axis
                    ComputeResourceFlowRecursive(lateralBranch, resource * (1.0f - LAMBDA) * Ql / denom); // Call this function on the lateral branch with the other formula
                    resource = currentBud.resourceBH; // resource reaching the remaining buds in this branch have the attenuated resource
                    break;
                }
                case FORMED_FLOWER:
                    currentBud.resourceBH = 0.0f;
                    break;
                default: // FORMED_FLOWER or ABORT
                    currentBud.resourceBH = 0.0f;
                    break;

                }
                break;
            }
        }
    }
    void ComputeBHModelAcropetalPass() { // Recursive like basipetal pass, but will definitely need to memoize or something
        // pass in the first branch and the base amount of resource (v)
        ComputeResourceFlowRecursive(branches[0], (branches[0].buds[0].type == TERMINAL) ? branches[0].buds[0].accumEnvironmentQuality * 1.0f : branches[0].buds[0].accumEnvironmentQuality * ALPHA);
    }

    // Determine where to grow new shoots and their length(s)
    void AppendNewShoots() {
        // for each branch, for each bud, compute floor(v). if that's > 0, check if its a terminal bud. if yes, just extend the current axis.
        // if its a lateral bud, do the hard invariant stuff.
        // need to compute the new growth axis. use golden angle for lateral buds
        const int numBranches = branches.size();
        for (int br = 0; br < numBranches; ++br) {
            TreeBranch& currentBranch = branches[br];
            std::vector<Bud>& buds = currentBranch.buds;
            const int numBuds = buds.size();
            for (int bu = 0; bu < numBuds; ++bu) {
                Bud& currentBud = buds[bu];
                const int numMetamers = std::floor(currentBud.resourceBH);
                const float metamerLength = currentBud.resourceBH / (float)numMetamers * 0.35f; // TODO remove fake scale *************
                switch (currentBud.type) {
                case TERMINAL: {
                    if (numMetamers > 0) {
                        currentBranch.AddAxillaryBuds(currentBud, numMetamers, metamerLength);
                    }
                    break;
                }
                case LATERAL: {
                    if (numMetamers > 0 && currentBud.fate == DORMANT) {
                        TreeBranch newBranch = TreeBranch(currentBud.point, currentBud.naturalGrowthDir, branches[br].axisOrder + 1, br);
                        newBranch.AddAxillaryBuds(currentBud, numMetamers, metamerLength);
                        branches.emplace_back(newBranch);
                        currentBud.fate = FORMED_BRANCH;
                        currentBud.formedBranchIndex = branches.size() - 1;
                    }
                    break;
                }
                }
            }
        }
    }

    // Using the "pipe model" described in the paper, compute the radius of each branch
    float ComputeBranchRadiiRecursive(TreeBranch& branch) {
        float branchRadius = MINIMUM_BRANCH_RADIUS;
        for (int bu = branch.buds.size() - 1; bu >= 0; --bu) { // iterate in reverse
            Bud& currentBud = branch.buds[bu];
            switch (currentBud.type) {
            case TERMINAL:
                //branchRadius = MINIMUM_BRANCH_RADIUS;
                break;
            case LATERAL:
            {
                switch (currentBud.fate) {
                case DORMANT:
                    //branchRadius += std::pow(currentBud.branchRadius, PIPE_EXPONENT);
                    // do nothing I think, only add at branching points
                    break;
                case FORMED_BRANCH:
                    branchRadius = std::pow(branchRadius, PIPE_EXPONENT) + std::pow(ComputeBranchRadiiRecursive(branches[currentBud.formedBranchIndex]), PIPE_EXPONENT);
                    break;
                case FORMED_FLOWER:
                    // don't change radius for now?
                    break;
                default: // includes the ABORT case of bud fate
                    break;
                }
                break;
            }
            }
            currentBud.branchRadius = branchRadius;
            //std::cout << "BRANCH RADIUS SET TO: " << branchRadius << std::endl;
        }
        return branchRadius;
    }
    void ComputeBranchRadii() {
        ComputeBranchRadiiRecursive(branches[0]); // ignore return value
    }
    void ResetState() {
        for (int br = 0; br < branches.size(); ++br) {
            std::vector<Bud>& buds = branches[br].buds;
            for (int bu = 0; bu < buds.size(); ++bu) {
                Bud& currentBud = buds[bu];
                currentBud.accumEnvironmentQuality = 0.0f;
                currentBud.environmentQuality = 0.0f;
                currentBud.numNearbyAttrPts = 0;
                currentBud.optimalGrowthDir = glm::vec3(0.0f);
                currentBud.resourceBH = 0.0f;
            }
        }
    }
};

// Wrap the overall application up into a TreeApplication class eventually that controls both GL calls and Tree objects
/*class TreeApplication {

};*/





// OLD
// Tree growth code - will need to be refactored TO ANOTHER FILE PLS ALSO CANT TURN OFF CAPS LOCK RN SO
// this is now outdated **

class TreeNode {
private:
    glm::vec3 point; // Point in world space
    float influenceDist; // Radius of sphere of influence
    int parentIdx; // index of the parent of this Node, in the array of nodes. Will probably change
    std::vector<int> childrenIndices; // vector containing the indices of each child. Each child should have this node as its parent

public:
    TreeNode(const glm::vec3& p, const float& d, const int& i, const int& it) :
        point(p), influenceDist(d), parentIdx(i), optimalGrowthDir(glm::vec3(0.0f)), branchDir(glm::vec3(0.0f)), hasNearbyAttrPts(false), iter(it) {
        childrenIndices = std::vector<int>();
        childrenIndices.reserve(10); // reserve 10 children...this can be tweaked and should be performance analyzed
    }
    ~TreeNode() {}
    inline bool InfluencesPoint(const glm::vec3& p) const {
        return glm::distance2(p, point) < (influenceDist * influenceDist);
    }
    inline const glm::vec3 GetPoint() const {
        return point;
    }
    inline const int GetParentIndex() const {
        return parentIdx;
    }
    // Variables, I'm being lazy, make them private
    // Newer paper parameters
    glm::vec3 optimalGrowthDir;
    glm::vec3 branchDir;
    bool hasNearbyAttrPts;
    unsigned int iter; // what iteration this node was added during
};

int main() {
    // Test Mesh Loading
    //Mesh m = Mesh();
    //m.LoadFromFile("OBJs/plane.obj");

    // GLFW Window Setup
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

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

    // Test loop vectorization. Note: using the compiler flags, this stuff only seems to compile in Release Mode
    // Needed flags: /O2 /Qvec-report:1 (can also use report:2)
    // Source: https://software.intel.com/en-us/articles/a-guide-to-auto-vectorization-with-intel-c-compilers

    // Stores the point positions: currently a list of floats. I need to include glm or eigen
    // Is it faster to initialize a vector of points with # and value and then set the values, or to push_back values onto an empty list
    // Answer to that: https://stackoverflow.com/questions/32199388/what-is-better-reserve-vector-capacity-preallocate-to-size-or-push-back-in-loo
    // Best options seem to be preallocate or emplace_back with reserve
    const unsigned int numPoints = 100000;
    unsigned int numPointsIncluded = 0;
    std::vector<glm::vec3> points = std::vector<glm::vec3>();

    // Using PCG RNG: http://www.pcg-random.org/using-pcg-cpp.html

    // Make a random number engine
    pcg32 rng(101);

    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    auto start = std::chrono::system_clock::now();
    // Unfortunately, we can't really do any memory preallocating because we don't actually know how many points will be included
    for (unsigned int i = 0; i < numPoints; ++i) {
        const glm::vec3 p = glm::vec3(dis(rng) * 5.0f, dis(rng) * 5.0f, dis(rng) * 5.0f); // for big cube growth chamber: scales of 10, 20, 10
        if (glm::length(p) < 5.0f /*p.y > 0.2f*/ /*&& (p.x * p.x + p.y * p.y) > 0.2f*/) {
            points.emplace_back(p + glm::vec3(0.0f, 2.51f, 0.0f));
            ++numPointsIncluded;
        }
    }
    // Create the actual AttractorPoints
    const float killDist = 0.05f;
    std::vector<AttractorPoint> attractorPoints = std::vector<AttractorPoint>();
    attractorPoints.reserve(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        attractorPoints.emplace_back(AttractorPoint(points[i], killDist));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Attractor Point Generation: " << elapsed_seconds.count() << "s\n";

    // new tree generation
    Tree tree = Tree(glm::vec3(0.0f, 0.0f, 0.0f));

    start = std::chrono::system_clock::now();
    tree.IterateGrowth(NUM_ITERATIONS, attractorPoints);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Total Elapsed time for Tree Generation: " << elapsed_seconds.count() << "s\n";

    // Code for GL stuff

    // Create indices for the attractor points
    std::vector<unsigned int> indices = std::vector<unsigned int>(numPointsIncluded);
    for (unsigned int i = 0; i < numPointsIncluded; ++i) {
        indices[i] = i;
    }

    // Now using cubes / rectangular prisms, will be slightly faster and have less visual issues

    std::vector<glm::vec3> cubePoints;
    std::vector<glm::vec3> cubeNormals;
    std::vector<unsigned int> cubeIndices;

    start = std::chrono::system_clock::now();

    ///Positions
    //Front face
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, 1.0f));

    //Right face
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, 1.0f));

    //Left face
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, -1.0f));

    //Back face
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, -1.0f));

    //Top face
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, 1.0f, -1.0f));

    //Bottom face
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, 1.0f));
    cubePoints.emplace_back(glm::vec3(1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, -1.0f));
    cubePoints.emplace_back(glm::vec3(-1.0f, -1.0f, 1.0f));

    /// Normals
    //Front
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, 0, 1));
    }
    //Right
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(1, 0, 0));
    }
    //Left
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(-1, 0, 0));
    }
    //Back
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, 0, -1));
    }
    //Top
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, 1, 0));
    }
    //Bottom
    for (int i = 0; i < 4; ++i) {
        cubeNormals.emplace_back(glm::vec3(0, -1, 0));
    }

    /// Indices
    for (int i = 0; i < 6; i++) {
        cubeIndices.emplace_back(i * 4);
        cubeIndices.emplace_back(i * 4 + 1);
        cubeIndices.emplace_back(i * 4 + 2);
        cubeIndices.emplace_back(i * 4);
        cubeIndices.emplace_back(i * 4 + 2);
        cubeIndices.emplace_back(i * 4 + 3);
    }

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Cube Primitive Generation: " << elapsed_seconds.count() << "s\n";

    // End cube generation

    // GL code

    start = std::chrono::system_clock::now();

    // Vectors of stuff to render
    #define INTERNODE_CUBES

    std::vector<glm::vec3> budPoints = std::vector<glm::vec3>();
    std::vector<unsigned int> budIndices = std::vector<unsigned int>();
    std::vector<unsigned int> branchIndices = std::vector<unsigned int>(); // one gl line corresponds to one internode. These indices will use the budPoints vector.

    std::vector<glm::vec3> internodePoints = std::vector <glm::vec3>();
    std::vector<glm::vec3> internodeNormals = std::vector <glm::vec3>();
    std::vector<unsigned int> internodeIndices = std::vector <unsigned int>();

    // get the points vectors
    const std::vector<TreeBranch>& branches = tree.GetBranches();
    for (int br = 0; br < tree.GetBranches().size(); ++br) {
        const std::vector<Bud>& buds = branches[br].GetBuds();
        #ifndef INTERNODE_CUBES
        const unsigned int idxOffset = budPoints.size();
        #endif
        int bu;
        #ifdef INTERNODE_CUBES
        bu = 1;
        #else
        bu = 0;
        #endif;
        for (; bu < buds.size(); ++bu) {
            #ifdef INTERNODE_CUBES // create internode VBO
            const Bud& currentBud = buds[bu];
            const glm::vec3& internodeEndPoint = currentBud.point; // effectively, just the position of the bud at the end of the current internode
            glm::vec3 branchAxis = glm::normalize(internodeEndPoint - buds[bu - 1].point); // not sure if i store this direction somewhere

            // Back to rotation

            // orientation is messed up in certain cases still...
            const bool axesAreAligned = std::abs(glm::dot(branchAxis, WORLD_UP_VECTOR)) > 0.99f;
            glm::vec3 crossVec = axesAreAligned ? glm::vec3(1.0f, 0.0f, 0.0f) : WORLD_UP_VECTOR; // avoid glm::cross returning a nan or 0-vector
            const glm::vec3 axis = glm::normalize(glm::cross(crossVec, branchAxis));
            const float angle = std::acos(glm::dot(branchAxis, WORLD_UP_VECTOR));

            const glm::quat branchQuat = glm::angleAxis(angle, axis);
            glm::mat4 branchTransform = glm::toMat4(branchQuat); // initially just a rotation matrix, eventually stores the entire transformation

            // Compute the translation component - set to the the base branch point + 0.5 * internodeLength, placing the cylinder at the halfway point
            const glm::vec3 translation = internodeEndPoint - 0.5f * branchAxis * currentBud.internodeLength;

            // Create an overall transformation matrix of translation and rotation
            branchTransform = glm::translate(glm::mat4(1.0f), translation) * branchTransform * glm::scale(glm::mat4(1.0f), glm::vec3(currentBud.branchRadius * 0.01f, currentBud.internodeLength * 0.5f, currentBud.branchRadius * 0.01f));

            std::vector<glm::vec3> cubePointsTrans = std::vector<glm::vec3>();
            std::vector<glm::vec3> cubeNormalsTrans = std::vector<glm::vec3>();
            // Interleave VBO data or nah
            for (int i = 0; i < cubePoints.size(); ++i) {
                cubePointsTrans.emplace_back(glm::vec3(branchTransform * glm::vec4(cubePoints[i], 1.0f)));
                glm::vec3 transformedNormal = glm::normalize(glm::vec3(glm::inverse(glm::transpose(branchTransform)) * glm::vec4(cubeNormals[i], 0.0f)));
                cubeNormalsTrans.emplace_back(transformedNormal);
            }

            std::vector<unsigned int> cubeIndicesNew = std::vector<unsigned int>();
            for (int i = 0; i < cubeIndices.size(); ++i) {
                const unsigned int size = (unsigned int)internodePoints.size();
                cubeIndicesNew.emplace_back(cubeIndices[i] + size); // offset this set of indices by the # of positions. Divide by two bc it contains positions and normals
            }

            internodePoints.insert(internodePoints.end(), cubePointsTrans.begin(), cubePointsTrans.end());
            internodeNormals.insert(internodeNormals.end(), cubeNormalsTrans.begin(), cubeNormalsTrans.end());
            internodeIndices.insert(internodeIndices.end(), cubeIndicesNew.begin(), cubeIndicesNew.end());

            #else // old GL_LINES code
            if (bu < buds.size() - 1) { // proper indexing, just go with it
                branchIndices.emplace_back(bu + idxOffset);
                branchIndices.emplace_back(bu + 1 + idxOffset);
            }
            #endif
            budPoints.emplace_back(buds[bu].point);
        }
    }

    // create indices vector
    for (int i = 0; i < budPoints.size(); ++i) {
        budIndices.emplace_back(i);
    }

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for Tree VBO Info Creation: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();

    /// GL calls and drawing

    ShaderProgram sp = ShaderProgram("Shaders/point-vert.vert", "Shaders/point-frag.frag");
    ShaderProgram sp2 = ShaderProgram("Shaders/treeNode-vert.vert", "Shaders/treeNode-frag.frag");
    ShaderProgram sp3 = ShaderProgram("Shaders/mesh-vert.vert", "Shaders/mesh-frag.frag");

    // Array/Buffer Objects
    unsigned int VAO, VAO2, VAO3, VAO4;
    unsigned int VBO, VBO2, VBO3, VBO4;
    unsigned int EBO, EBO2, EBO3, EBO4;
    glGenVertexArrays(1, &VAO);
    glGenVertexArrays(1, &VAO2);
    glGenVertexArrays(1, &VAO3);
    glGenVertexArrays(1, &VAO4);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &VBO3);
    glGenBuffers(1, &VBO4);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &EBO2);
    glGenBuffers(1, &EBO3);
    glGenBuffers(1, &EBO4);

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



    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER, VBO2);

    #ifndef INTERNODE_CUBES
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * budPoints.size(), budPoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * budIndices.size(), budIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0); // pass positions
    glEnableVertexAttribArray(0);

    glBindVertexArray(VAO3);
    glBindBuffer(GL_ARRAY_BUFFER, VBO3);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * budPoints.size(), budPoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO3);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * branchIndices.size(), branchIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0); // pass positions
    glEnableVertexAttribArray(0);

    #else

    // Points
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * internodePoints.size(), internodePoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * internodeIndices.size(), internodeIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Normals
    glBindBuffer(GL_ARRAY_BUFFER, VBO3);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * internodeNormals.size(), internodeNormals.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO3);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * internodeIndices.size(), internodeIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(1);

    #endif

    // Bud Points
    glBindVertexArray(VAO4);
    glBindBuffer(GL_ARRAY_BUFFER, VBO4);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * budPoints.size(), budPoints.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO4);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * budIndices.size(), budIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Elapsed time for GL calls and ShaderPrograms and VAO/VBO/EBO Creation: " << elapsed_seconds.count() << "s\n";



    // This was TinyOBJ Debugging
    /*for (int i = 0; i < m.GetVertices().size(); i++) {
        std::cout << m.GetVertices()[i].pos.x << m.GetVertices()[i].pos.y << m.GetVertices()[i].pos.z << std::endl;
        std::cout << m.GetVertices()[i].nor.x << m.GetVertices()[i].nor.y << m.GetVertices()[i].nor.z << std::endl;
    }

    for (int i = 0; i < m.GetIndices().size(); i++) {
        std::cout << m.GetIndices()[i] << std::endl;
    }*/

    //std::vector<unsigned int> idx = m.GetIndices();

    // Mesh buffers
    /*glBindVertexArray(VAO3);
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
    glBindVertexArray(0);*/

    glPointSize(2);
    glLineWidth(1);
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_MULTISAMPLE);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Attractor Points
        /*glBindVertexArray(VAO);
        sp.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_POINTS, (GLsizei) tempPtsIdx.size(), GL_UNSIGNED_INT, 0);*/

        // old cubes / new bud points
        glBindVertexArray(VAO2);
        sp2.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        #ifndef INTERNODE_CUBES
        // draws bud points NOPE******
        //glDrawElements(GL_POINTS, (GLsizei)budIndices.size(), GL_UNSIGNED_INT, 0);
        #else
        // draw the internode geometry
        glDrawElements(GL_TRIANGLES, (GLsizei)internodeIndices.size(), GL_UNSIGNED_INT, 0);
        #endif

        glBindVertexArray(VAO3);
        // new tree branches
        sp3.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        #ifndef INTERNODE_CUBES
        // draw GL_LINES intenodes
        glDrawElements(GL_LINES, (GLsizei)branchIndices.size(), GL_UNSIGNED_INT, 0);
        #else
        // do nothing idk
        #endif

        /*glBindVertexArray(VAO4);
        sp.setCameraViewProj("cameraViewProj", camera.GetViewProj());
        glDrawElements(GL_POINTS, (GLsizei)budIndices.size(), GL_UNSIGNED_INT, 0);*/

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteVertexArrays(1, &VAO2);
    glDeleteVertexArrays(1, &VAO3);
    glDeleteVertexArrays(1, &VAO4);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &VBO2);
    glDeleteBuffers(1, &VBO3);
    glDeleteBuffers(1, &VBO4);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &EBO2);
    glDeleteBuffers(1, &EBO3);
    glDeleteBuffers(1, &EBO4);

    glfwTerminate();
    return 0;
}
