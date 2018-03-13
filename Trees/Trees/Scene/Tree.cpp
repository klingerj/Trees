#include "Tree.h"
#include <iostream>

/// TreeBranch Class Functions

void TreeBranch::AddAxillaryBuds(const Bud& sourceBud, const int numBuds, const float internodeLength) {
    // Create a temporary list of Buds that will be inserted in this branch's list of buds
    std::vector<Bud> newBuds = std::vector<Bud>();

    // Axillary bud orientation: Golden angle of 137.5 about the growth axis
    glm::vec3 crossVec = (std::abs(glm::dot(growthDirection, WORLD_UP_VECTOR)) > 0.99f) ? glm::vec3(1.0f, 0.0f, 0.0f) : WORLD_UP_VECTOR; // avoid glm::cross returning a nan or 0-vector
    const glm::quat branchQuat = glm::angleAxis(glm::radians(25.0f), glm::normalize(glm::cross(growthDirection, crossVec)));
    const glm::mat4 budRotMat = glm::toMat4(branchQuat);

    // Direction in which the bud itself is oriented
    glm::vec3 budGrowthDir = glm::normalize(glm::vec3(budRotMat * glm::vec4(growthDirection, 0.0f)));

    // Direction in which growth occurs
    const glm::vec3 newShootGrowthDir = glm::normalize(sourceBud.naturalGrowthDir + OPTIMAL_GROWTH_DIR_WEIGHT * sourceBud.optimalGrowthDir + TROPISM_DIR_WEIGHT * TROPISM_VECTOR);

    // Buds will be inserted @ current terminal bud pos + (float)b * branchGrowthDir * internodeLength
    Bud& terminalBud = buds[buds.size() - 1]; // last bud is always the terminal bud
    for (int b = 0; b < numBuds; ++b) {
        // Account for golden angle here
        const glm::quat branchQuatGoldenAngle = glm::angleAxis(glm::radians(137.5f * (float)(buds.size() - 1 + b)), growthDirection);
        const glm::mat4 budRotMatGoldenAngle = glm::toMat4(branchQuatGoldenAngle);
        const glm::vec3 budGrowthGoldenAngle = glm::normalize(glm::vec3(budRotMatGoldenAngle * glm::vec4(budGrowthDir, 0.0f)));
        newBuds.emplace_back(terminalBud.point + (float)b * newShootGrowthDir * internodeLength, budGrowthGoldenAngle, glm::vec3(0.0f),
            BUD_OCCUPANCY_RADIUS, 0.0f, 0.0f, 0.0f, -1, internodeLength, 0.0f, 0, AXILLARY, DORMANT);
    }
    // Update terminal bud position
    terminalBud.point = terminalBud.point + (float)(newBuds.size()) * growthDirection * internodeLength;
    terminalBud.internodeLength = internodeLength;
    buds.insert(buds.begin() + buds.size() - 1, newBuds.begin(), newBuds.end());
}


/// Tree Class Functions

void Tree::IterateGrowth(const int numIters, std::vector<AttractorPoint>& attractorPoints) {
    for (int n = 0; n < numIters; ++n) {

        std::cout << "Iteration #: " << n << std::endl;

        auto start = std::chrono::system_clock::now();
        PerformSpaceColonization(attractorPoints); // 1. Compute Q (presence of space/light) and optimal growth direction using space colonization
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::time_t end_time = std::chrono::system_clock::to_time_t(end);
        std::cout << "Elapsed time for Space Colonization: " << elapsed_seconds.count() << "s\n";

        start = std::chrono::system_clock::now();
        ComputeBHModelBasipetalPass();             // 2. Using BH Model, flow resource basipetally and then acropetally
        ComputeBHModelAcropetalPass();
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
void Tree::PerformSpaceColonization(std::vector<AttractorPoint>& attractorPoints) {
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
        const unsigned int numBuds = (unsigned int)buds.size();
        for (unsigned int bu = 0; bu < numBuds; ++bu) {
            Bud& currentBud = buds[bu];
            auto attrPtIter = attractorPoints.begin();
            while (attrPtIter != attractorPoints.end()) {
                if (currentBud.fate == DORMANT) { // TODO find out if buds can grow @ FORMED_BRANCH fate
                    glm::vec3 budToPtDir = attrPtIter->GetPoint() - currentBud.point; // Use current axillary or terminal bud
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

float Tree::ComputeQAccumRecursive(TreeBranch& branch) {
    float accumQ = 0.0f;
    for (int bu = (int)branch.buds.size() - 1; bu >= 0; --bu) {
        Bud& currentBud = branch.buds[bu];
        switch (currentBud.type) {
        case TERMINAL:
            accumQ += currentBud.environmentQuality;
            break;
        case AXILLARY:
        {
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
void Tree::ComputeBHModelBasipetalPass() {
    // TODO: This is a little inefficient and I should eventually memoize the information so we don't have to recompute branches.
    ComputeQAccumRecursive(branches[0]); // ignore return value
}

// make this a non-member helper function in the cpp file
void Tree::ComputeResourceFlowRecursive(TreeBranch& branch, float resource) {
    for (int bu = 0; bu < branch.buds.size(); ++bu) {
        Bud& currentBud = branch.buds[bu];
        switch (currentBud.type) {
        case TERMINAL:
            currentBud.resourceBH = resource;
            break;
        case AXILLARY:
            switch (currentBud.fate) {
            case DORMANT:
                currentBud.resourceBH = resource;
                break;
                // Have to scope w/ brackets for nontrivial cases, apparently: https://stackoverflow.com/questions/10381144/error-c2361-initialization-of-found-is-skipped-by-default-label
            case FORMED_BRANCH: {
                TreeBranch& axillaryBranch = branches[currentBud.formedBranchIndex];
                const float Qm = branch.buds[bu + 1].accumEnvironmentQuality; // Q on main axis
                const float Ql = axillaryBranch.buds[1].accumEnvironmentQuality; // Q on axillary axis
                const float denom = LAMBDA * Qm + (1.0f - LAMBDA) * Ql;
                currentBud.resourceBH = resource * (LAMBDA * Qm) / denom; // formula for main axis
                ComputeResourceFlowRecursive(axillaryBranch, resource * (1.0f - LAMBDA) * Ql / denom); // call this function on the axillary branch with the other formula
                resource = currentBud.resourceBH; // Resource reaching the remaining buds in this branch have the attenuated resource
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
void Tree::ComputeBHModelAcropetalPass() { // Recursive like basipetal pass, but will definitely need to memoize or something
                                     // pass in the first branch and the base amount of resource (v)
    ComputeResourceFlowRecursive(branches[0], (branches[0].buds[0].type == TERMINAL) ? branches[0].buds[0].accumEnvironmentQuality * 1.0f : branches[0].buds[0].accumEnvironmentQuality * ALPHA);
}

// Determine whether to grow new shoots and their length(s)
void Tree::AppendNewShoots() {
    const unsigned int numBranches = (unsigned int)branches.size();
    for (unsigned int br = 0; br < numBranches; ++br) {
        TreeBranch& currentBranch = branches[br];
        std::vector<Bud>& buds = currentBranch.buds;
        const unsigned int numBuds = (unsigned int)buds.size();
        for (unsigned int bu = 0; bu < numBuds; ++bu) {
            Bud& currentBud = buds[bu];
            const int numMetamers = static_cast<int>(std::floor(currentBud.resourceBH));
            const float metamerLength = currentBud.resourceBH / (float)numMetamers * 0.35f; // TODO remove fake scale *************
            switch (currentBud.type) {
            case TERMINAL: {
                if (numMetamers > 0) {
                    currentBranch.AddAxillaryBuds(currentBud, numMetamers, metamerLength);
                }
                break;
            }
            case AXILLARY: {
                if (numMetamers > 0 && currentBud.fate == DORMANT) {
                    TreeBranch newBranch = TreeBranch(currentBud.point, currentBud.naturalGrowthDir, branches[br].axisOrder + 1, br);
                    newBranch.AddAxillaryBuds(currentBud, numMetamers, metamerLength);
                    branches.emplace_back(newBranch);
                    currentBud.fate = FORMED_BRANCH;
                    currentBud.formedBranchIndex = (int)branches.size() - 1;
                }
                break;
            }
            }
        }
    }
}

// Using the "pipe model" described in the paper, compute the radius of each branch
float Tree::ComputeBranchRadiiRecursive(TreeBranch& branch) {
    float branchRadius = MINIMUM_BRANCH_RADIUS;
    for (int bu = (int)branch.buds.size() - 1; bu >= 0; --bu) {
        Bud& currentBud = branch.buds[bu];
        switch (currentBud.type) {
        case TERMINAL:
            break;
        case AXILLARY:
        {
            switch (currentBud.fate) {
            case DORMANT:
                //branchRadius += std::pow(currentBud.branchRadius, PIPE_EXPONENT);
                // do nothing I think, only add at branching points
                break;
            case FORMED_BRANCH:
                branchRadius = std::pow(std::pow(branchRadius, PIPE_EXPONENT) + std::pow(ComputeBranchRadiiRecursive(branches[currentBud.formedBranchIndex]), PIPE_EXPONENT), 1.0f / PIPE_EXPONENT);
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
    }
    return branchRadius;
}
void Tree::ComputeBranchRadii() {
    ComputeBranchRadiiRecursive(branches[0]); // ignore return value
}
void Tree::ResetState() {
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