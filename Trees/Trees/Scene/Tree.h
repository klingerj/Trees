#pragma once


#include "glm\glm.hpp"
#include "glm/gtx/norm.hpp"

#include "Globals.h"

#include <vector>
#include <chrono>
#include <ctime>


// This class is only here temporarily

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







/// User-defined Parameters for the growth simulation

// For Space Colonization
#define INITIAL_BRANCH_RADIUS 0.1f
#define INITIAL_BUD_INTERNODE_RADIUS 0.05f
#define COS_THETA 0.70710678118f // cos(pi/4)
#define COS_THETA_SMALL 0.86602540378 // cos(pi6)

// For BH Model
#define ALPHA 1.0f // proportionality constant for resource flow computation
#define LAMBDA 0.5f

// For Addition of new shoots
#define OPTIMAL_GROWTH_DIR_WEIGHT 0.1f
#define TROPISM_DIR_WEIGHT 0.2f
#define TROPISM_VECTOR glm::vec3(0.0f, -1.0f, 0.0f)

// For branch radius computation
#define MINIMUM_BRANCH_RADIUS 0.1f // Radius of outermost branches
#define PIPE_EXPONENT 2.8f // somewhere between 2 and 3 usually according to the paper

/// Definition of structures

enum BUD_FATE {
    DORMANT,
    FORMED_BRANCH,
    FORMED_FLOWER,
    ABORT // TODO: when does this happen?
};

enum BUD_TYPE {
    TERMINAL, // at the end of a branch
    AXILLARY // along the sides of a branch
};

// Store any information relevant to a particular bud
struct Bud {
    glm::vec3 point;
    glm::vec3 naturalGrowthDir; // Growth direction of this bud. Use Golden Angle (137.5 degrees) for axillary buds.
    glm::vec3 optimalGrowthDir; // optimal growth direction computing during space colonization
    float environmentQuality; // In space colonization, this is a binary 0 or 1
    float accumEnvironmentQuality; // Using Borchert-Honda Model, indicates the accumulated amount of resources reaching this bud
    float resourceBH; // amount of available resource reaching this Bud using the BH Model
    int formedBranchIndex; // If this bud's fate is FORMED_BRANCH, this is the index in the Tree's list of branches of that formed branch. -1 o.w.
    float internodeLength;
    float branchRadius;
    int numNearbyAttrPts;
    BUD_TYPE type;
    BUD_FATE fate;

    // Constructor: to allow use with emplace_back() for std::vectors
    Bud(const glm::vec3& p, const glm::vec3& nd, const glm::vec3& d, const float q, const float aq, const float re,
        const int i, const float l, const float br, const int n, BUD_TYPE t, BUD_FATE f) :
        point(p), naturalGrowthDir(nd), optimalGrowthDir(d), environmentQuality(q), accumEnvironmentQuality(aq), resourceBH(re),
        formedBranchIndex(i), internodeLength(l), branchRadius(br), numNearbyAttrPts(n), type(t), fate(f) {}
};

// Wraps up necessary information regarding a tree branch.
class TreeBranch {
    friend class Tree;
private:
    std::vector<Bud> buds; // List of buds. Last bud is always the terminal bud.
    glm::vec3 growthDirection; // World space direction in which this branch is oriented
    float radius; // Branch radius. Computed using pipe model
    unsigned int axisOrder; // Order n (0, 1, ..., n) of this axis. Original trunk of a tree is 0, each branch supported by this branch has order 1, etc
    int prevBranchIndex; // Index of the branch supporting this one in the 

public:
    TreeBranch(const glm::vec3& p, const glm::vec3& d, const int ao, const int bi) :
        growthDirection(d), radius(INITIAL_BRANCH_RADIUS), axisOrder(ao), prevBranchIndex(bi) {
        buds = std::vector<Bud>();
        buds.reserve(8); // Reserve memory beforehand so we are less likely to have to resize the array later on. Performance test this.
        buds.emplace_back(p, glm::vec3(growthDirection), glm::vec3(0.0f), 0.0f, 0.0f, 0.0f, -1, INITIAL_BUD_INTERNODE_RADIUS, 0.0f, 0, TERMINAL, DORMANT); // add the terminal bud for this branch. Applies a prelim internode length (tweak, TODO)
    }
    inline const std::vector<Bud>& GetBuds() const { return buds; }
    inline const int GetAxisOrder() const { return axisOrder; }
    // Adds a certain number of axillary buds to the list of buds, starting at the index just before the terminal bud
    void AddAxillaryBuds(const Bud& sourceBud, const int numBuds, const float internodeLength);
};

// Wrap up branches into one Tree class. This class also organizes the simulation functions
class Tree {
private:
    std::vector<TreeBranch> branches;
    inline void InitializeTree(const glm::vec3& p) { branches.emplace_back(TreeBranch(p, glm::vec3(0.0f, 1.0f, 0.0f), 0, -1)); } // Initialize a tree to be a single branch
public:
    Tree(const glm::vec3& p) {
        branches.reserve(65536); // Reserve a lot so we don't have to resize often. This vector will definitely expand a lot. Also, the code will crash without this due to some contiguous memory issue, probably. TODO fix this?
        InitializeTree(p);
    }
    inline const std::vector<TreeBranch>& GetBranches() const {
        return branches;
    }
    void IterateGrowth(const int numIters, std::vector<AttractorPoint>& attractorPoints);
    void PerformSpaceColonization(std::vector<AttractorPoint>& attractorPoints);
    float ComputeQAccumRecursive(TreeBranch & branch);
    void ComputeBHModelBasipetalPass();
    void ComputeResourceFlowRecursive(TreeBranch & branch, float resource);
    void ComputeBHModelAcropetalPass();
    void AppendNewShoots();
    float ComputeBranchRadiiRecursive(TreeBranch & branch);
    void ComputeBranchRadii();
    void ResetState();
};
