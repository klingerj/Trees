#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>

#include "../Scene/Tree.h"

namespace TreeApp {
    void PerformSpaceColonization(Bud* buds, glm::vec3* attractorPoints);
}