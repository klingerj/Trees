#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS
#pragma once

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>

struct Bud;
struct AttractorPoint;

namespace TreeApp {
    void PerformSpaceColonizationParallel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints);
}
