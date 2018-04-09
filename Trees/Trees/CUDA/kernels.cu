
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include "../Scene/Tree.h"

#include <stdio.h>

// Note: this implementation uses the "nearestBudIdx" field differently than the CPU implementation. This is because on the GPU, we don't
// have access to the Tree's "branches" vector, so we just make the bud idx the index in the one big array of buds, not the index in the vector
// of buds for a certain branch.
__global__ void kernSetNearestBudForAttractorPoints(Bud* dev_buds, const int numBuds, AttractorPoint* dev_attrPts, const int numAttractorPoints, int* dev_mutex) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= numBuds) {
        return;
    }

    Bud& currentBud = dev_buds[index];

    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        for (int ap = 0; ap < numAttractorPoints; ++ap) {
            AttractorPoint& currentAttrPt = dev_attrPts[ap];
            glm::vec3 budToPtDir = currentAttrPt.point - currentBud.point; // Use current axillary or terminal bud
            const float budToPtDist2 = glm::length2(budToPtDir);
            budToPtDir = glm::normalize(budToPtDir);
            const float dotProd = glm::dot(budToPtDir, currentBud.naturalGrowthDir);
            if (budToPtDist2 < (14.0f * currentBud.internodeLength * currentBud.internodeLength) && dotProd > std::abs(COS_THETA_SMALL)) {
                int* mutex = dev_mutex + ap;
                bool isSet = false;
                do {
                    isSet = (atomicCAS(mutex, 0, 1) == 0);
                    if (isSet) {
                        if (budToPtDist2 < currentAttrPt.nearestBudDist2) {
                            currentAttrPt.nearestBudDist2 = budToPtDist2;
                            currentAttrPt.nearestBudIdx = index;
                    }
                    *mutex = 0;
                    }
                } while (!isSet);
            }
        }
    }
}

__global__ void kernSpaceCol(Bud* dev_buds, const int numBuds, AttractorPoint* dev_attrPts, const int numAttractorPoints, int* dev_mutex) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= numBuds) {
        return;
    }
    
    Bud& currentBud = dev_buds[index];

    // Space Colonization
    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        for (int ap = 0; ap < numAttractorPoints; ++ap) {
            AttractorPoint& currentAttrPt = dev_attrPts[ap];
            glm::vec3 budToPtDir = currentAttrPt.point - currentBud.point; // Use current axillary or terminal bud
            const float budToPtDist2 = glm::length2(budToPtDir);
            budToPtDir = glm::normalize(budToPtDir);
            const float dotProd = glm::dot(budToPtDir, currentBud.naturalGrowthDir);
            if (budToPtDist2 < (14.0f * currentBud.internodeLength * currentBud.internodeLength) && dotProd > std::abs(COS_THETA_SMALL)) {
                        if (currentAttrPt.nearestBudIdx == index) {
                            currentBud.optimalGrowthDir += budToPtDir;
                            ++currentBud.numNearbyAttrPts;
                            currentBud.environmentQuality = 1.0f;
                        }
            }
        }
    }
    currentBud.optimalGrowthDir = currentBud.numNearbyAttrPts > 0 ? glm::normalize(currentBud.optimalGrowthDir) : glm::vec3(0.0f);
}

cudaError_t RunSpaceColonizationKernel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints) {
    cudaError_t cudaStatus;

    Bud* dev_buds = 0;
    AttractorPoint* dev_attrPts = 0;
    int* dev_mutex = 0;

    const int blockSize = 32;

    // Device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Cuda Malloc
    cudaStatus = cudaMalloc((void**)&dev_buds, numBuds * sizeof(Bud));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_attrPts, numAttractorPoints * sizeof(AttractorPoint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mutex, numAttractorPoints * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Cuda memcpy
    cudaStatus = cudaMemcpy(dev_buds, buds, numBuds * sizeof(Bud), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_attrPts, attractorPoints, numAttractorPoints * sizeof(AttractorPoint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaMemset(dev_mutex, 0, numAttractorPoints * sizeof(int));

    // Run the kernel
    kernSetNearestBudForAttractorPoints << < (numBuds + blockSize - 1) / blockSize, blockSize >> > (dev_buds, numBuds, dev_attrPts, numAttractorPoints, dev_mutex);
    kernSpaceCol << < (numBuds + blockSize - 1) / blockSize, blockSize >> > (dev_buds, numBuds, dev_attrPts, numAttractorPoints, dev_mutex);

    // Cuda Memcpy the Bud info back to the CPU
    cudaStatus = cudaMemcpy(buds, dev_buds, numBuds * sizeof(Bud), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error: {
    cudaFree(dev_buds);
    cudaFree(dev_attrPts);
    cudaFree(dev_mutex);
}

    return cudaStatus;
}

void TreeApp::PerformSpaceColonizationParallel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints) {
    cudaError_t cudaStatus = RunSpaceColonizationKernel(buds, numBuds, attractorPoints, numAttractorPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }
}
