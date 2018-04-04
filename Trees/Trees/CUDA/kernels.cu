
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include "../Scene/Tree.h"

#include <stdio.h>

__global__ void kernSpaceCol(Bud* dev_buds, const int numBuds, AttractorPoint* dev_attrPts, const int numAttractorPoints) {
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
            if (budToPtDist2 < (12.0f * currentBud.internodeLength * currentBud.internodeLength) && dotProd > std::abs(COS_THETA_SMALL)) {
                if (budToPtDist2 < currentAttrPt.nearestBudDist2) {
                    currentAttrPt.nearestBudDist2 = budToPtDist2;
                    if (currentAttrPt.nearestBudBranchIdx != -1 && currentAttrPt.nearestBudIdx != -1) {
                        Bud& oldNearestBud = branches[attrPtIter->nearestBudBranchIdx].buds[attrPtIter->nearestBudIdx];
                        glm::vec3& oldNearestBudDir = oldNearestBud.optimalGrowthDir * (float)oldNearestBud.numNearbyAttrPts;
                        oldNearestBudDir -= budToPtDir;
                        if (--oldNearestBud.numNearbyAttrPts > 0) {
                            oldNearestBudDir = glm::normalize(oldNearestBudDir);
                        } else {
                            oldNearestBudDir = glm::vec3(0.0f);
                        }
                    }
                    currentAttrPt.nearestBudBranchIdx = ;
                    currentAttrPt.nearestBudIdx = ;
                    currentBud.optimalGrowthDir += budToPtDir;
                    ++currentBud.numNearbyAttrPts;
                }
            }
        }
    }

    if (currentBud.numNearbyAttrPts > 0) {
        currentBud.optimalGrowthDir = glm::normalize(currentBud.optimalGrowthDir);
        currentBud.environmentQuality = 1.0f;
    }
}

cudaError_t RunSpaceColonizationKernel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints) {
    cudaError_t cudaStatus;

    Bud* dev_buds = 0;
    AttractorPoint* dev_attrPts = 0;

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

    // Run the kernel
    const int blockSize = 32;
    kernSpaceCol << < (numBuds + blockSize - 1) / blockSize, blockSize >> > (dev_buds, numBuds, dev_attrPts, numAttractorPoints);

    // Cuda Memcpy the Bud info back to the CPU
    cudaStatus = cudaMemcpy(buds, dev_buds, numBuds * sizeof(Bud), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_buds);
    cudaFree(dev_attrPts);

    return cudaStatus;
}

void TreeApp::PerformSpaceColonizationParallel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints) {
    cudaError_t cudaStatus = RunSpaceColonizationKernel(buds, numBuds, attractorPoints, numAttractorPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }
}
