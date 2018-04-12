
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include "../Scene/Tree.h"

#include <stdio.h>

// Note: this implementation uses the "nearestBudIdx" field differently than the CPU implementation. This is because on the GPU, we don't
// have access to the Tree's "branches" vector, so we just make the bud idx the index in the one big array of buds, not the index in the vector
// of buds for a certain branch.
__global__ void kernSetNearestBudForAttractorPoints(Bud* dev_buds, const glm::vec3& gridMin, int gridResolution, float inverseCellWidth, int numBuds,
                                                    AttractorPoint* dev_attrPts, const int numAttractorPoints, int* dev_mutex, int* gridCellStartIndices,
                                                    int* gridCellEndIndices) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= numBuds) {
        return;
    }

    Bud& currentBud = dev_buds[index];
    const glm::vec3& budPosLocalToGrid = currentBud.point - gridMin;
    const glm::vec3 index3D = floor(budPosLocalToGrid * inverseCellWidth);

    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                for (int z = -1; z <= 1; ++z) {
                    const glm::vec3 currentGridIndex = index3D + glm::vec3(x, y, z);

                    if (((((int)currentGridIndex.x) >= 0 && ((int)currentGridIndex.x) < gridResolution) &&
                        (((int)currentGridIndex.y) >= 0 && ((int)currentGridIndex.y) < gridResolution)) &&
                        (((int)currentGridIndex.z) >= 0 && ((int)currentGridIndex.z) < gridResolution)) {
                        int index1D = gridIndex3Dto1D(currentGridIndex.x, currentGridIndex.y, currentGridIndex.z, gridResolution);
                        for (int g = gridCellStartIndices[index1D]; g <= gridCellEndIndices[index1D]; ++g) {
                            if (g < 0) break;
                            AttractorPoint& currentAttrPt = dev_attrPts[g];
                            glm::vec3 budToPtDir = currentAttrPt.point - currentBud.point;
                            const float budToPtDist2 = glm::length2(budToPtDir);
                            budToPtDir = glm::normalize(budToPtDir);
                            const float dotProd = glm::dot(budToPtDir, currentBud.naturalGrowthDir);
                            if (budToPtDist2 < (14.0f * currentBud.internodeLength * currentBud.internodeLength) && dotProd > std::abs(COS_THETA_SMALL)) {
                                int* mutex = dev_mutex + g;
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
            }
        }



        /*for (int ap = 0; ap < numAttractorPoints; ++ap) {
            AttractorPoint& currentAttrPt = dev_attrPts[ap];
            glm::vec3 budToPtDir = currentAttrPt.point - currentBud.point;
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
        }*/
    }
}

__global__ void kernSpaceCol(Bud* dev_buds, const glm::vec3& gridMin, int gridResolution, float inverseCellWidth, int numBuds,
                             AttractorPoint* dev_attrPts, const int numAttractorPoints, int* dev_mutex, int* gridCellStartIndices,
                             int* gridCellEndIndices) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= numBuds) {
        return;
    }
    
    Bud& currentBud = dev_buds[index];

    const glm::vec3& budPosLocalToGrid = currentBud.point - gridMin;
    const glm::vec3 index3D = floor(budPosLocalToGrid * inverseCellWidth);

    // Space Colonization
    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                for (int z = -1; z <= 1; ++z) {
                    const glm::vec3 currentGridIndex = index3D + glm::vec3(x, y, z);

                    if (((((int)currentGridIndex.x) >= 0 && ((int)currentGridIndex.x) < gridResolution) &&
                        (((int)currentGridIndex.y) >= 0 && ((int)currentGridIndex.y) < gridResolution)) &&
                        (((int)currentGridIndex.z) >= 0 && ((int)currentGridIndex.z) < gridResolution)) {
                        int index1D = gridIndex3Dto1D(currentGridIndex.x, currentGridIndex.y, currentGridIndex.z, gridResolution);
                        for (int g = gridCellStartIndices[index1D]; g <= gridCellEndIndices[index1D]; ++g) {
                            if (g < 0) break;
                            AttractorPoint& currentAttrPt = dev_attrPts[g];
                            glm::vec3 budToPtDir = currentAttrPt.point - currentBud.point;
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
                }
            }
        }


        /*for (int ap = 0; ap < numAttractorPoints; ++ap) {
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
        }*/
    }
    currentBud.optimalGrowthDir = currentBud.numNearbyAttrPts > 0 ? glm::normalize(currentBud.optimalGrowthDir) : glm::vec3(0.0f);
}

// Uniform Grid Implementation functions
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
    return z + y * gridResolution + x * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int numAttrPts, int gridResolution,
    glm::vec3 gridMin, float inverseCellWidth,
    AttractorPoint* attrPts, int* attrPtIndices, int* gridIndices) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numAttrPts) {
        return;
    }
    glm::vec3 index3D = floor((attrPts[index].point - gridMin) * inverseCellWidth);
    int index1D = gridIndex3Dto1D(index3D.x, index3D.y, index3D.z, gridResolution);
    gridIndices[index] = index1D;
    attrPtIndices[index] = index;
}

__global__ void kernMakeDataMemoryCoherent(int numAttrPts, int* attrPtIndices,
    AttractorPoint* attrPts, AttractorPoint* attrPts_memCoherent) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numAttrPts) {
        return;
    }
    attrPts_memCoherent[index] = attrPts[attrPtIndices[index]];
}

__global__ void kernIdentifyCellStartEnd(int numAttrPts, int *gridCellIndices,
    int* gridCellStartIndices, int* gridCellEndIndices) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int gridIdx = gridCellIndices[index];
    int gridIdxNext = gridCellIndices[index + 1];

    if (index == 0) { 
        gridCellStartIndices[gridIdxNext] = 0;
        return;
    }
    if (index == numAttrPts - 1) {
        gridCellEndIndices[gridIdx] = index;
        return;
    }

    if (index > numAttrPts - 1) {
        return;
    }

    if (gridIdx != gridIdxNext) {
        gridCellEndIndices[gridIdx] = index;
        gridCellStartIndices[gridIdxNext] = index + 1;
    }
}

cudaError_t RunSpaceColonizationKernel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints) {
    cudaError_t cudaStatus;

    Bud* dev_buds = 0;
    AttractorPoint* dev_attrPts = 0;
    AttractorPoint* dev_attrPts_memCoherent = 0;
    int* dev_mutex = 0;
    int* dev_attrPtIndices; // indices of each attractor point (0, 1, ..., n)
    int* dev_gridCellIndices; // grid cell index of each attractor point
    int* dev_gridCellStartIndices; // start index of a grid cell
    int* dev_gridCellEndIndices; // end index of a grid cell

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

    cudaStatus = cudaMalloc((void**)&dev_attrPts_memCoherent, numAttractorPoints * sizeof(AttractorPoint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mutex, numAttractorPoints * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_attrPtIndices, numAttractorPoints * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_gridCellIndices, numAttractorPoints * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_gridCellStartIndices, numAttractorPoints * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_gridCellEndIndices, numAttractorPoints * sizeof(int));
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
    cudaMemset(dev_gridCellStartIndices, -1, numAttractorPoints * sizeof(int));
    cudaMemset(dev_gridCellEndIndices, -1, numAttractorPoints * sizeof(int));

    cudaThreadSynchronize();

    // fix params
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numAttractorPoints, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // Sorting with thrust
    thrust::device_ptr<int> dev_thrust_gridcell_indices(dev_gridCellIndices);
    thrust::device_ptr<int> dev_thrust_attrpt_indices(dev_attrPtIndices);
    thrust::sort_by_key(dev_thrust_gridcell_indices, dev_thrust_gridcell_indices + numAttractorPoints, dev_thrust_attrpt_indices);

    kernMakeDataMemoryCoherent << <fullBlocksPerGrid, blockSize >> > (numAttractorPoints, dev_attrPtIndices, dev_pos, dev_attrPts_memCoherent);
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numAttractorPoints, dev_gridCellIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

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
    cudaFree(dev_attrPts_memCoherent);
    cudaFree(dev_mutex);
    cudaFree(dev_gridCellIndices);
    cudaFree(dev_gridCellStartIndices);
    cudaFree(dev_gridCellEndIndices);
}

    return cudaStatus;
}

void TreeApp::PerformSpaceColonizationParallel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints) {
    cudaError_t cudaStatus = RunSpaceColonizationKernel(buds, numBuds, attractorPoints, numAttractorPoints);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }
}
