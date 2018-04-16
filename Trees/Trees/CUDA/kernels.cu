
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include "../Scene/Tree.h"

#include <stdio.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

// Uniform grid for attractor points
int* dev_attrPtIndices = 0; // indices of each attractor point (0, 1, ..., n)
int* dev_gridCellIndices = 0; // grid cell index of each attractor point
int* dev_gridCellStartIndices = 0; // start index of a grid cell
int* dev_gridCellEndIndices = 0; // end index of a grid cell

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
}

__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
    return z + y * gridResolution + x * gridResolution * gridResolution;
}

__global__ void kernMarkAttractorPointsAsRemoved(Bud* dev_buds, const glm::vec3 gridMin, const int gridResolution, const float inverseCellWidth, const int numBuds,
    AttractorPoint* dev_attrPts_memCoherent, const int numAttractorPoints, int* dev_mutex, int* gridCellStartIndices,
    int* gridCellEndIndices) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numBuds) {
        return;
    }
    Bud& currentBud = dev_buds[index];

    const glm::vec3 budPosLocalToGrid = currentBud.point - gridMin;
    const glm::vec3 index3D = glm::floor(budPosLocalToGrid * inverseCellWidth);
    const int lookupRadius = (int)glm::ceil(3.74165738677f * currentBud.internodeLength * inverseCellWidth); // sqrt(14) as used in space col nearby point lookup

    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        for (int x = -lookupRadius; x <= lookupRadius; ++x) {
            for (int y = -lookupRadius; y <= lookupRadius; ++y) {
                for (int z = -lookupRadius; z <= lookupRadius; ++z) {
                    const glm::vec3 currentGridIndex = index3D + glm::vec3(x, y, z);

                    if (((((int)currentGridIndex.x) >= 0 && ((int)currentGridIndex.x) < gridResolution) &&
                        (((int)currentGridIndex.y) >= 0 && ((int)currentGridIndex.y) < gridResolution)) &&
                        (((int)currentGridIndex.z) >= 0 && ((int)currentGridIndex.z) < gridResolution)) {
                        int index1D = gridIndex3Dto1D(currentGridIndex.x, currentGridIndex.y, currentGridIndex.z, gridResolution);
                        for (int g = gridCellStartIndices[index1D]; g <= gridCellEndIndices[index1D]; ++g) {
                            if (g < 0) { break; }
                            AttractorPoint& currentAttrPt = dev_attrPts_memCoherent[g];
                            glm::vec3 budToPtDir = currentAttrPt.point - currentBud.point;
                            const float budToPtDist2 = glm::length2(budToPtDir);
                            budToPtDir = glm::normalize(budToPtDir);
                            const float dotProd = glm::dot(budToPtDir, currentBud.naturalGrowthDir);
                            const float budToPtDist = glm::length2(currentAttrPt.point - currentBud.point);
                            if (budToPtDist < 5.1f * currentBud.internodeLength * currentBud.internodeLength) { // ~2x internode length - use distance squared
                                currentAttrPt.removed = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Note: this implementation uses the "nearestBudIdx" field differently than the CPU implementation. This is because on the GPU, we don't
// have access to the Tree's "branches" vector, so we just make the bud idx the index in the one big array of buds, not the index in the vector
// of buds for a certain branch.
__global__ void kernSetNearestBudForAttractorPoints(Bud* dev_buds, const glm::vec3 gridMin, const int gridResolution, const float inverseCellWidth, const int numBuds,
                                                    AttractorPoint* dev_attrPts_memCoherent, const int numAttractorPoints, int* dev_mutex, int* gridCellStartIndices,
                                                    int* gridCellEndIndices) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numBuds) {
        return;
    }
    Bud& currentBud = dev_buds[index];
    
    const glm::vec3 budPosLocalToGrid = currentBud.point - gridMin;
    const glm::vec3 index3D = glm::floor(budPosLocalToGrid * inverseCellWidth);
    const int lookupRadius = (int)glm::ceil(3.74165738677f * currentBud.internodeLength * inverseCellWidth); // sqrt(14) as used below

    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        for (int x = -lookupRadius; x <= lookupRadius; ++x) {
            for (int y = -lookupRadius; y <= lookupRadius; ++y) {
                for (int z = -lookupRadius; z <= lookupRadius; ++z) {
                    const glm::vec3 currentGridIndex = index3D + glm::vec3(x, y, z);

                    if (((((int)currentGridIndex.x) >= 0 && ((int)currentGridIndex.x) < gridResolution) &&
                        (((int)currentGridIndex.y) >= 0 && ((int)currentGridIndex.y) < gridResolution)) &&
                        (((int)currentGridIndex.z) >= 0 && ((int)currentGridIndex.z) < gridResolution)) {
                        int index1D = gridIndex3Dto1D(currentGridIndex.x, currentGridIndex.y, currentGridIndex.z, gridResolution);
                        for (int g = gridCellStartIndices[index1D]; g <= gridCellEndIndices[index1D]; ++g) {
                            if (g < 0) { break; }
                            AttractorPoint& currentAttrPt = dev_attrPts_memCoherent[g];
                            if (currentAttrPt.removed) { break; }
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
    }
}

__global__ void kernSpaceCol(Bud* dev_buds, const glm::vec3 gridMin, const int gridResolution, const float inverseCellWidth, const int numBuds,
    AttractorPoint* dev_attrPts_memCoherent, const int numAttractorPoints, int* dev_mutex, int* gridCellStartIndices,
    int* gridCellEndIndices) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numBuds) {
        return;
    }
    
    Bud& currentBud = dev_buds[index];
    
    const glm::vec3 budPosLocalToGrid = currentBud.point - gridMin;
    const glm::vec3 index3D = floor(budPosLocalToGrid * inverseCellWidth);
    const int lookupRadius = (int)glm::ceil(3.74165738677f * currentBud.internodeLength * inverseCellWidth); // sqrt(14) as used below

    // Space Colonization
    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        for (int x = -lookupRadius; x <= lookupRadius; ++x) {
            for (int y = -lookupRadius; y <= lookupRadius; ++y) {
                for (int z = -lookupRadius; z <= lookupRadius; ++z) {
                    const glm::vec3 currentGridIndex = index3D + glm::vec3(x, y, z);

                    if (((((int)currentGridIndex.x) >= 0 && ((int)currentGridIndex.x) < gridResolution) &&
                        (((int)currentGridIndex.y) >= 0 && ((int)currentGridIndex.y) < gridResolution)) &&
                        (((int)currentGridIndex.z) >= 0 && ((int)currentGridIndex.z) < gridResolution)) {
                        int index1D = gridIndex3Dto1D(currentGridIndex.x, currentGridIndex.y, currentGridIndex.z, gridResolution);
                        for (int g = gridCellStartIndices[index1D]; g <= gridCellEndIndices[index1D]; ++g) {
                            if (g < 0) break;
                            const AttractorPoint& currentAttrPt = dev_attrPts_memCoherent[g];
                            if (currentAttrPt.removed) { break; }
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
    }
    currentBud.optimalGrowthDir = currentBud.numNearbyAttrPts > 0 ? glm::normalize(currentBud.optimalGrowthDir) : glm::vec3(0.0f);
}

// Uniform Grid Implementation functions

__global__ void kernComputeIndices(const int numAttrPts, const int gridResolution,
    const glm::vec3 gridMin, const float inverseCellWidth,
    const AttractorPoint* attrPts, int* attrPtIndices, int* gridIndices) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numAttrPts) {
        return;
    }
    glm::vec3 index3D = floor((attrPts[index].point - gridMin) * inverseCellWidth);
    int index1D = gridIndex3Dto1D(index3D.x, index3D.y, index3D.z, gridResolution);
    gridIndices[index] = index1D;
    attrPtIndices[index] = index;
}

__global__ void kernMakeDataMemoryCoherent(const int numAttrPts, const int* attrPtIndices,
    const AttractorPoint* attrPts, AttractorPoint* attrPts_memCoherent) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numAttrPts) {
        return;
    }
    attrPts_memCoherent[index] = attrPts[attrPtIndices[index]];
}

__global__ void kernIdentifyCellStartEnd(const int numAttrPts, int* gridCellIndices,
    int* gridCellStartIndices, int* gridCellEndIndices) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= numAttrPts) {
        return;
    } else {
        int gridIdx = gridCellIndices[index];

        if (index == 0) {
            gridCellStartIndices[gridIdx] = 0;
            return;
        }

        if (index == numAttrPts - 1) {
            gridCellEndIndices[gridIdx] = index;
        }

        int gridIdxPrev = gridCellIndices[index - 1];

        if (gridIdx != gridIdxPrev) {
            gridCellStartIndices[gridIdx] = index;
            gridCellEndIndices[gridIdxPrev] = index - 1;
        }
    }
}

cudaError_t RunSpaceColonizationKernel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints,
                                       const int gridSideCount, const int numTotalGridCells, const glm::vec3& gridMin, const float gridCellWidth, bool& reconstructUniformGrid) {
    cudaError_t cudaStatus;

    Bud* dev_buds = 0;
    AttractorPoint* dev_attrPts = 0;
    AttractorPoint* dev_attrPts_memCoherent = 0;
    int* dev_mutex = 0;

    const int blockSize = 32;
    dim3 fullBlocksPerGrid_Buds((numBuds + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGrid_AttrPts((numAttractorPoints + blockSize - 1) / blockSize);

    // Create the uniform grid if it hasn't been created / needs to be recreated
    if (reconstructUniformGrid) {
        const float gridInverseCellWidth = 1.0f / gridCellWidth;
        cudaStatus = cudaMalloc((void**)&dev_attrPtIndices, numAttractorPoints * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc dev_attrPtIndices failed!");
            //goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_gridCellIndices, numAttractorPoints * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc dev_gridCellIndices failed!");
            //goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_gridCellStartIndices, numTotalGridCells * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc dev_gridCellStartIndices failed!");
            //goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_gridCellEndIndices, numTotalGridCells * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            //goto Error;
        }

        cudaMemset(dev_gridCellIndices, -1, numAttractorPoints * sizeof(int));
        checkCUDAErrorWithLine("Before Compute Indices");
        cudaMemset(dev_gridCellStartIndices, -1, numTotalGridCells * sizeof(int));
        checkCUDAErrorWithLine("Before Compute Indices");
        cudaMemset(dev_gridCellEndIndices, -1, numTotalGridCells * sizeof(int));
        checkCUDAErrorWithLine("Before Compute Indices");
    }

    

    // Device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        //goto Error;
    }

    // Cuda Malloc
    cudaStatus = cudaMalloc((void**)&dev_buds, numBuds * sizeof(Bud));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_buds failed!");
        //goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_attrPts, numAttractorPoints * sizeof(AttractorPoint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_attrPts failed!");
        //goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_attrPts_memCoherent, numAttractorPoints * sizeof(AttractorPoint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_attrPts_memCoherent failed!");
        //goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mutex, numAttractorPoints * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_mutex failed!");
        //goto Error;
    }

    // Cuda memcpy
    cudaStatus = cudaMemcpy(dev_buds, buds, numBuds * sizeof(Bud), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_buds failed!");
        //goto Error;
    }

    cudaStatus = cudaMemcpy(dev_attrPts, attractorPoints, numAttractorPoints * sizeof(AttractorPoint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_attrPts failed!");
        //goto Error;
    }

    cudaDeviceSynchronize();

    cudaMemset(dev_mutex, 0, numAttractorPoints * sizeof(int));
    checkCUDAErrorWithLine("Before Compute Indices");

    cudaDeviceSynchronize();
    
    checkCUDAErrorWithLine("Before Compute Indices");

    kernMarkAttractorPointsAsRemoved << < fullBlocksPerGrid_Buds, blockSize >> > (dev_buds, gridMin, gridSideCount, gridInverseCellWidth, numBuds, dev_attrPts_memCoherent,
                                                                                  numAttractorPoints, dev_mutex, dev_gridCellStartIndices, dev_gridCellEndIndices);

    cudaDeviceSynchronize();

    kernComputeIndices << <fullBlocksPerGrid_AttrPts, blockSize >> > (numAttractorPoints, gridSideCount, gridMin, gridInverseCellWidth, dev_attrPts, dev_attrPtIndices, dev_gridCellIndices);

    cudaDeviceSynchronize();

    checkCUDAErrorWithLine("After Compute Indices");

    thrust::device_ptr<int> dev_thrust_gridcell_indices(dev_gridCellIndices);
    thrust::device_ptr<int> dev_thrust_attrpt_indices(dev_attrPtIndices);

    checkCUDAErrorWithLine("After Compute Indices");

    // Sorting with thrust
    thrust::sort_by_key(dev_thrust_gridcell_indices, dev_thrust_gridcell_indices + numAttractorPoints, dev_thrust_attrpt_indices);

    cudaDeviceSynchronize();

    checkCUDAErrorWithLine("After thrust sort");

    kernIdentifyCellStartEnd << <fullBlocksPerGrid_AttrPts, blockSize >> > (numAttractorPoints, dev_gridCellIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

    cudaDeviceSynchronize();

    checkCUDAErrorWithLine("After identify cell start/end");

    kernMakeDataMemoryCoherent << <fullBlocksPerGrid_AttrPts, blockSize >> > (numAttractorPoints, dev_attrPtIndices, dev_attrPts, dev_attrPts_memCoherent);

    cudaDeviceSynchronize();

    checkCUDAErrorWithLine("After make data coherent");

    kernSetNearestBudForAttractorPoints << < fullBlocksPerGrid_Buds, blockSize >> > (dev_buds, gridMin, gridSideCount, gridInverseCellWidth, numBuds, dev_attrPts_memCoherent,
                                                                                     numAttractorPoints, dev_mutex, dev_gridCellStartIndices, dev_gridCellEndIndices);

    cudaDeviceSynchronize();

    checkCUDAErrorWithLine("After space col pass 1");

    kernSpaceCol << < fullBlocksPerGrid_Buds, blockSize >> > (dev_buds, gridMin, gridSideCount, gridInverseCellWidth, numBuds, dev_attrPts_memCoherent,
                                                              numAttractorPoints, dev_mutex, dev_gridCellStartIndices, dev_gridCellEndIndices);

    cudaDeviceSynchronize();

    checkCUDAErrorWithLine("After space col pass 2");

    cudaDeviceSynchronize();

    // Cuda Memcpy the Bud info back to the CPU
    cudaStatus = cudaMemcpy(buds, dev_buds, numBuds * sizeof(Bud), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to buds failed!");
        //goto Error;
    }

//Error: {
    cudaFree(dev_buds);
    cudaFree(dev_attrPts);
    cudaFree(dev_attrPts_memCoherent);
    cudaFree(dev_mutex);
    cudaFree(dev_gridCellIndices);
    cudaFree(dev_gridCellStartIndices);
    cudaFree(dev_gridCellEndIndices);
//}
    return cudaStatus;
}

void TreeApp::PerformSpaceColonizationParallel(Bud* buds, const int numBuds, AttractorPoint* attractorPoints, const int numAttractorPoints,
                                               const int gridSideCount, const int numTotalGridCells, const glm::vec3& gridMin, const float gridCellWidth, bool& reconstructUniformGrid) {
    cudaError_t cudaStatus = RunSpaceColonizationKernel(buds, numBuds, attractorPoints, numAttractorPoints, gridSideCount, numTotalGridCells, gridMin, gridCellWidth, reconstructUniformGrid);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Space Colonization failed!\n");
    }
}
