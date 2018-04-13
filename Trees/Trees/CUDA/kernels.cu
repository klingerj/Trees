
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernels.h"
#include "../Scene/Tree.h"

#include <stdio.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

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
    //printf("line11 %d", index);
    Bud& currentBud = dev_buds[index];
    
    //printf("line12 %d", index);
    const glm::vec3 budPosLocalToGrid = currentBud.point - gridMin;
    //printf("line13\n");
    const glm::vec3 index3D = glm::floor(budPosLocalToGrid * inverseCellWidth);
    //printf("line14\n");
    //printf("index3D.x: %f", index3D.x);
    //printf("index3D.y: %f", index3D.y);
    //printf("index3D.z: %f\n", index3D.z);

    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        //printf("line15\n");
        for (int x = -1; x <= 1; ++x) {
            //printf("line16\n");
            for (int y = -1; y <= 1; ++y) {
                //printf("line17\n");
                for (int z = -1; z <= 1; ++z) {
                    //printf("line18\n");
                    const glm::vec3 currentGridIndex = index3D + glm::vec3(x, y, z);

                    //printf("line19\n");
                    if (((((int)currentGridIndex.x) >= 0 && ((int)currentGridIndex.x) < gridResolution) &&
                        (((int)currentGridIndex.y) >= 0 && ((int)currentGridIndex.y) < gridResolution)) &&
                        (((int)currentGridIndex.z) >= 0 && ((int)currentGridIndex.z) < gridResolution)) {
                        //printf("line120\n");
                        int index1D = gridIndex3Dto1D(currentGridIndex.x, currentGridIndex.y, currentGridIndex.z, gridResolution);
                        for (int g = gridCellStartIndices[index1D]; g <= gridCellEndIndices[index1D]; ++g) {
                            //printf("%d\n", g);
                            if (g < 0) { break; }
                            //printf("%d\n", g);
                            AttractorPoint& currentAttrPt = dev_attrPts_memCoherent[g];
                            glm::vec3 budToPtDir = currentAttrPt.point - currentBud.point;
                            const float budToPtDist2 = glm::length2(budToPtDir);
                            budToPtDir = glm::normalize(budToPtDir);
                            const float dotProd = glm::dot(budToPtDir, currentBud.naturalGrowthDir);
                            if (budToPtDist2 < (14.0f * currentBud.internodeLength * currentBud.internodeLength) && dotProd > std::abs(COS_THETA_SMALL)) {
                                //printf("Got a point\n");
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

__global__ void kernSpaceCol(Bud* dev_buds, const glm::vec3 gridMin, const int gridResolution, const float inverseCellWidth, const int numBuds,
    AttractorPoint* dev_attrPts_memCoherent, const int numAttractorPoints, int* dev_mutex, int* gridCellStartIndices,
    int* gridCellEndIndices) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numBuds) {
        return;
    }
    
    //printf("line21\n");
    Bud& currentBud = dev_buds[index];
    
    //printf("line22\n");
    const glm::vec3 budPosLocalToGrid = currentBud.point - gridMin;
    //printf("line23\n");
    const glm::vec3 index3D = floor(budPosLocalToGrid * inverseCellWidth);
    //printf("line24\n");

    // Space Colonization
    if (currentBud.internodeLength > 0.0f && currentBud.fate == DORMANT) {
        //printf("line25\n");
        for (int x = -1; x <= 1; ++x) {
            //printf("line26\n");
            for (int y = -1; y <= 1; ++y) {
                //printf("line27\n");
                for (int z = -1; z <= 1; ++z) {
                    //printf("line28\n");
                    const glm::vec3 currentGridIndex = index3D + glm::vec3(x, y, z);
                    //printf("line29\n");

                    if (((((int)currentGridIndex.x) >= 0 && ((int)currentGridIndex.x) < gridResolution) &&
                        (((int)currentGridIndex.y) >= 0 && ((int)currentGridIndex.y) < gridResolution)) &&
                        (((int)currentGridIndex.z) >= 0 && ((int)currentGridIndex.z) < gridResolution)) {
                        //printf("line210\n");
                        int index1D = gridIndex3Dto1D(currentGridIndex.x, currentGridIndex.y, currentGridIndex.z, gridResolution);
                        //printf("line211\n");
                        for (int g = gridCellStartIndices[index1D]; g <= gridCellEndIndices[index1D]; ++g) {
                            if (g < 0) break;
                            const AttractorPoint& currentAttrPt = dev_attrPts_memCoherent[g];
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
    currentBud.optimalGrowthDir = currentBud.numNearbyAttrPts > 0 ? glm::normalize(currentBud.optimalGrowthDir) : glm::vec3(0.0f);
    //printf("Optimal Growth Dir: %f, %f, %f", currentBud.optimalGrowthDir.x, currentBud.optimalGrowthDir.y, currentBud.optimalGrowthDir.z);
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
                                       const int gridSideCount, const int numTotalGridCells, const glm::vec3& gridMin, const float gridCellWidth) {
    cudaError_t cudaStatus;

    Bud* dev_buds = 0;
    AttractorPoint* dev_attrPts = 0;
    AttractorPoint* dev_attrPts_memCoherent = 0;
    int* dev_mutex = 0;
    int* dev_attrPtIndices = 0; // indices of each attractor point (0, 1, ..., n)
    int* dev_gridCellIndices = 0; // grid cell index of each attractor point
    int* dev_gridCellStartIndices = 0; // start index of a grid cell
    int* dev_gridCellEndIndices = 0; // end index of a grid cell

    const float gridInverseCellWidth = 1.0f / gridCellWidth;
    const int blockSize = 32;
    dim3 fullBlocksPerGrid_Buds((numBuds + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGrid_AttrPts((numAttractorPoints + blockSize - 1) / blockSize);

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
    cudaMemset(dev_gridCellIndices, -1, numAttractorPoints * sizeof(int));
    checkCUDAErrorWithLine("Before Compute Indices");
    cudaMemset(dev_gridCellStartIndices, -1, numTotalGridCells * sizeof(int));
    checkCUDAErrorWithLine("Before Compute Indices");
    cudaMemset(dev_gridCellEndIndices, -1, numTotalGridCells * sizeof(int));
    checkCUDAErrorWithLine("Before Compute Indices");

    cudaDeviceSynchronize();
    
    checkCUDAErrorWithLine("Before Compute Indices");

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
                                               const int gridSideCount, const int numTotalGridCells, const glm::vec3& gridMin, const float gridCellWidth) {
    cudaError_t cudaStatus = RunSpaceColonizationKernel(buds, numBuds, attractorPoints, numAttractorPoints, gridSideCount, numTotalGridCells, gridMin, gridCellWidth);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Space Colonization failed!\n");
    }
}
