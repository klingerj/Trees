
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
