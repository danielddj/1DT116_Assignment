#include <cuda_runtime.h>
#include "ped_model.h"
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

//-----------------------------------------------------------------------------
// CUDA kernel to update agent positions.
// For each agent i, we do roughly the same as in sequential_tick():
//   diff = destination - current position
//   len = sqrt(diff.x*diff.x + diff.y*diff.y)
//   new position = current position + (diff/len)  (rounded)
// Then we store the new positions in desiredX/Y and update X/Y.
//-----------------------------------------------------------------------------
__global__ void cudaTickKernel(float *X, float *Y,
                               float *desiredX, float *desiredY,
                               const float *destX, const float *destY,
                               int numAgents)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numAgents)
    {
        float dx = destX[i] - X[i];
        float dy = destY[i] - Y[i];
        float len = sqrtf(dx * dx + dy * dy);

        if (len > 0.0f)
        {
            float newX = X[i] + dx / len;
            float newY = Y[i] + dy / len;

            desiredX[i] = roundf(newX);
            desiredY[i] = roundf(newY);
        }
        else
        {
            desiredX[i] = X[i];
            desiredY[i] = Y[i];
        }

        X[i] = desiredX[i];
        Y[i] = desiredY[i];
    }
}

namespace Ped
{
    void Ped::Model::cuda_tick()
    {
        int numAgents = agents.size();
        size_t size = numAgents * sizeof(float);

        for (int i = 0; i < numAgents; i++)
        {
            agents[i]->callNextDestination();
        }

        float *d_X, *d_Y, *d_desiredX, *d_desiredY, *d_destX, *d_destY;

        cudaMalloc((void **)&d_X, size);
        cudaMalloc((void **)&d_Y, size);
        cudaMalloc((void **)&d_desiredX, size);
        cudaMalloc((void **)&d_desiredY, size);
        cudaMalloc((void **)&d_destX, size);
        cudaMalloc((void **)&d_destY, size);

        cudaMemcpy(d_X, Ped::X.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, Ped::Y.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_desiredX, desiredX.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_desiredY, desiredY.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_destX, destinationX.data(), size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_destY, destinationY.data(), size, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (numAgents + blockSize - 1) / blockSize;

        cudaTickKernel<<<gridSize, blockSize>>>(d_X, d_Y, d_desiredX, d_desiredY, d_destX, d_destY, numAgents);

        cudaDeviceSynchronize();

        cudaMemcpy(Ped::X.data(), d_X, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(Ped::Y.data(), d_Y, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(desiredX.data(), d_desiredX, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(desiredY.data(), d_desiredY, size, cudaMemcpyDeviceToHost);

        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_desiredX);
        cudaFree(d_desiredY);
        cudaFree(d_destX);
        cudaFree(d_destY);

        for (int i = 0; i < numAgents; i++)
        {
            agents[i]->setX(desiredX[i]);
            agents[i]->setY(desiredY[i]);
        }
    }
}
