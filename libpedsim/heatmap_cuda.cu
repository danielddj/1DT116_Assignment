// heatmap_cuda.cu
#include <cuda_runtime.h>
#include <cstring>     // for memcpy
#include "ped_model.h" // This header should declare Ped::Model and its members

#define FADE_FACTOR 0.8f
#define HEAT_INCREMENT 40

// Kernel: Fade heatmap values.
__global__ void fadeHeatmapKernel(int *heatmap, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size)
    {
        float oldVal = heatmap[idx];
        int newVal = __float2int_rn(oldVal * FADE_FACTOR);
        heatmap[idx] = newVal;
    }
}

// Kernel: Add agents’ contributions using atomicAdd.
__global__ void addAgentsKernel(int *heatmap, int size,
                                const int *agentX, const int *agentY, int numAgents)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numAgents)
    {
        int x = agentX[i];
        int y = agentY[i];
        if (x >= 0 && x < size && y >= 0 && y < size)
        {
            atomicAdd(&heatmap[y * size + x], HEAT_INCREMENT);
        }
    }
}

// Kernel: Clamp heatmap values to 255.
__global__ void clampHeatmapKernel(int *heatmap, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * size)
    {
        int val = heatmap[idx];
        if (val > 255)
            heatmap[idx] = 255;
    }
}

// Kernel: Scale the heatmap so that each cell expands to a CELLSIZE x CELLSIZE block.
__global__ void scaleHeatmapKernel(const int *heatmap, int *scaled,
                                   int size, int cellSize)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x; // scaled x coordinate
    int sy = blockIdx.y * blockDim.y + threadIdx.y; // scaled y coordinate
    int scaledSize = size * cellSize;
    if (sx < scaledSize && sy < scaledSize)
    {
        int origX = sx / cellSize;
        int origY = sy / cellSize;
        int value = heatmap[origY * size + origX];
        scaled[sy * scaledSize + sx] = value;
    }
}

#define BLOCK_DIM 16
// Gaussian weights stored in constant memory.
__constant__ int GAUSS_KERNEL[5][5] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}};

// Kernel: Apply a 5x5 Gaussian blur filter using shared memory.
__global__ void blurHeatmapKernel(const int *scaled, int *blurred, int scaledSize)
{
    __shared__ int tile[BLOCK_DIM + 4][BLOCK_DIM + 4]; // shared tile with halo
    int outX = 2 + blockIdx.x * BLOCK_DIM + threadIdx.x;
    int outY = 2 + blockIdx.y * BLOCK_DIM + threadIdx.y;
    int tileX = outX - 2;
    int tileY = outY - 2;

    // Load the tile with halo into shared memory.
    for (int dy = threadIdx.y; dy < BLOCK_DIM + 4; dy += BLOCK_DIM)
    {
        for (int dx = threadIdx.x; dx < BLOCK_DIM + 4; dx += BLOCK_DIM)
        {
            int globalX = tileX + dx;
            int globalY = tileY + dy;
            if (globalX >= 0 && globalX < scaledSize && globalY >= 0 && globalY < scaledSize)
                tile[dy][dx] = scaled[globalY * scaledSize + globalX];
            else
                tile[dy][dx] = 0;
        }
    }
    __syncthreads();

    if (outX < scaledSize - 2 && outY < scaledSize - 2)
    {
        int sum = 0;
        for (int ky = 0; ky < 5; ++ky)
        {
            for (int kx = 0; kx < 5; ++kx)
            {
                sum += GAUSS_KERNEL[ky][kx] * tile[threadIdx.y + ky][threadIdx.x + kx];
            }
        }
        int blurredVal = sum / 273; // 273 is the sum of the Gaussian weights.
        // Store as ARGB (red color, heat encoded in alpha channel).
        blurred[outY * scaledSize + outX] = 0x00FF0000 | (blurredVal << 24);
    }
}

////////////////////////////////////////////////////////////////////////
// In Ped::Model (in ped_model.cpp), update updateHeatmapCUDA() as follows:
////////////////////////////////////////////////////////////////////////

void Ped::Model::updateHeatmapCUDA()
{
    // Use SIZE and CELLSIZE (defined as in your sequential code)
    int size = SIZE;         // original heatmap dimension
    int cellSize = CELLSIZE; // each cell's scaling factor
    int totalCells = size * size;
    int scaledSize = SCALED_SIZE; // should equal size * cellSize

    // --- Step 1: Flatten the host heatmap into a contiguous buffer.
    int *flatHeatmap = new int[totalCells];
    for (int i = 0; i < size; i++)
    {
        // Each heatmap[i] is a pointer to a row.
        memcpy(&flatHeatmap[i * size], this->heatmap[i], size * sizeof(int));
    }

    // Copy the flattened heatmap into device memory.
    cudaMemcpy(d_heatmap, flatHeatmap, totalCells * sizeof(int), cudaMemcpyHostToDevice);

    // Suppose hostDesiredX and hostDesiredY contain the current desired positions.
    cudaMemcpy(d_agentX, desiredX.data(), agents.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_agentY, desiredY.data(), agents.size() * sizeof(int), cudaMemcpyHostToDevice);

    // --- Step 2: Launch CUDA kernels.
    int threadsPerBlock = 256;
    int blocks = (totalCells + threadsPerBlock - 1) / threadsPerBlock;
    fadeHeatmapKernel<<<blocks, threadsPerBlock>>>(d_heatmap, size);

    int agentBlocks = (agents.size() + threadsPerBlock - 1) / threadsPerBlock;
    addAgentsKernel<<<agentBlocks, threadsPerBlock>>>(d_heatmap, size, d_agentX, d_agentY, agents.size());

    clampHeatmapKernel<<<blocks, threadsPerBlock>>>(d_heatmap, size);

    // Scale the heatmap for visualization.
    dim3 threadsPerBlock2D(16, 16);
    dim3 blocks2D((scaledSize + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                  (scaledSize + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
    scaleHeatmapKernel<<<blocks2D, threadsPerBlock2D>>>(d_heatmap, d_scaled, size, cellSize);

    // Apply Gaussian blur using shared memory.
    dim3 threadsPerBlockBlur(BLOCK_DIM, BLOCK_DIM);
    dim3 blocksBlur((scaledSize - 4 + BLOCK_DIM - 1) / BLOCK_DIM,
                    (scaledSize - 4 + BLOCK_DIM - 1) / BLOCK_DIM);
    blurHeatmapKernel<<<blocksBlur, threadsPerBlockBlur>>>(d_scaled, d_blurred, scaledSize);

    // --- Step 3: Synchronize and copy back results.
    cudaDeviceSynchronize();

    // Copy blurred heatmap for visualization.
    cudaMemcpy(blurred_heatmap[0], d_blurred, scaledSize * scaledSize * sizeof(int), cudaMemcpyDeviceToHost);
    // Optionally, update the host heatmap.
    cudaMemcpy(flatHeatmap, d_heatmap, totalCells * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++)
    {
        memcpy(this->heatmap[i], &flatHeatmap[i * size], size * sizeof(int));
    }
    delete[] flatHeatmap;
}

void Ped::Model::allocateCudaMemory()
{
    int totalCells = SIZE * SIZE;
    int scaledTotal = SCALED_SIZE * SCALED_SIZE;
    // Allocate device memory for heatmap and scaled/blurred images.
    cudaMalloc((void **)&d_heatmap, totalCells * sizeof(int));
    cudaMalloc((void **)&d_scaled, scaledTotal * sizeof(int));
    cudaMalloc((void **)&d_blurred, scaledTotal * sizeof(int));

    // Allocate memory for agent data (if needed)
    cudaMalloc((void **)&d_agentX, agents.size() * sizeof(int));
    cudaMalloc((void **)&d_agentY, agents.size() * sizeof(int));
}

void Ped::Model::freeCudaMemory()
{
    cudaFree(d_heatmap);
    cudaFree(d_scaled);
    cudaFree(d_blurred);
    cudaFree(d_agentX);
    cudaFree(d_agentY);
}