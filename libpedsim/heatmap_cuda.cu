#include "heatmap_cuda.h"

void Ped::Model::setupHeatmapCUDA()
{
    /* Sequential implementation for reference
    int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
    int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
    int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

    heatmap = (int**)malloc(SIZE*sizeof(int*));

    scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
    blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

    for (int i = 0; i < SIZE; i++)
    {
        heatmap[i] = hm + SIZE*i;
    }
    for (int i = 0; i < SCALED_SIZE; i++)
    {
        scaled_heatmap[i] = shm + SCALED_SIZE*i;
        blurred_heatmap[i] = bhm + SCALED_SIZE*i;
    }
    */

    cudaError_t err;
    err = cudaMalloc((void **)&dev_heatmap, SIZE * SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dev_heatmap failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaMalloc((void **)&dev_scaled_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dev_scaled_heatmap failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaMalloc((void **)&dev_blurred_heatmap, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dev_blurred_heatmap failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // 3) Allocate device arrays for agent X,Y positions
    //    (so the GPU can do kernel_addAgents)
    int n = static_cast<int>(agents.size());
    err = cudaMalloc((void **)&dev_agentX, n * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dev_agentX failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    err = cudaMalloc((void **)&dev_agentY, n * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc dev_agentY failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // 4) Initialize the device memory to zero
    cudaMemset(dev_heatmap, 0, SIZE * SIZE * sizeof(int));
    cudaMemset(dev_scaled_heatmap, 0, SCALED_SIZE * SCALED_SIZE * sizeof(int));
    cudaMemset(dev_blurred_heatmap, 0, SCALED_SIZE * SCALED_SIZE * sizeof(int));

    // 5) Initialize agentX/agentY on the host,
    //    then copy to dev_agentX/dev_agentY
    {
        std::vector<int> hostAx(n), hostAy(n);
        for (int i = 0; i < n; i++)
        {
            // pull from your Tagent objects
            hostAx[i] = agents[i]->getX();
            hostAy[i] = agents[i]->getY();
        }
        cudaMemcpy(dev_agentX, hostAx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_agentY, hostAy.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Flag to remember we have allocated everything
    device_allocated = true;

    //  Done!
    printf("setupHeatmapCUDA() complete: host & device allocations done.\n");
}

static const int BLOCK_SIZE = 16;
static const int W[5][5] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}};
__constant__ int d_W[5][5];

#define WEIGHTSUM 273

// KERNEL: Fade the heatmap by 80%
__global__ void kernel_fade(int *heatmap, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size)
    {
        // Multiply by 0.8
        float faded = heatmap[y * size + x] * 0.8f;
        heatmap[y * size + x] = (int)lrintf(faded); // or roundf/floorf
    }
}

// KERNEL: Add agent contributions using atomicAdd
__global__ void kernel_addAgents(int *heatmap, int size,
                                 const int *agentX, const int *agentY,
                                 int numAgents)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numAgents)
    {
        int x = agentX[i];
        int y = agentY[i];
        // Check bounds
        if (x >= 0 && x < size && y >= 0 && y < size)
        {
            // Data race possible here --> so we use atomic
            atomicAdd(&heatmap[y * size + x], 40);
        }
    }
}

// KERNEL: Clamp values in heatmap to [0..255]
__global__ void kernel_clamp(int *heatmap, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size)
    {
        int val = heatmap[y * size + x];
        if (val > 255)
            val = 255;
        heatmap[y * size + x] = val;
    }
}

// KERNEL: Scale the heatmap into a bigger array
//   scaled_heatmap: SCALED_SIZE x SCALED_SIZE
//   each cell in 'heatmap' replicates into a CELLSIZE x CELLSIZE block
__global__ void kernel_scale(const int *heatmap, int size,
                             int *scaled_heatmap, int scaledSize,
                             int cellSize)
{
    int sx = blockIdx.x * blockDim.x + threadIdx.x; // scaled x
    int sy = blockIdx.y * blockDim.y + threadIdx.y; // scaled y

    if (sx < scaledSize && sy < scaledSize)
    {
        // Map back to original heatmap coordinates
        int x = sx / cellSize;
        int y = sy / cellSize;
        scaled_heatmap[sy * scaledSize + sx] = heatmap[y * size + x];
    }
}

// KERNEL: 5x5 Gaussian blur using shared memory
__global__ void kernel_blur(const int *in, int *out,
                            int scaledSize)
{
    // We assume BLOCK_SIZE x BLOCK_SIZE threads in x,y
    // We'll load a (BLOCK_SIZE+4) x (BLOCK_SIZE+4) tile in shared mem.

    // 1) Compute global coords
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // 2) Allocate shared memory for the tile
    __shared__ int tile[BLOCK_SIZE + 4][BLOCK_SIZE + 4];

    // 3) Figure out which element(s) this thread must load
    //    We need +2 "halo" on each side. One approach:
    int lx = threadIdx.x + 2; // local x in shared mem
    int ly = threadIdx.y + 2; // local y in shared mem
    int tileX = blockIdx.x * blockDim.x + lx - 2;
    int tileY = blockIdx.y * blockDim.y + ly - 2;

    // If tileX/tileY are in range, load them into tile.
    if (tileX >= 0 && tileX < scaledSize &&
        tileY >= 0 && tileY < scaledSize)
    {
        tile[ly][lx] = in[tileY * scaledSize + tileX];
    }
    else
    {
        tile[ly][lx] = 0; // out of bounds
    }

    // Use __syncthreads() to make sure the entire tile is loaded
    __syncthreads();

    // 4) If (gx,gy) is in range, do the blur
    if (gx >= 2 && gx < (scaledSize - 2) &&
        gy >= 2 && gy < (scaledSize - 2))
    {
        // local x,y in the tile memory is (threadIdx.x+2, threadIdx.y+2)
        // we want a 5x5 sum around that point
        int sum = 0;
        for (int ky = -2; ky <= 2; ky++)
        {
            for (int kx = -2; kx <= 2; kx++)
            {
                // local tile coords
                int ty = ly + ky;
                int tx = lx + kx;
                sum += d_W[ky + 2][kx + 2] * tile[ty][tx];
            }
        }
        int value = sum / WEIGHTSUM;

        // For example, store as 0x00FF0000 | value<<24:
        out[gy * scaledSize + gx] = 0x00FF0000 | (value << 24);
    }
}

// KERNEL: 5x5 Gaussian blur using *global* memory (no shared memory)
__global__ void kernel_blur_global_mem(const int *in, int *out, int scaledSize)
{
    // 1) Compute global coords
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // 2) Check if (gx, gy) is in a valid range for a 5x5 blur
    if (gx >= 2 && gx < (scaledSize - 2) &&
        gy >= 2 && gy < (scaledSize - 2))
    {
        // 3) Perform the 5x5 weighted sum directly from global memory
        int sum = 0;
        for (int ky = -2; ky <= 2; ky++)
        {
            for (int kx = -2; kx <= 2; kx++)
            {
                // neighbor coordinates
                int nx = gx + kx;
                int ny = gy + ky;

                // multiply by weight from constant memory and accumulate
                sum += d_W[ky + 2][kx + 2] * in[ny * scaledSize + nx];
            }
        }

        // 4) Divide by sum of weights and store as ARGB
        int value = sum / WEIGHTSUM;
        out[gy * scaledSize + gx] = 0x00FF0000 | (value << 24);
    }
}

void Ped::Model::updateHeatmapCUDA()
{
    int numAgents = agents.size();
    int n = static_cast<int>(agents.size());
    // 0) Fetch agent positions from host to device

    std::vector<int> hostAx(n), hostAy(n);
    for (int i = 0; i < n; i++)
    {
        // pull from your Tagent objects
        hostAx[i] = agents[i]->getX();
        hostAy[i] = agents[i]->getY();
    }
    cudaMemcpyAsync(dev_agentX, hostAx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(dev_agentY, hostAy.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_W, W, sizeof(W)); // Copy W to device memory

    // Set up the blocks & grids
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // For the scaled version:
    dim3 gridScaled((SCALED_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (SCALED_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int threadsPerBlock = 128;
    int blocksForAgents = (numAgents + threadsPerBlock - 1) / threadsPerBlock;

    // We'll use these for timing each kernel
    // cudaEvent_t startEvent, stopEvent;
    // float elapsedMs = 0.0f;

    // 1) Fade
    // cudaEventCreate(&startEvent);
    // cudaEventCreate(&stopEvent);
    // cudaEventRecord(startEvent);

    kernel_fade<<<grid, block>>>(dev_heatmap, SIZE);

    // cudaEventRecord(stopEvent);
    // cudaEventSynchronize(stopEvent);
    // cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    // std::cout << "Fade kernel time (ms): " << elapsedMs << std::endl;
    //  Accumulate in global variable
    // totalFadeTime += elapsedMs;

    // cudaEventDestroy(startEvent);
    // cudaEventDestroy(stopEvent);

    // 2) Agent additions with atomicAdd
    //    we use a 1D grid for convenience

    // cudaEventCreate(&startEvent);
    // cudaEventCreate(&stopEvent);
    // cudaEventRecord(startEvent);

    kernel_addAgents<<<blocksForAgents, threadsPerBlock>>>(dev_heatmap, SIZE,
                                                           dev_agentX, dev_agentY,
                                                           numAgents);

    // cudaEventRecord(stopEvent);
    // cudaEventSynchronize(stopEvent);
    // cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    // std::cout << "AddAgents kernel time (ms): " << elapsedMs << std::endl;
    //  Accumulate in global variable
    // totalAddAgentsTime += elapsedMs;

    // cudaEventDestroy(startEvent);
    // cudaEventDestroy(stopEvent);

    // 3) Clamp
    // cudaEventCreate(&startEvent);
    // cudaEventCreate(&stopEvent);
    // cudaEventRecord(startEvent);

    kernel_clamp<<<grid, block>>>(dev_heatmap, SIZE);

    // cudaEventRecord(stopEvent);
    // cudaEventSynchronize(stopEvent);
    // cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    //  << "Clamp kernel time (ms): " << elapsedMs << std::endl;
    //  Accumulate in global variable
    // totalClampTime += elapsedMs;

    // cudaEventDestroy(startEvent);
    // cudaEventDestroy(stopEvent);

    // 4) Scale
    // cudaEventCreate(&startEvent);
    // cudaEventCreate(&stopEvent);
    // cudaEventRecord(startEvent);

    kernel_scale<<<gridScaled, block>>>(dev_heatmap, SIZE,
                                        dev_scaled_heatmap, SCALED_SIZE,
                                        CELLSIZE);

    // cudaEventRecord(stopEvent);
    // cudaEventSynchronize(stopEvent);
    // cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    // std::cout << "Scale kernel time (ms): " << elapsedMs << std::endl;
    //  Accumulate in global variable
    // totalScaleTime += elapsedMs;

    // cudaEventDestroy(startEvent);
    // cudaEventDestroy(stopEvent);

    // 5) Blur (using shared memory)
    // cudaEventCreate(&startEvent);
    // cudaEventCreate(&stopEvent);
    // cudaEventRecord(startEvent);

    kernel_blur<<<gridScaled, block>>>(dev_scaled_heatmap, dev_blurred_heatmap, SCALED_SIZE);
    // kernel_blur_global_mem<<<gridScaled, block>>>(dev_scaled_heatmap, dev_blurred_heatmap, SCALED_SIZE);

    // cudaEventRecord(stopEvent);
    // cudaEventSynchronize(stopEvent);
    // cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    // std::cout << "Blur kernel time (ms): " << elapsedMs << std::endl;
    //  Accumulate in global variable
    // totalBlurTime += elapsedMs;

    // cudaEventDestroy(startEvent);
    // cudaEventDestroy(stopEvent);

    // Count one heatmap tick
    //heatmapTickCount++;
}

void Ped::Model::synchronizeCUDAHeatmapCalc()
{
    cudaDeviceSynchronize();
    cudaMemcpy(blurred_heatmap[0], dev_blurred_heatmap,
               SCALED_SIZE * SCALED_SIZE * sizeof(int),
               cudaMemcpyDeviceToHost);
}
