#include "heatmap_cuda.h"

void Ped::Model::setupHeatmapCUDA()
{
    /* Sequential implementation
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

    

}   

static const int BLOCK_SIZE = 16; // Example block size; tune as needed
static const int W[5][5] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}};
__constant__ int d_W[5][5];

#define WEIGHTSUM 273

//------------------------------------------------------------
// KERNEL: Fade the heatmap by 80%
//------------------------------------------------------------
__global__ void kernel_fade(int *heatmap, int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < size && y < size)
    {
        // Multiply by 0.8
        // (rounding can be done in various ways; here we just do int-cast)
        float faded = heatmap[y * size + x] * 0.8f;
        heatmap[y * size + x] = (int)lrintf(faded); // or roundf/floorf
    }
}

//------------------------------------------------------------
// KERNEL: Add agent contributions using atomicAdd
//   agentX[i], agentY[i] are each agent's position
//------------------------------------------------------------
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
            // Data race possible here --> use atomic
            atomicAdd(&heatmap[y * size + x], 40);
        }
    }
}

//------------------------------------------------------------
// KERNEL: Clamp values in heatmap to [0..255]
//------------------------------------------------------------
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

//------------------------------------------------------------
// KERNEL: Scale the heatmap into a bigger array
//   scaled_heatmap: SCALED_SIZE x SCALED_SIZE
//   each cell in 'heatmap' replicates into a CELLSIZE x CELLSIZE block
//------------------------------------------------------------
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

//------------------------------------------------------------
// KERNEL: 5x5 Gaussian blur using shared memory
//   Each thread computes one pixel of blurred output.
//   We copy the needed input region (BLOCKDIM + 4 in each dimension)
//   to shared memory to reduce repeated reads from global memory.
//------------------------------------------------------------
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

    // Additionally, the block boundary threads need to load the halos
    // on the left/right/top/bottom. You can do that with extra if-checks
    // or just let the "extra threads" load them.
    // ...
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

//------------------------------------------------------------
// Called from Ped::Model::updateHeatmapSeq (renamed to updateHeatmapCUDA?)
// This function launches the kernels in the correct sequence.
//------------------------------------------------------------
void Ped::Model::updateHeatmapCUDA()
{
    // 0) [One-time setup] Make sure we have dev_heatmap, dev_scaled_heatmap, etc.
    //    If not allocated, allocate them. Also copy the heatmap to dev_heatmap if needed.

    // For demonstration, we assume they're already allocated and contain the old data.

    cudaMemcpyToSymbol(d_W, W, sizeof(W)); // Copy W to device memory

    // 1) Fade
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_fade<<<grid, block>>>(dev_heatmap, SIZE);

    int numAgents = agents.size();

    // 2) Agent additions with atomicAdd
    //    Suppose we have agent positions in dev_agentX, dev_agentY, and the number is 'numAgents'.
    //    We use a 1D grid for convenience:
    int threadsPerBlock = 128;
    int blocksForAgents = (numAgents + threadsPerBlock - 1) / threadsPerBlock;
    kernel_addAgents<<<blocksForAgents, threadsPerBlock>>>(dev_heatmap, SIZE,
                                                           dev_agentX, dev_agentY,
                                                           numAgents);

    // 3) Clamp
    kernel_clamp<<<grid, block>>>(dev_heatmap, SIZE);

    // 4) Scale
    dim3 gridScaled((SCALED_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (SCALED_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel_scale<<<gridScaled, block>>>(dev_heatmap, SIZE,
                                        dev_scaled_heatmap, SCALED_SIZE,
                                        CELLSIZE);

    // 5) Blur (using shared memory)
    kernel_blur<<<gridScaled, block>>>(dev_scaled_heatmap, dev_blurred_heatmap, SCALED_SIZE);

    // NOTE: All calls above are asynchronous; to truly overlap with CPU collision handling,
    //       do not call cudaDeviceSynchronize() here. The CPU can go do collision handling
    //       while the GPU runs these kernels.

    // ... CPU does collision handling in parallel ...

    // When we need the final blurred heatmap on the CPU side:
    cudaDeviceSynchronize(); // Wait for GPU to finish
    cudaMemcpy(blurred_heatmap[0], dev_blurred_heatmap,
               SCALED_SIZE * SCALED_SIZE * sizeof(int),
               cudaMemcpyDeviceToHost);
}
