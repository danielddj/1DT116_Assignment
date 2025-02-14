#include "ped_model.h"
#include "ped_waypoint.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>

#define HEATMAP_WIDTH 160 * 5
#define HEATMAP_HEIGHT 120 * 5
#define HEATMAP_SKIP 5

bool firstRun = true;

struct Twaypoint
{
  float x;
  float y;
  float r; // threshold radius for "arrival"
};

#define CUDA_CHECK(call)                                                                                           \
  {                                                                                                                \
    cudaError_t err = call;                                                                                        \
    if (err != cudaSuccess)                                                                                        \
    {                                                                                                              \
      fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
      exit(err);                                                                                                   \
    }                                                                                                              \
  }

__device__ Twaypoint getNextDestination(const float *agentX, const float *agentY, const int *agentWaypoints,
                                        size_t agentWaypointsPitch, const float *waypointX, const float *waypointY,
                                        const float *waypointR, int numAgents, int *waypointIndex, int numWaypoints)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Default destination.
  Twaypoint dest = {0.0f, 0.0f, 0.0f};

  if (id >= numAgents)
    return dest;

  // Get current waypoint index for this agent.
  int wpIndex = waypointIndex[id];

  // Validate wpIndex before use.
  if (wpIndex < 0 || wpIndex >= numWaypoints)
  {
    wpIndex = 0;
    waypointIndex[id] = 0; // update agent's index to a valid value.
  }

  // Get pointer to the agent's waypoint list.
  const int *rowPtr = (const int *)((const char *)agentWaypoints + id * agentWaypointsPitch);

  // Read the waypoint ID for the current waypoint.
  int wpID = rowPtr[wpIndex];

  // Read destination values from the global waypoint arrays.
  float destX = waypointX[wpID - 1];
  float destY = waypointY[wpID - 1];
  float destR = waypointR[wpID - 1];

  // Compute distance between agent's current position and destination.
  float diffX = destX - agentX[id];
  float diffY = destY - agentY[id];
  float dist = sqrtf(diffX * diffX + diffY * diffY);

  // Check if the agent has reached its current destination.
  bool reached = (dist < destR);
  if (reached)
  {
    // Advance to the next waypoint in a circular manner.
    wpIndex = (wpIndex + 1) % numWaypoints;
    waypointIndex[id] = wpIndex;

    // Update the waypoint ID to the new one.
    wpID = rowPtr[wpIndex];
  }

  // Set the destination using the (possibly updated) waypoint ID.
  dest.x = destX;
  dest.y = destY;
  dest.r = destR;

  return dest;
}

__device__ void computeNextDesiredPosition(float *d_bufferXSim, float *d_bufferYSim, float *agentX, float *agentY,
                                           float *agentDesX, float *agentDesY, int *agentWaypoints,
                                           size_t agentWaypointsPitch, float *waypointX, float *waypointY,
                                           float *waypointR, int numAgents, int *waypointIndex, int numWaypoints)
{
  getNextDestination(agentX, agentY, agentWaypoints, agentWaypointsPitch, waypointX, waypointY, waypointR, numAgents,
                     waypointIndex, numWaypoints);

  int agentId = blockIdx.x * blockDim.x + threadIdx.x;
  if (agentId < numAgents)
  {
    float diffX = waypointX[agentWaypoints[agentId]] - agentX[agentId];
    float diffY = waypointY[agentWaypoints[agentId]] - agentY[agentId];
    float length = sqrt(diffX * diffX + diffY * diffY);

    d_bufferXSim[agentId] = agentX[agentId] + diffX / length;
    d_bufferYSim[agentId] = agentY[agentId] + diffY / length;
  }
}

//-----------------------------------------------------------------------------
// CUDA kernel to update agent positions.
// For each agent i, we do roughly the same as in sequential_tick():
//   diff = destination - current position
//   len = sqrt(diff.x*diff.x + diff.y*diff.y)
//   new position = current position + (diff/len)  (rounded)
// Then we store the new positions in desiredX/Y and update X/Y.
//-----------------------------------------------------------------------------
__global__ void cudaTickKernel(float *d_bufferX1Sim, float *d_bufferX2Transfer, float *d_bufferYSim,
                               float *d_bufferY2Transfer, float *agentX, float *agentY, float *agentDesX,
                               float *agentDesY, int *agentWaypoints, size_t agentWaypointsPitch, float *waypointX,
                               float *waypointY, float *waypointR, int numAgents, int *waypointIndex,
                               int numWaypoints)
{
  // Compute the new desired positions
  computeNextDesiredPosition(d_bufferX1Sim, d_bufferYSim, agentX, agentY, agentDesX, agentDesY, agentWaypoints,
                             agentWaypointsPitch, waypointX, waypointY, waypointR, numAgents, waypointIndex,
                             numWaypoints);

  // Wait for all threads to finish
  __syncthreads();

  // Update the agent positions
  for (int i = 0; i < numAgents; i++)
  {
    agentX[i] = d_bufferX1Sim[i];
    agentY[i] = d_bufferYSim[i];
  }

  // Wait for all threads to finish
  __syncthreads();
}

void serializeDataCuda(const float *h_exportBufferX, const float *h_exportBufferY, const Ped::Model &model,
                       std::ofstream &file)
{
  const std::vector<Ped::Tagent *> &agents = model.getAgents();
  size_t num_agents = agents.size();

  file.write(reinterpret_cast<const char *>(&num_agents), sizeof(num_agents));

  for (size_t i = 0; i < num_agents; i++)
  {

    int16_t x = static_cast<int16_t>(h_exportBufferX[i]);
    int16_t y = static_cast<int16_t>(h_exportBufferY[i]);

    file.write(reinterpret_cast<const char *>(&x), sizeof(x));
    file.write(reinterpret_cast<const char *>(&y), sizeof(y));
  }

  size_t heatmap_elements = model.getHeatmapSize();
  int16_t height = HEATMAP_HEIGHT;
  int16_t width = HEATMAP_WIDTH;
  const int *const *heatmap = model.getHeatmap();

  unsigned long heatmap_start = 0xFFFF0000FFFF0000;
  file.write(reinterpret_cast<const char *>(&heatmap_start), sizeof(heatmap_start));
  printf("heatmap_start: %ld\n", sizeof(heatmap_start));

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      int ARGBvalue = heatmap[i][j];
      int8_t Avalue = (ARGBvalue >> 24) & ((1 << 8) - 1);
      file.write(reinterpret_cast<const char *>(&Avalue), sizeof(Avalue));
    }
  }

  file.flush();
}

namespace Ped
{
  void Ped::Model::tick_cuda(size_t ticks, float *agentStartX, float *agentStartY, float *agentDesX, float *agentDesY,
                             float *waypointX, float *waypointY, float *waypointR, int *agentWaypoints,
                             size_t agentWaypointsPitch, int *waypointIndex)
  {
    // Calculate the number of blocks needed
    int threadsPerBlock = 256;
    int numBlocks = (X.size() + threadsPerBlock - 1) / 256;

    bool useBuffer1ForSim = true;

    int numAgents = X.size();
    int numWaypoints = X_WP.size();

    size_t size_agent = numAgents * sizeof(float);

    float *d_bufferX1 = nullptr, *d_bufferX2 = nullptr, *d_bufferY1 = nullptr, *d_bufferY2 = nullptr;
    cudaMalloc((void **)&d_bufferX1, size_agent);
    cudaMalloc((void **)&d_bufferX2, size_agent);
    cudaMalloc((void **)&d_bufferY1, size_agent);
    cudaMalloc((void **)&d_bufferY2, size_agent);

    cudaMemset(d_bufferX1, 0, size_agent);
    cudaMemset(d_bufferX2, 0, size_agent);
    cudaMemset(d_bufferY1, 0, size_agent);
    cudaMemset(d_bufferY2, 0, size_agent);

    float *h_exportBufferX = nullptr, *h_exportBufferY = nullptr;
    cudaMallocHost((void **)&h_exportBufferX, size_agent);
    cudaMallocHost((void **)&h_exportBufferY, size_agent);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (size_t i = 0; i < ticks; i++)
    {
      // Launch the kernel with 256 threads per block
      if (useBuffer1ForSim)
      {
        cudaTickKernel<<<numBlocks, threadsPerBlock>>>(
            d_bufferX1, d_bufferX2, d_bufferY1, d_bufferY2, agentStartX, agentStartY, agentDesX, agentDesY,
            agentWaypoints, agentWaypointsPitch, waypointX, waypointY, waypointR, numAgents, waypointIndex, numWaypoints);
      }
      else
      {
        cudaTickKernel<<<numBlocks, threadsPerBlock>>>(
            d_bufferX2, d_bufferX1, d_bufferY2, d_bufferY1, agentStartX, agentStartY, agentDesX, agentDesY,
            agentWaypoints, agentWaypointsPitch, waypointX, waypointY, waypointR, numAgents, waypointIndex, numWaypoints);
      }

      if (useBuffer1ForSim)
      {
        cudaMemcpyAsync(h_exportBufferX, d_bufferX2, size_agent, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_exportBufferY, d_bufferY2, size_agent, cudaMemcpyDeviceToHost, stream);
      }
      else
      {
        cudaMemcpyAsync(h_exportBufferX, d_bufferX1, size_agent, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_exportBufferY, d_bufferY1, size_agent, cudaMemcpyDeviceToHost, stream);
      }

      serializeDataCuda(h_exportBufferX, h_exportBufferY, *this, file);

      useBuffer1ForSim = !useBuffer1ForSim;
    }
  }

  void Ped::Model::start_cuda()
  {

    float *agentStartX, *agentStartY, *agentDesX, *agentDesY, *waypointX, *waypointY, *waypointR;
    int *agentWaypoints, *waypointIndex;

    int numAgents = X.size();
    int numWaypoints = X_WP.size();

    size_t size_agent = numAgents * sizeof(float);
    size_t size_waypoint = numWaypoints * sizeof(int);
    size_t pitch;

    CUDA_CHECK(cudaMalloc(&agentStartX, size_agent));
    CUDA_CHECK(cudaMalloc(&agentStartY, size_agent));
    CUDA_CHECK(cudaMalloc(&agentDesX, size_agent));
    CUDA_CHECK(cudaMalloc(&agentDesY, size_agent));
    CUDA_CHECK(cudaMallocPitch(&agentWaypoints, &pitch, size_waypoint, size_agent));
    CUDA_CHECK(cudaMalloc(&waypointX, size_agent));
    CUDA_CHECK(cudaMalloc(&waypointY, size_agent));
    CUDA_CHECK(cudaMalloc(&waypointR, size_agent));
    CUDA_CHECK(cudaMalloc(&waypointIndex, size_agent));

    CUDA_CHECK(cudaMemcpy(agentStartX, X.data(), size_agent, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(agentStartY, Y.data(), size_agent, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(agentDesX, X.data(), size_agent, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(agentDesY, Y.data(), size_agent, cudaMemcpyHostToDevice));

    int *agentWaypointsTmp = new int[numAgents * numWaypoints];

    for (size_t i = 0; i < numAgents; i++)
    {
      for (size_t j = 0; j < numWaypoints; j++)
      {
        agentWaypointsTmp[i * numWaypoints + j] = agents.at(i)->getDestination().at(j)->getid();
      }
    }

    CUDA_CHECK(cudaMemcpy2D(agentWaypoints, pitch, agentWaypointsTmp, size_waypoint, size_waypoint, numAgents,
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(waypointX, X_WP.data(), size_waypoint, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(waypointY, Y_WP.data(), size_waypoint, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(waypointR, R_WP.data(), size_waypoint, cudaMemcpyHostToDevice));

    int *waypointIndexTmp = new int[numAgents];

    for (size_t i = 0; i < numAgents; i++)
    {
      waypointIndexTmp[i] = agents.at(i)->getDestination().at(0)->getid();
    }

    CUDA_CHECK(cudaMemcpy(waypointIndex, waypointIndexTmp, size_agent, cudaMemcpyHostToDevice));

    tick_cuda(200, agentStartX, agentStartY, agentDesX, agentDesY, waypointX, waypointY, waypointR, agentWaypoints, pitch,
              waypointIndex);
  }
} // namespace Ped
