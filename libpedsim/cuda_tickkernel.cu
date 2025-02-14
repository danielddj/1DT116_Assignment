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

    destX = waypointX[wpID];
    destY = waypointY[wpID];
    destR = waypointR[wpID];
  }

  dest.x = destX;
  dest.y = destY;
  dest.r = destR;

  // Set the destination using the (possibly updated) waypoint ID.

  return dest;
}

__device__ void computeNextDesiredPosition(float *d_bufferXSim, float *d_bufferYSim, float *agentDesX, float *agentDesY, int *agentWaypoints,
                                           size_t agentWaypointsPitch, float *waypointX, float *waypointY,
                                           float *waypointR, int numAgents, int *waypointIndex, int numWaypoints)
{
  Twaypoint dest = getNextDestination(d_bufferXSim, d_bufferYSim, agentWaypoints, agentWaypointsPitch, waypointX, waypointY, waypointR, numAgents,
                                      waypointIndex, numWaypoints);

  int agentId = blockIdx.x * blockDim.x + threadIdx.x;
  if (agentId < numAgents)
  {
    float diffX = dest.x - d_bufferXSim[agentId];
    float diffY = dest.y - d_bufferYSim[agentId];
    float length = sqrt(diffX * diffX + diffY * diffY);

    d_bufferXSim[agentId] = d_bufferXSim[agentId] + diffX / length;
    d_bufferYSim[agentId] = d_bufferYSim[agentId] + diffY / length;
  }
}

__global__ void cudaTickKernel(float *d_bufferXSim, float *d_bufferX2Transfer, float *d_bufferYSim,
                               float *d_bufferY2Transfer, float *agentDesX, float *agentDesY,
                               int *agentWaypoints, size_t agentWaypointsPitch, float *waypointX,
                               float *waypointY, float *waypointR, int numAgents, int *waypointIndex,
                               int numWaypoints)
{
  computeNextDesiredPosition(d_bufferXSim, d_bufferYSim, agentDesX, agentDesY, agentWaypoints,
                             agentWaypointsPitch, waypointX, waypointY, waypointR, numAgents, waypointIndex,
                             numWaypoints);

  int agentId = blockIdx.x * blockDim.x + threadIdx.x;

  if (agentId < numAgents)
  {
    d_bufferX2Transfer[agentId] = d_bufferXSim[agentId];
    d_bufferY2Transfer[agentId] = d_bufferYSim[agentId];
  }
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
  size_t Ped::Model::tick_cuda(size_t ticks, float *d_bufferX1, float *d_bufferX2, float *d_bufferY1, float *d_bufferY2, float *agentDesX, float *agentDesY,
                               float *waypointX, float *waypointY, float *waypointR, int *agentWaypoints,
                               size_t agentWaypointsPitch, int *waypointIndex, bool serialize, std::ofstream *file)
  {
    // Calculate the number of blocks needed
    size_t tickCount;

    int threadsPerBlock = 256;
    int numBlocks = (X.size() + threadsPerBlock - 1) / 256;

    bool useBuffer1ForSim = true;

    int numAgents = X.size();
    int numWaypoints = X_WP.size();

    size_t size_agent = numAgents * sizeof(float);

    float *h_exportBufferX = nullptr, *h_exportBufferY = nullptr;
    cudaMallocHost((void **)&h_exportBufferX, size_agent);
    cudaMallocHost((void **)&h_exportBufferY, size_agent);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    bool firstFrame = true;

    for (size_t i = 0; i < ticks; i++)
    {
      // Launch the kernel with 256 threads per block.
      if (useBuffer1ForSim)
      {
        cudaTickKernel<<<numBlocks, threadsPerBlock>>>(
            d_bufferX1, d_bufferX2, d_bufferY1, d_bufferY2, agentDesX, agentDesY,
            agentWaypoints, agentWaypointsPitch, waypointX, waypointY, waypointR,
            numAgents, waypointIndex, numWaypoints);
      }
      else
      {
        cudaTickKernel<<<numBlocks, threadsPerBlock>>>(
            d_bufferX2, d_bufferX1, d_bufferY2, d_bufferY1, agentDesX, agentDesY,
            agentWaypoints, agentWaypointsPitch, waypointX, waypointY, waypointR,
            numAgents, waypointIndex, numWaypoints);
      }

      // Copy device simulation data to host export buffers.

      if (firstFrame)
      {
        cudaMemcpy(h_exportBufferX, (useBuffer1ForSim ? d_bufferX1 : d_bufferX2), size_agent, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_exportBufferY, (useBuffer1ForSim ? d_bufferY1 : d_bufferY2), size_agent, cudaMemcpyDeviceToHost);
        firstFrame = false;
      }
      if (useBuffer1ForSim)
      {
        cudaMemcpyAsync(h_exportBufferX, d_bufferX1, size_agent, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_exportBufferY, d_bufferY1, size_agent, cudaMemcpyDeviceToHost, stream);
      }
      else
      {
        cudaMemcpyAsync(h_exportBufferX, d_bufferX2, size_agent, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_exportBufferY, d_bufferY2, size_agent, cudaMemcpyDeviceToHost, stream);
      }

      // Now the host export buffers are ready.
      if (serialize)
      {
        serializeDataCuda(h_exportBufferX, h_exportBufferY, *this, *file);
      }

      tickCount++;
      useBuffer1ForSim = !useBuffer1ForSim;
    }

    cudaFreeHost(h_exportBufferX);
    cudaFreeHost(h_exportBufferY);

    return tickCount;
  }

  size_t Ped::Model::start_cuda(size_t maxSteps, bool serialize, std::ofstream *file)
  {

    float *agentStartX, *agentStartY, *agentDesX, *agentDesY, *waypointX, *waypointY, *waypointR;
    int *agentWaypoints, *waypointIndex;

    int numAgents = X.size();
    int numWaypoints = X_WP.size();

    size_t size_agent = numAgents * sizeof(float);
    size_t size_waypoint = numWaypoints * sizeof(int);
    size_t pitch;

    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    CUDA_CHECK(cudaMallocAsync(&agentStartX, size_agent, stream1));
    CUDA_CHECK(cudaMallocAsync(&agentStartY, size_agent, stream2));
    CUDA_CHECK(cudaMallocAsync(&agentDesX, size_agent, stream3))
    CUDA_CHECK(cudaMallocAsync(&agentDesY, size_agent, stream4));
    CUDA_CHECK(cudaMallocPitch(&agentWaypoints, &pitch, size_waypoint, size_agent));
    CUDA_CHECK(cudaMallocAsync(&waypointX, size_agent, stream1));
    CUDA_CHECK(cudaMallocAsync(&waypointY, size_agent, stream2));
    CUDA_CHECK(cudaMallocAsync(&waypointR, size_agent, stream3));
    CUDA_CHECK(cudaMallocAsync(&waypointIndex, size_agent, stream4));

    float *d_bufferX1 = nullptr, *d_bufferX2 = nullptr, *d_bufferY1 = nullptr, *d_bufferY2 = nullptr;
    cudaMallocAsync((void **)&d_bufferX1, size_agent, stream1);
    cudaMallocAsync((void **)&d_bufferX2, size_agent, stream2);
    cudaMallocAsync((void **)&d_bufferY1, size_agent, stream3);
    cudaMallocAsync((void **)&d_bufferY2, size_agent, stream4);

    CUDA_CHECK(cudaMemcpy(d_bufferX1, X.data(), size_agent, cudaMemcpyHostToDevice));
    cudaMemset(d_bufferX2, 0, size_agent);
    CUDA_CHECK(cudaMemcpy(d_bufferY1, Y.data(), size_agent, cudaMemcpyHostToDevice));
    cudaMemset(d_bufferY2, 0, size_agent);

    CUDA_CHECK(cudaMemset(agentDesX, 0, size_agent));
    CUDA_CHECK(cudaMemset(agentDesY, 0, size_agent));

    int *agentWaypointsTmp = new int[numAgents * numWaypoints];

    for (size_t i = 0; i < numAgents; i++)
    {
      // Insert the new waypoint at the front and push the rest back
      for (size_t j = numWaypoints - 1; j > 0; j--)
      {
        agentWaypointsTmp[i * numWaypoints + j] = agents.at(i)->getDestination().at(j - 1)->getid();
      }
      // Insert the first waypoint at the front
      agentWaypointsTmp[i * numWaypoints] = agents.at(i)->getDestination().at(numWaypoints - 1)->getid();
    }

    CUDA_CHECK(cudaMemcpy2DAsync(agentWaypoints, pitch, agentWaypointsTmp, size_waypoint, size_waypoint, numAgents,
                                 cudaMemcpyHostToDevice, stream1));

    CUDA_CHECK(cudaMemcpyAsync(waypointX, X_WP.data(), size_waypoint, cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(waypointY, Y_WP.data(), size_waypoint, cudaMemcpyHostToDevice, stream3));
    CUDA_CHECK(cudaMemcpyAsync(waypointR, R_WP.data(), size_waypoint, cudaMemcpyHostToDevice, stream4));

    int *waypointIndexTmp = new int[numAgents];

    for (size_t i = 0; i < numAgents; i++)
    {
      waypointIndexTmp[i] = agents.at(i)->getDestination().at(0)->getid();
    }

    CUDA_CHECK(cudaMemcpyAsync(waypointIndex, waypointIndexTmp, size_agent, cudaMemcpyHostToDevice, stream1));

    if (serialize && !file)
    {
      std::runtime_error("File needs to be open!");
    }

    size_t tickCount = tick_cuda(maxSteps, d_bufferX1, d_bufferX2, d_bufferY1, d_bufferY2, agentDesX, agentDesY, waypointX, waypointY, waypointR, agentWaypoints, pitch,
                                 waypointIndex, serialize, file);

    // Free allocated memory
    cudaFree(agentStartX);
    cudaFree(agentStartY);
    cudaFree(agentDesX);
    cudaFree(agentDesY);

    cudaFree(agentWaypoints);
    cudaFree(waypointX);
    cudaFree(waypointY);
    cudaFree(waypointR);
    cudaFree(waypointIndex);

    cudaFree(d_bufferX1);
    cudaFree(d_bufferX2);
    cudaFree(d_bufferY1);
    cudaFree(d_bufferY2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);

    return tickCount;
  }

  void Ped::Model::warmup()
  {
    cudaFree(0);
  }
} // namespace Ped
