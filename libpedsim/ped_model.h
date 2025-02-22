//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ped_agent.h"
#include "ped_regionhandler.h"

namespace Ped {

class Tagent;
class Region_handler;

// The implementation modes for Assignment 1 + 2:
// chooses which implementation to use for tick()
enum IMPLEMENTATION {
  CUDA,
  VECTOR,
  OMP,
  PTHREAD,
  SEQ,
  THREADS,
  SEQ_REGION,
  OMP_REGION
};

class Model {
public:
  // Sets everything up
  void setup(std::vector<Tagent *> agentsInScenario,
             std::vector<Twaypoint *> destinationsInScenario,
             IMPLEMENTATION implementation, size_t start_regions = 4,
             size_t width = 160, size_t height = 120, size_t min_agents = 20,
             size_t max_agents = 100, bool resize = true);

  // Coordinates a time step in the scenario: move all agents by one step (if
  // applicable).
  void tick();
  void region_tick();
  void seq_region_tick();

  // tick for sequential implementation
  void sequential_tick();

  void vector_tick();

  // tick for openmp implementation
  void openmp_tick1();
  void openmp_tick2();

  // tick for c++ thread implementation
  void threads_tick();
  size_t start_cuda(size_t maxSteps, bool serialize,
                    std::ofstream *file = nullptr, bool timingMode = false);

  // Returns the agents of this scenario
  const std::vector<Tagent *> &getAgents() const { return agents; };

  // Adds an agent to the tree structure
  void placeAgent(const Ped::Tagent *a);

  // Cleans up the tree and restructures it. Worth calling every now and then.
  void cleanup();
  ~Model();

  // Returns the heatmap visualizing the density of agents
  int const *const *getHeatmap() const { return blurred_heatmap; };
  int getHeatmapSize() const;

  static int numberOfThreads;
  IMPLEMENTATION getImplementation() { return implementation; }
  static void warmup();

  // Moves an agent towards its next position
  void move(Ped::Tagent *agent);

private:
  // Denotes which implementation (sequential, parallel implementations..)
  // should be used for calculating the desired positions of
  // agents (Assignment 1)
  IMPLEMENTATION implementation;
  static const int MAP_WIDTH = 160;
  static const int MAP_HEIGHT = 120;
  bool attemptOccupyCell(int newX, int newY, int myId);

  // Clear a cell if (and only if) it’s still occupied by me.
  void releaseCellIfOccupiedByMe(int oldX, int oldY, int myId);
  std::atomic<int> occupant[MAP_WIDTH][MAP_HEIGHT];

  std::vector<float> X, Y;               // Position
  std::vector<float> desiredX, desiredY; // Desired movement
  std::vector<float> destinationX, destinationY,
      destinationR; // Destination points

  std::vector<float> X_WP, Y_WP, R_WP; // Waypoint Position
  void init_region(size_t start_regions, size_t width, size_t height,
                   size_t min_agents, size_t max_agents, bool resize);
  Ped::Region_handler *handler;

  size_t tick_cuda(size_t ticks, float *d_bufferX1, float *d_bufferX2,
                   float *d_bufferY1, float *d_bufferY2, float *agentDesX,
                   float *agentDesY, float *waypointX, float *waypointY,
                   float *waypointR, int *agentWaypoints,
                   size_t agentWaypointsPitch, int *waypointIndex,
                   bool serialize, std::ofstream *file = nullptr,
                   bool timingMode = false);
  // The agents in this scenario
  std::vector<Tagent *> agents;

  // The waypoints in this scenario
  std::vector<Twaypoint *> destinations;
  std::ofstream file;

  ////////////
  /// Everything below here won't be relevant until Assignment 3
  ///////////////////////////////////////////////

  // Returns the set of neighboring agents for the specified position
  set<const Ped::Tagent *> getNeighbors(int x, int y, int dist) const;

  ////////////
  /// Everything below here won't be relevant until Assignment 4
  ///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE *CELLSIZE

  // The heatmap representing the density of agents
  int **heatmap;

  // The scaled heatmap that fits to the view
  int **scaled_heatmap;

  // The final heatmap: blurred and scaled to fit the view
  int **blurred_heatmap;

  void setupHeatmapSeq();
  void updateHeatmapSeq();
  void resize_vectors();
  void populate_agent_vectors();
  void popluate_waypoint_vectors();

  inline __m128 compute_update_mask(__m128 destX, __m128 destY, __m128 destR,
                                    __m128 posX, __m128 posY);
  inline void update_agents(int start_idx, __m128 update_mask);
  inline void compute_new_desired_positions(__m128 posX, __m128 posY,
                                            __m128 destX, __m128 destY,
                                            __m128 &newX, __m128 &newY);
  inline void process_agents_simd(int i);
};
} // namespace Ped
#endif
