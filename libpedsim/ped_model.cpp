//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//

/*TODO: Add regoins, how to create, store boundaries, CAS for switching. Start
w. static regions move to dynamic sizing. Split into multiple instances of
TAgent maybe, new class for bdry and have ptr to it in each agent?.
*/

#include "ped_model.h"
#include "ped_region.h"
#include "ped_regionhandler.h"
#include "ped_waypoint.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stack>
#include <thread>
#include <vector>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

namespace Ped {}

int Ped::Model::numberOfThreads =
    omp_get_num_threads(); // by default use the number of threads available
                           // unless specified otherwise

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario,
                       std::vector<Twaypoint *> destinationsInScenario,
                       IMPLEMENTATION implementation, size_t start_regions,
                       size_t width, size_t height, size_t min_agents,
                       size_t max_agents, bool resize) {
#ifndef NOCUDA
  // Convenience test: does CUDA work on this machine?
#else
  std::cout << "Not compiled for CUDA" << std::endl;
#endif

  // Set
  agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(),
                                      agentsInScenario.end());

  // Set up destinations
  destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(),
                                               destinationsInScenario.end());

  // Sets the chosen implemenation. Standard in the given code is SEQ
  this->implementation = implementation;

  // Set up heatmap (relevant for Assignment 4)
  setupHeatmapSeq();

  resize_vectors();

  popluate_waypoint_vectors();

  populate_agent_vectors();

  if (implementation == OMP_REGION || implementation == SEQ_REGION) {
    if (start_regions < 4) {
      std::runtime_error("Start_regions can not be less than 4");
    }

    init_region(start_regions, width, height, min_agents, max_agents, resize);

    for (int x = 0; x < MAP_WIDTH; x++) {
      for (int y = 0; y < MAP_HEIGHT; y++) {
        occupant[x][y].store(-1, std::memory_order_relaxed);
      }
    }
  }
}

/*
This function is called at each time step of the simulation.
Implementation:
1) it retrieves each agent
2) it calculates the next desired position of each agent
3) it sets the next position for each agent to the calculated desired position

For now, the agents are "ghosts" and cannot collide with each other.

*/
void Ped::Model::tick() {
  // Choose the implementation to use
  switch (implementation) {
  case SEQ:
    sequential_tick();
    break;
  case OMP:
    openmp_tick2();
    break;
  case THREADS:
    threads_tick();
    break;
  case VECTOR:
    vector_tick();
    break;
  case OMP_REGION:
    region_tick();
    break;
  case SEQ_REGION:
    seq_region_tick();
    break;
  default:
    std::cout << "Unknown implementation." << std::endl;
    exit(1);
  }

  for (int i = 0; i < agents.size(); ++i) {
    // TESTING (for visualization)
    agents[i]->setX(X[i]);
    agents[i]->setY(Y[i]);
  }
}

void Ped::Model::region_tick() {
  // Parallelize processing over regions (using OpenMP here).
  handler->tick_regions(this);
}
void Ped::Model::seq_region_tick() { handler->seq_tick_regions(this); }

void Ped::Model::sequential_tick() {
  int num_agents = agents.size();

  for (int i = 0; i < num_agents; ++i) {
    // Compute the next desired position of the agent
    agents[i]->computeNextDesiredPosition();

    X[i] = agents[i]->getDesiredX();
    Y[i] = agents[i]->getDesiredY();
  }
}

// The refactored vector_tick() function.
void Ped::Model::vector_tick() {
  const int num_agents = static_cast<int>(agents.size());
  int i = 0;

  // Process agents in SIMD blocks of 4.
  for (; i <= num_agents - 4; i += 4) {
    process_agents_simd(i);
  }

  // Process any remaining agents using scalar code.
  for (; i < num_agents; i++) {
    agents[i]->computeNextDesiredPosition();
    X[i] = desiredX[i];
    Y[i] = desiredY[i];
  }
}

void Ped::Model::openmp_tick2() {
// parallelize the outer loop for multiple ticks
#pragma omp parallel num_threads(numberOfThreads) shared(agents)
  {
// perform the tick operation for all agents
#pragma omp for schedule(static)
    for (int i = 0; i < agents.size(); i++) {
      agents[i]->computeNextDesiredPosition();
      X[i] = desiredX[i];
      Y[i] = desiredY[i];
    }
  }
}

void Ped::Model::threads_tick() {
  // store references to member vectors before launching threads
  std::vector<float> &X_ref = X;
  std::vector<float> &Y_ref = Y;
  std::vector<float> &desiredX_ref = desiredX;
  std::vector<float> &desiredY_ref = desiredY;
  std::vector<Ped::Tagent *> &agents_ref = agents;

  // Helper function to process a range of agents
  auto processAgents = [&](int start, int end) {
    for (int i = start; i < end; i++) {
      agents_ref[i]->computeNextDesiredPosition();
      X_ref[i] = desiredX_ref[i];
      Y_ref[i] = desiredY_ref[i];
    }
  };

  std::vector<std::thread> threads;

  // calculate the workload for each thread
  int totalAgents = agents.size();
  int agentsPerThread =
      std::ceil(static_cast<double>(totalAgents) / numberOfThreads);

  // launch threads and distribute the work to them
  for (int t = 0; t < numberOfThreads; t++) {

    // start and end index (of the agents) for current thread
    int start = t * agentsPerThread;
    int end = std::min(start + agentsPerThread, totalAgents);

    // do not launch if there is no work left ofc
    if (start < totalAgents) {
      std::thread thread(processAgents, start, end);
      threads.push_back(std::move(thread));
    }
  }

  // wait for all threads to finish
  for (std::thread &thread : threads) {
    thread.join();
  }
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.

void Ped::Model::move(Ped::Tagent *agent) {
  const int agentID = agent->getId();

  // Current (old) position
  int oldX = X[agentID];
  int oldY = Y[agentID];

  // Build the three candidate positions
  int dX = desiredX[agentID];
  int dY = desiredY[agentID];

  // Always try the desired position first
  std::vector<std::pair<int, int>> candidates;
  candidates.push_back({dX, dY});

  int diffX = dX - oldX;
  int diffY = dY - oldY;
  if (diffX == 0 || diffY == 0) {
    // agent moves straight N, S, E, or W
    candidates.push_back({dX + diffY, dY + diffX}); // 2
    candidates.push_back({dX - diffY, dY - diffX}); // 3
  } else {
    // agent moves diagonally
    candidates.push_back({dX, oldY}); // 2
    candidates.push_back({oldX, dY}); // 3
  }

  bool moved = false;
  for (auto &pos : candidates) {
    int newX = pos.first;
    int newY = pos.second;

    // Try to claim (newX, newY) via CAS (occupant[newX][newY] == -1 => myID)
    if (attemptOccupyCell(newX, newY, agentID)) {
      // Occupation succeeded, so unclaim the old cell now:
      releaseCellIfOccupiedByMe(oldX, oldY, agentID);

      // Update the agent's known position
      X[agentID] = newX;
      Y[agentID] = newY;
      moved = true;
      break;
    }
  }

  if (!moved) {
  }
}

bool Ped::Model::attemptOccupyCell(int x, int y, int myId) {
  // Boundary check:
  if (x < 0 || x >= MAP_WIDTH || y < 0 || y >= MAP_HEIGHT) {
    return false; // out of bounds
  }
  int expected = -1; // -1 means "free cell"
  // If occupant[x][y] is -1, replace it with myId (occupy).
  // compare_exchange_strong returns true on success.
  return occupant[x][y].compare_exchange_strong(
      expected, myId, std::memory_order_acq_rel, std::memory_order_relaxed);
}

void Ped::Model::releaseCellIfOccupiedByMe(int x, int y, int myId) {
  // If occupant[x][y] is me, set it back to -1.
  if (x >= 0 && x < MAP_WIDTH && y >= 0 && y < MAP_HEIGHT) {
    int expected = myId;
    occupant[x][y].compare_exchange_strong(
        expected, -1, std::memory_order_acq_rel, std::memory_order_relaxed);
  }
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents
/// (search field is a square in the current implementation)
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y,
                                                  int dist) const {

  // create the output list
  // ( It would be better to include only the agents close by, but this
  // programmer is lazy.)
  return set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
  // Nothing to do here right now.
}

Ped::Model::~Model() {
  std::for_each(agents.begin(), agents.end(),
                [](Ped::Tagent *agent) { delete agent; });
  std::for_each(destinations.begin(), destinations.end(),
                [](Ped::Twaypoint *destination) { delete destination; });
}

void Ped::Model::init_region(size_t start_regions, size_t width, size_t height,
                             size_t min_agents, size_t max_agents,
                             bool resize) {
  handler = new Region_handler(start_regions, resize, width, height, max_agents,
                               min_agents, agents);
}

void Ped::Model::popluate_waypoint_vectors() {
  for (int i = 0; i < destinations.size(); ++i) {
    X_WP[i] = destinations[i]->getx();
    Y_WP[i] = destinations[i]->gety();
    R_WP[i] = destinations[i]->getr();
  }
}

void Ped::Model::populate_agent_vectors() {
  for (int i = 0; i < agents.size(); ++i) {
    agents[i]->initialize(i, &X, &Y, &desiredX, &desiredY, &destinationX,
                          &destinationY, &destinationR);

    X[i] = agents[i]->getX();
    Y[i] = agents[i]->getY();
    desiredX[i] = agents[i]->getDesiredX();
    desiredY[i] = agents[i]->getDesiredY();

    // Initialize as null values, to ensure we getnew  dest at the start
    destinationX[i] = NAN;
    destinationY[i] = NAN;
  }
}

void Ped::Model::resize_vectors() {
  // Initialize global vectors
  int num_agents = agents.size();

  X.resize(num_agents);
  Y.resize(num_agents);
  desiredX.resize(num_agents);
  desiredY.resize(num_agents);
  destinationX.resize(num_agents);
  destinationY.resize(num_agents);
  destinationR.resize(num_agents);

  X_WP.resize(destinations.size());
  Y_WP.resize(destinations.size());
  R_WP.resize(destinations.size());
}

// Helper function: compute an update mask based on destination validity and
// reached condition.
inline __m128 Ped::Model::compute_update_mask(__m128 destX, __m128 destY,
                                              __m128 destR, __m128 posX,
                                              __m128 posY) {
  // Check for NaN in destination values.
  __m128 isNaN_X = _mm_cmpunord_ps(destX, destX);
  __m128 isNaN_Y = _mm_cmpunord_ps(destY, destY);
  __m128 isNaN_dest = _mm_or_ps(isNaN_X, isNaN_Y);

  // Compute the difference and Euclidean length.
  __m128 diffX = _mm_sub_ps(destX, posX);
  __m128 diffY = _mm_sub_ps(destY, posY);
  __m128 len = _mm_sqrt_ps(
      _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

  // Check if the agent has reached its destination (length < destR).
  __m128 reached_dest = _mm_cmplt_ps(len, destR);

  // Combine both conditions: if destination is invalid OR reached.
  return _mm_or_ps(isNaN_dest, reached_dest);
}

// Helper function: update agents whose mask indicates they should update.
inline void Ped::Model::update_agents(int start_idx, __m128 update_mask) {
  alignas(16) uint32_t mask_array[4];
  _mm_store_si128(reinterpret_cast<__m128i *>(mask_array),
                  _mm_castps_si128(update_mask));
  for (int lane = 0; lane < 4; lane++) {
    if (mask_array[lane] == 0xFFFFFFFF) {
      agents[start_idx + lane]->callNextDestination();
    }
  }
}

// Helper function: compute new desired positions for a SIMD block.
inline void Ped::Model::compute_new_desired_positions(__m128 posX, __m128 posY,
                                                      __m128 destX,
                                                      __m128 destY,
                                                      __m128 &newX,
                                                      __m128 &newY) {
  // Compute vector difference and its length.
  __m128 diffX = _mm_sub_ps(destX, posX);
  __m128 diffY = _mm_sub_ps(destY, posY);
  __m128 len = _mm_sqrt_ps(
      _mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

  // Avoid division by zero.
  __m128 zero = _mm_setzero_ps();
  __m128 mask = _mm_cmpneq_ps(len, zero);
  len =
      _mm_or_ps(_mm_and_ps(mask, len), _mm_andnot_ps(mask, _mm_set1_ps(1.0f)));

  // Normalize the difference vector.
  __m128 normX = _mm_div_ps(diffX, len);
  __m128 normY = _mm_div_ps(diffY, len);

  // Compute the new desired position.
  newX = _mm_add_ps(posX, normX);
  newY = _mm_add_ps(posY, normY);

  // Round to the nearest integer.
  newX = _mm_round_ps(newX, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  newY = _mm_round_ps(newY, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// Helper function: process a SIMD block (4 agents) starting at index i.
inline void Ped::Model::process_agents_simd(int i) {
  // Load current positions.
  __m128 posX = _mm_loadu_ps(&X[i]);
  __m128 posY = _mm_loadu_ps(&Y[i]);

  // Load destination data.
  __m128 destX = _mm_loadu_ps(&destinationX[i]);
  __m128 destY = _mm_loadu_ps(&destinationY[i]);
  __m128 destR = _mm_loadu_ps(&destinationR[i]);

  // Compute update mask and update agents accordingly.
  __m128 update_mask = compute_update_mask(destX, destY, destR, posX, posY);
  update_agents(i, update_mask);

  // Reload destination values in case they were updated.
  destX = _mm_loadu_ps(&destinationX[i]);
  destY = _mm_loadu_ps(&destinationY[i]);

  // Compute new desired positions.
  __m128 newX, newY;
  compute_new_desired_positions(posX, posY, destX, destY, newX, newY);

  // Store the computed desired positions.
  _mm_storeu_ps(&desiredX[i], newX);
  _mm_storeu_ps(&desiredY[i], newY);

  // Update current positions with the new desired positions.
  _mm_store_ps(&X[i], newX);
  _mm_store_ps(&Y[i], newY);
}
