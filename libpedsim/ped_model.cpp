//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_region.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include <math.h>
#include <atomic>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>


namespace Ped
{
}

int Ped::Model::numberOfThreads = omp_get_num_threads(); // by default use the number of threads available unless specified otherwise

void Ped::Model::initializeRegions(int num_regions_x, int num_regions_y) {
    int region_w = 160 / num_regions_x; // Region width
    int region_h = 120 / num_regions_y; // Region height

    regions.clear(); // Reset regions just in case

    for (int i = 0; i < num_regions_x; ++i) {
        for (int j = 0; j < num_regions_y; ++j) {
            int x_start = i * region_w;
            int x_end = (i + 1) * region_w;
            int y_start = j * region_h;
            int y_end = (j + 1) * region_h;
            regions.emplace_back(new Ped::Region(x_start, x_end, y_start, y_end));
        }
    }
    // Assign agents to regions
    int num_agents = agents.size();
    int num_regions = regions.size();

    for (int i=0; i<num_agents; ++i) {
        int x = X[i];
        int y = Y[i];

        for (int j=0; j<num_regions; ++j) {
            if (x >= regions[j]->getXStart() && x < regions[j]->getXEnd() &&
            y >= regions[j]->getYStart() && y < regions[j]->getYEnd()) {
            regions[j]->addAgent(agents[i]);

            break; // If agent is assigned to a region, stop
        }
        }        
    }
}

void Ped::Model::moveAgentsInRegion(Ped::Region *region) {
    if (region == nullptr) return;

    std::vector<Ped::Tagent *> &region_agents = region->getAgents();
    int num_agents = region_agents.size();

    for (int i = 0; i < num_agents; ++i) {
        if (region_agents[i] == nullptr) continue;

        int id = region_agents[i]->getId();

        // Get desiredX, desiredY
        region_agents[i]->computeNextDesiredPosition();

        int old_x = X[id];
        int old_y = Y[id];
        int new_x = desiredX[id];
        int new_y = desiredY[id];

        // Try to move to desired position
        if (!region->attemptMove(old_x, old_y, new_x, new_y, id)) {
            // If Position is taken, try other nearby positions (random)
            int offsets[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}}; // Up right down left
            std::random_shuffle(std::begin(offsets), std::end(offsets)); 

            for (int j = 0; j < 4; j++) {
                int alt_x = old_x + offsets[j][0];
                int alt_y = old_y + offsets[j][1];

                if (region->attemptMove(old_x, old_y, alt_x, alt_y, id)) {
                    X[id] = alt_x;
                    Y[id] = alt_y;
                    break; // IF move successful, break
                }
            }
        } else {
            X[id] = new_x;
            Y[id] = new_y;
        }
    }
}



void Ped::Model::moveAgentsCrossRegion() {
    for (Ped::Region *region : regions) {
        std::vector<Ped::Tagent *> &region_agents = region->getAgents();
        std::vector<Ped::Tagent *> agents_to_reassign;

        // Check each agent in the region
        for (size_t i = 0; i < region_agents.size(); ) {
            Ped::Tagent *agent = region_agents[i];
            int x = X[agent->getId()];
            int y = Y[agent->getId()];

            // If the agent is still in the region, move to next agent
            if (x >= region->getXStart() && x < region->getXEnd() &&
                y >= region->getYStart() && y < region->getYEnd()) {
                ++i;
                continue;
            }

            // If Agent moved out of region bounds, remove it from current region
            agents_to_reassign.push_back(agent);
            region_agents.erase(region_agents.begin() + i);
        }

        // Reassign moved agents to correct region
        for (Ped::Tagent *agent : agents_to_reassign) {
            int new_x = X[agent->getId()];
            int new_y = Y[agent->getId()];

            for (Ped::Region *new_region : regions) {
                if (new_x >= new_region->getXStart() && new_x < new_region->getXEnd() &&
                    new_y >= new_region->getYStart() && new_y < new_region->getYEnd()) {
                    new_region->addAgent(agent);
                    break;
                }
            }
        }
    }
}



void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
    // Convenience test: does CUDA work on this machine?
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif

    // Set
    agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

    // Set up destinations
    destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(), destinationsInScenario.end());

    // Sets the chosen implemenation. Standard in the given code is SEQ
    this->implementation = implementation;

    // Set up heatmap (relevant for Assignment 4)
    setupHeatmapSeq();

    // Initialize global vectors
    int num_agents = agents.size();
    X.resize(num_agents);
    Y.resize(num_agents);
    desiredX.resize(num_agents);
    desiredY.resize(num_agents);
    destinationX.resize(num_agents);
    destinationY.resize(num_agents);
    destinationR.resize(num_agents);

    if (implementation == CUDA)
    {
        //initCudaMemory();
    }

    // Populate global vectors with starting data
    for (int i = 0; i < num_agents; ++i)
    {
        agents[i]->initialize(i, &X, &Y, &desiredX, &desiredY, &destinationX, &destinationY, &destinationR);
        X[i] = agents[i]->getX();
        Y[i] = agents[i]->getY();
        desiredX[i] = agents[i]->getDesiredX();
        desiredY[i] = agents[i]->getDesiredY();

        // Initialize as null values, to ensure we get new dest at the start
        destinationX[i] = NAN;
        destinationY[i] = NAN;
    }

    initializeRegions(2,2); // starting with 4 regions

}

/*
This function is called at each time step of the simulation.
Implementation:
1) it retrieves each agent
2) it calculates the next desired position of each agent
3) it sets the next position for each agent to the calculated desired position

For now, the agents are "ghosts" and cannot collide with each other.

*/
void Ped::Model::tick()
{
    // Choose the implementation to use
    switch (implementation)
    {
    case SEQ:
        sequential_tick();
        break;
    case OMP:
        openmp_tick();
        break;
    case THREADS:
        threads_tick();
        break;
    case VECTOR:
        vector_tick();
        break;
    case CUDA:
        //cuda_tick();
        break;
    default:
        std::cout << "Unknown implementation." << std::endl;
        exit(1);
    }

    for (int i=0; i<agents.size(); ++i) {
        // TESTING (for visualization)
        agents[i]->setX(X[i]);
        agents[i]->setY(Y[i]);
    }
}

void Ped::Model::sequential_tick()
{
    int num_agents = agents.size();

    for (int i = 0; i < num_agents; ++i)
    {
        // Compute the next desired position of the agent
        agents[i]->computeNextDesiredPosition();

        // Set the agent's position to the desired position
        move(agents[i]);
    }
}

void Ped::Model::vector_tick()
{
    int num_agents = agents.size();

    // Initialize outside of loop, to access it for remaining agents
    int i = 0;

    for (; i <= num_agents - 4; i += 4)
    {
        int num_agents = agents.size();

        // Load destination and current position for agents i to i+3
        __m128 destX = _mm_loadu_ps(&destinationX[i]);
        __m128 destY = _mm_loadu_ps(&destinationY[i]);
        __m128 destR = _mm_loadu_ps(&destinationR[i]);
        __m128 posX = _mm_loadu_ps(&X[i]);
        __m128 posY = _mm_loadu_ps(&Y[i]);

        // Check if destinationX, destinationY are null
        __m128 isNaN_X = _mm_cmpunord_ps(destX, destX);
        __m128 isNaN_Y = _mm_cmpunord_ps(destY, destY);
        __m128 isNaN_dest = _mm_or_ps(isNaN_X, isNaN_Y);

        // Compute difference, destination - current pos
        __m128 diffX = _mm_sub_ps(destX, posX);
        __m128 diffY = _mm_sub_ps(destY, posY);

        // Compute euclidean distance, sqrt(diffX * diffX + diffY * diffY)
        __m128 len = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

        // Check if agents reached destination (euclidian distance < R)
        __m128 reached_dest = _mm_cmplt_ps(len, destR);

        // Combine conditions (agent has arrived OR destination is NULL)
        __m128 shouldUpdate = _mm_or_ps(isNaN_dest, reached_dest);

        // Store results into an array for scalar checks
        alignas(16) uint32_t update_array[4];
        _mm_store_si128((__m128i *)update_array, _mm_castps_si128(shouldUpdate));

        // Call getNextDestination only for agents that meet conditions
        for (int lane = 0; lane < 4; lane++)
        {
            int idx = i + lane;
            if (update_array[lane] == 0xFFFFFFFF)
            {
                agents[idx]->callNextDestination();
            }
        }

        // Update vectors with new values
        destX = _mm_loadu_ps(&destinationX[i]);
        destY = _mm_loadu_ps(&destinationY[i]);

        // Recompute difference (for updated destinations)
        diffX = _mm_sub_ps(destX, posX);
        diffY = _mm_sub_ps(destY, posY);

        // Recompute Euclidean distance
        len = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

        // Avoid division by zero (maybe unnecessary?)
        __m128 mask = _mm_cmpneq_ps(len, _mm_setzero_ps());
        len = _mm_or_ps(_mm_and_ps(mask, len), _mm_andnot_ps(mask, _mm_set1_ps(1.0f)));

        // Normalize (diffX / len, diffY / len)
        __m128 normX = _mm_div_ps(diffX, len);
        __m128 normY = _mm_div_ps(diffY, len);

        // New desired position
        __m128 newX = _mm_add_ps(posX, normX);
        __m128 newY = _mm_add_ps(posY, normY);

        // Rounding
        __m128 roundedX = _mm_round_ps(newX, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128 roundedY = _mm_round_ps(newY, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // Store results
        _mm_storeu_ps(&desiredX[i], roundedX);
        _mm_storeu_ps(&desiredY[i], roundedY);

        // Now we have the new desired positions for agents i to i+3
        // Move agents according to these
        move(agents[i]);

    }

    // Take care of any leftover loops/agents if num_agents not multiple of 4
    // Scalar
    for (; i < num_agents; i++)
    {
        agents[i]->computeNextDesiredPosition();
        move(agents[i]);
    }
}

void Ped::Model::openmp_tick_original()
{
// parallelize the outer loop for multiple ticks
#pragma omp parallel num_threads(numberOfThreads) shared(agents)
    {
// perform the tick operation for all agents
#pragma omp for schedule(static)
        for (int i = 0; i < agents.size(); i++)
        {
            agents[i]->computeNextDesiredPosition();
            move(agents[i]);
        }
    }
    moveAgentsCrossRegion();
}

void Ped::Model::openmp_tick() {

    //cout << "number of regions :" << regions.size() << endl;

    #pragma omp parallel num_threads(numberOfThreads) shared(agents)
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < regions.size(); ++i) {
            moveAgentsInRegion(regions[i]);
        }
    }
}


void Ped::Model::threads_tick()
{
    // store references to member vectors before launching threads
    std::vector<float>& X_ref = X;
    std::vector<float>& Y_ref = Y;
    std::vector<float>& desiredX_ref = desiredX;
    std::vector<float>& desiredY_ref = desiredY;
    std::vector<Ped::Tagent*>& agents_ref = agents;

    // Helper function to process a range of agents
    auto processAgents = [&](int start, int end)
    {
        for (int i = start; i < end; i++)
        {
            agents_ref[i]->computeNextDesiredPosition();
            move(agents[i]);
        }
    };

    std::vector<std::thread> threads;

    // calculate the workload for each thread
    int totalAgents = agents.size();
    int agentsPerThread = std::ceil(static_cast<double>(totalAgents) / numberOfThreads);

    // launch threads and distribute the work to them
    for (int t = 0; t < numberOfThreads; t++)
    {

        // start and end index (of the agents) for current thread
        int start = t * agentsPerThread;
        int end = std::min(start + agentsPerThread, totalAgents);
        
        // do not launch if there is no work left ofc
        if (start < totalAgents)
        {
            std::thread thread(processAgents, start, end);
            threads.push_back(std::move(thread));
        }

    }

    // wait for all threads to finish
    for (std::thread &thread : threads)
    {
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
    int id = agent->getId();  // Get agent's ID for vector indexing

    // Retrieve agent's current position
    int currentX = X[id];
    int currentY = Y[id];

    // Retrieve agent's desired position
    int desiredPosX = desiredX[id];
    int desiredPosY = desiredY[id];

    // Search for neighboring agents
    set<const Ped::Tagent *> neighbors = getNeighbors(currentX, currentY, 2);

    // Retrieve their positions using global vectors
    std::vector<std::pair<int, int>> takenPositions;
    for (const Ped::Tagent *neighbor : neighbors) {
        int neighborId = neighbor->getId();
        takenPositions.emplace_back(X[neighborId], Y[neighborId]);
    }

    // Compute alternative positions
    std::vector<std::pair<int, int>> prioritizedAlternatives;
    prioritizedAlternatives.emplace_back(desiredPosX, desiredPosY);

    int diffX = desiredPosX - currentX;
    int diffY = desiredPosY - currentY;
    std::pair<int, int> p1, p2;

    if (diffX == 0 || diffY == 0) {
        // Agent wants to walk straight (N, S, W, E)
        p1 = {desiredPosX + diffY, desiredPosY + diffX};
        p2 = {desiredPosX - diffY, desiredPosY - diffX};
    } else {
        // Agent wants to walk diagonally
        p1 = {desiredPosX, currentY};
        p2 = {currentX, desiredPosY};
    }

    prioritizedAlternatives.push_back(p1);
    prioritizedAlternatives.push_back(p2);

    // Find the first empty alternative position
    for (const auto &alternative : prioritizedAlternatives) {
        if (std::find(takenPositions.begin(), takenPositions.end(), alternative) == takenPositions.end()) {
            // Update position using vectors instead of agent methods
            X[id] = alternative.first;
            Y[id] = alternative.second;
            return;
        }
    }
}


/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const
{

    // create the output list
    // ( It would be better to include only the agents close by, but this programmer is lazy.)
    return set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup()
{
    // Nothing to do here right now.
}

Ped::Model::~Model()
{
    std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent)
                  { delete agent; });
    std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination)
                  { delete destination; });

    if (implementation == CUDA)
    {
        //freeCudaMemory();
    }
}