//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include <math.h>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

namespace Ped
{
	std::vector<float> X;
	std::vector<float> Y;
	std::vector<float> desiredX;
	std::vector<float> desiredY;
	std::vector<float> destinationX;
	std::vector<float> destinationY;
	std::vector<float> destinationR;
}

int Ped::Model::numberOfThreads = omp_get_num_threads(); // by default use the number of threads available unless specified otherwise

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
		initCudaMemory();
	}

	// Populate global vectors with starting data
	for (int i = 0; i < num_agents; ++i)
	{
		agents[i]->setID(i); // Set ID for current agent
		X[i] = agents[i]->getX();
		Y[i] = agents[i]->getY();
		desiredX[i] = agents[i]->getDesiredX();
		desiredY[i] = agents[i]->getDesiredY();

		// Initialize as null values, to ensure we get new dest at the start
		destinationX[i] = NAN;
		destinationY[i] = NAN;
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
void Ped::Model::tick()
{
	// Choose the implementation to use
	switch (implementation)
	{
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
		/* 		vector_tick();
		 */
		break;
	case CUDA:
		cuda_tick();
		break;
	default:
		std::cout << "Unknown implementation." << std::endl;
		exit(1);
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
		X[i] = desiredX[i];
		Y[i] = desiredY[i];

		// TESTING (for visualization)
		agents[i]->setX(desiredX[i]);
		agents[i]->setY(desiredY[i]);
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

		// !! We could just store directly on X, Y instead of desiredX, Y
		// But looks like desired X, Y will be important in later assignments
		// so maybe keep it like this?

		// Store desired positions back into X, Y
		_mm_store_ps(&X[i], roundedX);
		_mm_store_ps(&Y[i], roundedY);

		/* TESTING ONLY (for visualization) remove for performance */

		agents[i]->setX(desiredX[i]);
		agents[i]->setY(desiredY[i]);

		agents[i + 1]->setX(desiredX[i + 1]);
		agents[i + 1]->setY(desiredY[i + 1]);

		agents[i + 2]->setX(desiredX[i + 2]);
		agents[i + 2]->setY(desiredY[i + 2]);

		agents[i + 3]->setX(desiredX[i + 3]);
		agents[i + 3]->setY(desiredY[i + 3]);
	}

	// Take care of any leftover loops/agents if num_agents not multiple of 4
	// Scalar
	for (; i < num_agents; i++)
	{
		agents[i]->computeNextDesiredPosition();
		X[i] = desiredX[i];
		Y[i] = desiredY[i];

		// TESTING visualization
		agents[i]->setX(desiredX[i]);
		agents[i]->setY(desiredY[i]);
	}
}

void Ped::Model::openmp_tick2()
{
// parallelize the outer loop for multiple ticks
#pragma omp parallel num_threads(numberOfThreads) shared(agents)
	{
// perform the tick operation for all agents
#pragma omp for schedule(static)
		for (int i = 0; i < agents.size(); i++)
		{
			agents[i]->computeNextDesiredPosition();
			X[i] = desiredX[i];
			Y[i] = desiredY[i];

			// TESTING (visualization)
			agents[i]->setX(desiredX[i]);
			agents[i]->setY(desiredY[i]);
		}
	}
}

void Ped::Model::threads_tick()
{
	// Helper function to process a range of agents
	auto processAgents = [](std::vector<Ped::Tagent *> &agents, int start, int end)
	{
		for (int i = start; i < end; i++)
		{
			agents[i]->computeNextDesiredPosition();
			X[i] = desiredX[i];
			Y[i] = desiredY[i];

			// TESTING (visualization)
			agents[i]->setX(desiredX[i]);
			agents[i]->setY(desiredY[i]);
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
			std::thread thread(processAgents, std::ref(agents), start, end);
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
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int>> takenPositions;
	for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt)
	{
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else
	{
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
	{

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
		{

			// Set the agent's position
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
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
		freeCudaMemory();
	}
}
