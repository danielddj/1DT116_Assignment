//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <cmath>
#include <vector>
#include <ammintrin.h>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

int Ped::Model::numberOfThreads = omp_get_num_threads(); // by default use the number of threads available unless specified otherwise

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
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

	// EDIT HERE FOR ASSIGNMENT 1

	// Choose the implementation to use
	switch (implementation)
	{
	case SEQ:
		sequential_tick();
		break;
	case OMP:
		openmp_tick1();
		break;
	case THREADS:
		threads_tick();
		break;
	case VECTOR:
		vector_tick();
		break;
	case CUDA:
		std::cout << "CUDA tick not implemented." << std::endl;
		exit(1);
		break;
	default:
		std::cout << "Unknown implementation." << std::endl;
		exit(1);
	}
}

void Ped::Model::sequential_tick()
{
	for (Ped::Tagent *agent : agents)
	{
		// Compute the next desired position of the agent
		agent->computeNextDesiredPosition();

		// Set the agent's position to the desired position
		agent->setX(agent->getDesiredX());
		agent->setY(agent->getDesiredY());
	}
}

void Ped::Model::openmp_tick1()
{
#pragma omp parallel for shared(agents) schedule(guided) num_threads(numberOfThreads)
	for (int i = 0; i < agents.size(); i++)
	{
		// compute the next desired position of the agent
		agents[i]->computeNextDesiredPosition();

		// set the agent's position to the desired position
		agents[i]->setX(agents[i]->getDesiredX());
		agents[i]->setY(agents[i]->getDesiredY());
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
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
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
			agents[i]->setX(agents[i]->getDesiredX());
			agents[i]->setY(agents[i]->getDesiredY());
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

static inline __m128 my_round_ps(__m128 x)
{
	// Create a constant for +0.5.
	__m128 half = _mm_set1_ps(0.5);
	// Create a constant for -0.0; when ANDed with x, this extracts the sign.
	__m128 sign_mask = _mm_set1_ps(-0.0);
	// Use the sign of x to create a bias: for positive numbers, bias = +0.5;
	// for negative numbers, bias = -0.5.
	__m128 bias = _mm_or_ps(half, _mm_and_ps(x, sign_mask));
	// Add the bias to x.
	__m128 biased = _mm_add_ps(x, bias);
	// Truncate biased toward zero.
	__m128i i = _mm_cvttps_epi32(biased);
	// Convert the truncated integers back to double.
	return _mm_cvtepi32_ps(i);
}

void Ped::Model::vector_tick()
{
	size_t n = agents.size();

	// First, update each agent's destination pointer.
	// (This is done in scalar code because getNextDestination() may update internal state.)
	for (size_t i = 0; i < n; ++i)
	{
		agents[i]->destination = agents[i]->getNextDestination();
	}

	size_t i = 0;
	// Process groups of 4 agents at a time.
	for (; i + 3 < n; i += 4)
	{
		Ped::Tagent *a0 = agents[i];
		Ped::Tagent *a1 = agents[i + 1];
		Ped::Tagent *a2 = agents[i + 2];
		Ped::Tagent *a3 = agents[i + 3];

		// Check that all 4 agents have a valid destination.
		if (a0->destination != nullptr && a1->destination != nullptr &&
			a2->destination != nullptr && a3->destination != nullptr)
		{
			// Load current positions into __m128 registers.
			// _mm_set_ps expects values in order: [e3, e2, e1, e0]
			__m128 x_vec = _mm_set_ps((float)a3->x, (float)a2->x, (float)a1->x, (float)a0->x);
			__m128 y_vec = _mm_set_ps((float)a3->y, (float)a2->y, (float)a1->y, (float)a0->y);

			// Load destination coordinates.
			// Cast destination values from double to float.
			__m128 destX_vec = _mm_set_ps((float)a3->destination->getx(),
										  (float)a2->destination->getx(),
										  (float)a1->destination->getx(),
										  (float)a0->destination->getx());
			__m128 destY_vec = _mm_set_ps((float)a3->destination->gety(),
										  (float)a2->destination->gety(),
										  (float)a1->destination->gety(),
										  (float)a0->destination->gety());

			// Compute differences: diff = destination - current position.
			__m128 diffX = _mm_sub_ps(destX_vec, x_vec);
			__m128 diffY = _mm_sub_ps(destY_vec, y_vec);

			// Compute squared differences.
			__m128 diffX_sq = _mm_mul_ps(diffX, diffX);
			__m128 diffY_sq = _mm_mul_ps(diffY, diffY);
			__m128 sum = _mm_add_ps(diffX_sq, diffY_sq);

			// Compute the Euclidean distance (length).
			__m128 len = _mm_sqrt_ps(sum);

			// Compute ratios: diff / len.
			__m128 ratioX = _mm_div_ps(diffX, len);
			__m128 ratioY = _mm_div_ps(diffY, len);

			// Compute new positions: newPos = current position + ratio.
			__m128 newX = _mm_add_ps(x_vec, ratioX);
			__m128 newY = _mm_add_ps(y_vec, ratioY);

			// Round the computed positions using our custom SSE rounding function.
			__m128 roundedX = my_round_ps(newX);
			__m128 roundedY = my_round_ps(newY);

			// Store the results in temporary arrays.
			float resX[4], resY[4];
			_mm_storeu_ps(resX, roundedX);
			_mm_storeu_ps(resY, roundedY);

			// Update desired positions.
			// Note: _mm_set_ps packs values so that resX[0] corresponds to a0, etc.
			a0->desiredPositionX = (int)resX[0];
			a0->desiredPositionY = (int)resY[0];
			a1->desiredPositionX = (int)resX[1];
			a1->desiredPositionY = (int)resY[1];
			a2->desiredPositionX = (int)resX[2];
			a2->desiredPositionY = (int)resY[2];
			a3->desiredPositionX = (int)resX[3];
			a3->desiredPositionY = (int)resY[3];
		}
		else
		{
			// Fallback: if any agent does not have a destination, process them individually.
			a0->computeNextDesiredPosition();
			a1->computeNextDesiredPosition();
			a2->computeNextDesiredPosition();
			a3->computeNextDesiredPosition();
		}
	}

	// Process any remaining agents that did not form a complete batch of 4.
	for (; i < n; i++)
	{
		agents[i]->computeNextDesiredPosition();
	}

	// Finally, update each agent's actual position from its desired position.
	for (size_t i = 0; i < n; i++)
	{
		agents[i]->setX(agents[i]->getDesiredX());
		agents[i]->setY(agents[i]->getDesiredY());
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
}
