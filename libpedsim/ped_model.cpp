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

	// Set up the SoA layout (relevant for Assignment 2)
	if (implementation == VECTOR)
	{
		
		int numberOfAgents = agents.size();

		desiredPositionX = FloatVectorAligned32(numberOfAgents);
		desiredPositionY = FloatVectorAligned32(numberOfAgents);
		x = FloatVectorAligned32(numberOfAgents);
		y = FloatVectorAligned32(numberOfAgents);
		destinationsX = FloatVectorAligned32(numberOfAgents);
		destinationsY = FloatVectorAligned32(numberOfAgents);
		destinationsR = FloatVectorAligned32(numberOfAgents);
		destinationsID = IntVectorAligned32(numberOfAgents);

		int largestQueue = -1;




		for (int i = 0 ; i < agents.size(); i++)
		{
			x[i] = agents[i]->getX();
			y[i] = agents[i]->getY();

			desiredPositionX[i] = NAN;
			desiredPositionY[i] = NAN;
			destinationsX[i] = NAN;
			destinationsY[i] = NAN;
			destinationsR[i] = NAN;
			destinationsID[i] = NAN;

			// Set up the waypoints
			std::deque<float> queueX;
			std::deque<float> queueY;
			std::deque<float> queueR;
			std::deque<int> queueID;



			for (Ped::Twaypoint *waypoint : agents[i]->getWaypoints())
			{
				queueX.push_back(waypoint->getx());
				queueY.push_back(waypoint->gety());
				queueR.push_back(waypoint->getr());
				queueID.push_back(waypoint->getid());
			}

			if (queueX.size() > largestQueue)
			{
				largestQueue = queueX.size();
			}
			waypointsX.push_back(queueX);
			waypointsY.push_back(queueY);
			waypointsR.push_back(queueR);
			waypointsID.push_back(queueID);
		}

	}


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

void Ped::Model::vector_tick()
{
	int num_agents = x.size();

	// handle the destiantions
	Ped::Tagent::getNextDestinationSIMD(x.data(), y.data(), destinationsX.data(), destinationsY.data(), destinationsR.data(), destinationsID.data(), destinationsX.data(), destinationsY.data(), destinationsR.data(), destinationsID.data(), num_agents, waypointsX, waypointsY, waypointsR, waypointsID);

	Ped::Tagent::computeDesiredPositionsSIMD2(x.data(), y.data(), destinationsX.data(), destinationsY.data(), desiredPositionX.data(), desiredPositionY.data(), num_agents);

	assert(desiredPositionX.size() == num_agents && desiredPositionY.size() == num_agents);
	assert(x.size() == num_agents && y.size() == num_agents);
	assert(waypointsX.size() == num_agents && waypointsY.size() == num_agents && waypointsR.size() == num_agents && waypointsID.size() == num_agents);

	// Update the positions of the agents
	int i = 0;
	for (; i < num_agents; i++)
	{
		x[i] = desiredPositionX[i];
		y[i] = desiredPositionY[i];

		agents[i]->setX(x[i]);
		agents[i]->setY(y[i]);
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
