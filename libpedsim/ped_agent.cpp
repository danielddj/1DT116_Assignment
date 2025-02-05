#include "ped_agent.h"
#include <cmath>
#include <stdlib.h>

namespace Ped
{

	Tagent::Tagent(size_t idx, AgentSoA *soaPtr)
		: index(idx), soa(soaPtr) {}

	// Compute the next desired position for this agent.
	void Tagent::computeNextDesiredPosition()
	{
		Twaypoint *dest = getNextDestination();
		if (dest == nullptr)
		{
			return; // No destination—nothing to do.
		}

		float currX = soa->x[index];
		float currY = soa->y[index];
		float diffX = static_cast<float>(dest->getx()) - currX;
		float diffY = static_cast<float>(dest->gety()) - currY;
		float len = sqrtf(diffX * diffX + diffY * diffY);

		soa->desiredX[index] = roundf(currX + diffX / len);
		soa->desiredY[index] = roundf(currY + diffY / len);
	}

	// Add a waypoint for this agent.
	void Tagent::addWaypoint(Twaypoint *wp)
	{
		soa->waypoints[index].push_back(wp);
	}

	// Retrieve the next destination.
	Twaypoint *Tagent::getNextDestination()
	{
		Twaypoint *currDest = soa->destination[index];
		bool agentReachedDestination = false;

		if (currDest != nullptr)
		{
			float currX = soa->x[index];
			float currY = soa->y[index];
			float diffX = static_cast<float>(currDest->getx()) - currX;
			float diffY = static_cast<float>(currDest->gety()) - currY;
			float len = sqrtf(diffX * diffX + diffY * diffY);
			agentReachedDestination = (len < currDest->getr());
		}

		Twaypoint *nextDest = nullptr;
		if ((agentReachedDestination || currDest == nullptr) && !soa->waypoints[index].empty())
		{
			if (currDest != nullptr)
				soa->waypoints[index].push_back(currDest);
			nextDest = soa->waypoints[index].front();
			soa->waypoints[index].pop_front();
		}
		else
		{
			nextDest = currDest;
		}

		soa->destination[index] = nextDest;
		return nextDest;
	}

}
