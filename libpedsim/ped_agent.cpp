#include "ped_agent.h"
#include <cmath>
#include <stdlib.h>
#include <stdexcept>

namespace Ped
{

	Ped::TagentSoA::TagentSoA(size_t idx, AgentSoA *soaPtr)
		: index(idx), soa(soaPtr)
	{
	}

	// Compute the next desired position for this agent.
	void TagentSoA::computeNextDesiredPosition()
	{
		getNextDestination();

		for (size_t i = 0; i < soa->numAgents; i += 4)
		{
			__m128 has_dest = _mm_cmpneq_ps(_mm_load_ps(reinterpret_cast<float *>(soa->currentWaypointIndex + i)), _mm_set1_ps(-1.0f));

			__m128 currX = _mm_load_ps(soa->x + i);
			__m128 currY = _mm_load_ps(soa->y + i);
			__m128 destX = _mm_load_ps(soa->destX + i);
			__m128 destY = _mm_load_ps(soa->destY + i);

			__m128 diffX = _mm_sub_ps(destX, currX);
			__m128 diffY = _mm_sub_ps(destY, currY);
			__m128 length = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

			__m128 desiredX = _mm_add_ps(currX, _mm_div_ps(diffX, length));
			__m128 desiredY = _mm_add_ps(currY, _mm_div_ps(diffY, length));

			__m128 desiredX_update = _mm_blendv_ps(currX, desiredX, has_dest);
			__m128 desiredY_update = _mm_blendv_ps(currY, desiredY, has_dest);

			_mm_store_ps(soa->desiredX + i, desiredX_update);
			_mm_store_ps(soa->desiredY + i, desiredY_update);
		}
	}

	// Add a waypoint for this agent.
	void Ped::TagentSoA::addWaypoint(Twaypoint *wp)
	{
		if (!soa->waypoints)
		{
			soa->waypoints = wp->soa;
		}

		bool added = false;
		for (size_t i = index * soa->maxWaypointPerAgent; i < (index + 1) * soa->maxWaypointPerAgent; i++)
		{
			if (soa->waypointList[i] == -1)
			{
				soa->waypointList[i] = wp->getid();
				added = true;
				soa->numWaypointsForAgent[index]++;
				break;
			}
		}
		if (!added)
		{
			throw std::runtime_error("No free waypoint slots available for this agent.");
		}
	}

	// Retrieve the next destination.
	void Ped::TagentSoA::getNextDestination()
	{
		for (size_t i = 0; i < soa->numAgents; i += 4)
		{
			__m128 waypointsIndex = _mm_load_ps(reinterpret_cast<float *>(soa->currentWaypointIndex + i));

			// Check if the agent has a destination.
			__m128 no_waypoint_mask = _mm_cmpeq_ps(waypointsIndex, _mm_set1_ps(-1.0f));

			// If the agent has no destination, update for those that do not have one.
			__m128 waypointIndex = _mm_load_ps(reinterpret_cast<float *>(soa->currentWaypointIndex + i));

			__m128 currX = _mm_load_ps(soa->x + i);
			__m128 currY = _mm_load_ps(soa->y + i);
			__m128 destX = _mm_load_ps(soa->destX + i);
			__m128 destY = _mm_load_ps(soa->destY + i);

			__m128 diffX = _mm_sub_ps(destX, currX);
			__m128 diffY = _mm_sub_ps(destY, currY);
			__m128 length = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

			// Load the radius of the waypoint.
			__m128 radius = _mm_set_ps(
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i + 3]]],
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i + 2]]],
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i + 1]]],
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i]]]);

			__m128 reached_destination_mask = _mm_cmplt_ps(length, radius);
			__m128 next_waypoint = _mm_add_ps(waypointIndex, _mm_set1_ps(1.0f));

			__m128 update_destination_mask = _mm_or_ps(reached_destination_mask, no_waypoint_mask);

			__m128 newWaypointIndex = _mm_blendv_ps(waypointIndex, next_waypoint, update_destination_mask);

			_mm_store_ps(reinterpret_cast<float *>(soa->currentWaypointIndex + i), newWaypointIndex);

			// Update the destination.
			for (size_t j = 0; j < 4; j++)
			{
				size_t agentIndex = i + j;
				// Get the updated waypoint slot for this agent.
				int waypointSlot = (soa->currentWaypointIndex[agentIndex] % soa->numWaypointsForAgent[agentIndex]);

				// If there is a valid waypoint (i.e. not -1)
				if (waypointSlot != -1)
				{
					soa->currentWaypointIndex[agentIndex] = waypointSlot;
					// Compute the linear index into the waypoint list for this agent.
					// (Assuming each agent has 'maxWaypointPerAgent' slots.)
					int listIndex = static_cast<int>(agentIndex * soa->maxWaypointPerAgent + waypointSlot);

					// Retrieve the waypoint ID stored in the waypoint list.
					int wp_id = soa->waypointList[listIndex];

					// Look up the waypoint pointer from its ID.
					// (Assume getWaypointById is a helper function that returns a Twaypoint* given an ID.)

					// Update destination coordinates from the waypoint.
					soa->destX[agentIndex] = soa->waypoints->x[wp_id];
					soa->destY[agentIndex] = soa->waypoints->y[wp_id];
				}
			}
		}
	}

	Ped::TagentOld::TagentOld(int posX, int posY)
	{
		Ped::TagentOld::init(posX, posY);
	}

	Ped::TagentOld::TagentOld(double posX, double posY)
	{
		Ped::TagentOld::init((int)round(posX), (int)round(posY));
	}

	void Ped::TagentOld::init(int posX, int posY)
	{
		x = posX;
		y = posY;
		destination = NULL;
		lastDestination = NULL;
	}

	void Ped::TagentOld::computeNextDesiredPosition()
	{
		destination = getNextDestination();
		if (destination == NULL)
		{
			// no destination, no need to
			// compute where to move to
			return;
		}

		double diffX = destination->getx() - x;
		double diffY = destination->gety() - y;
		double len = sqrt(diffX * diffX + diffY * diffY);
		desiredPositionX = (int)round(x + diffX / len);
		desiredPositionY = (int)round(y + diffY / len);
	}

	void Ped::TagentOld::addWaypoint(Twaypoint *wp)
	{
		waypoints.push_back(wp);
	}

	Ped::Twaypoint *Ped::TagentOld::getNextDestination()
	{
		Ped::Twaypoint *nextDestination = NULL;
		bool agentReachedDestination = false;

		if (destination != NULL)
		{
			// compute if agent reached its current destination
			double diffX = destination->getx() - x;
			double diffY = destination->gety() - y;
			double length = sqrt(diffX * diffX + diffY * diffY);
			agentReachedDestination = length < destination->getr();
		}

		if ((agentReachedDestination || destination == NULL) && !waypoints.empty())
		{
			// Case 1: agent has reached destination (or has no current destination);
			// get next destination if available
			waypoints.push_back(destination);
			nextDestination = waypoints.front();
			waypoints.pop_front();
		}
		else
		{
			// Case 2: agent has not yet reached destination, continue to move towards
			// current destination
			nextDestination = destination;
		}

		return nextDestination;
	}
}
