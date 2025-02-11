#include "ped_agent.h"
#include <cmath>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>

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

			__m128 desiredX_rounded = _mm_round_ps(desiredX, _MM_FROUND_TO_NEAREST_INT);
			__m128 desiredY_rounded = _mm_round_ps(desiredY, _MM_FROUND_TO_NEAREST_INT);

			__m128 desiredX_update = _mm_blendv_ps(currX, desiredX_rounded, has_dest);
			__m128 desiredY_update = _mm_blendv_ps(currY, desiredY_rounded, has_dest);

			_mm_store_ps(soa->desiredX + i, desiredX);
			_mm_store_ps(soa->desiredY + i, desiredY);
		}
	}

	// Add a waypoint for this agent.
	void Ped::TagentSoA::addWaypoint(Twaypoint *wp)
	{
		if (!soa->waypoints)
		{
			soa->waypoints = wp->soa;
		}

		// Check if the waypoint queue is full
		if (soa->numWaypointsForAgent[index] >= soa->maxWaypointPerAgent)
		{
			throw std::runtime_error("No free waypoint slots available for this agent.");
		}

		// Shift existing waypoints to the right to make space at the front
		for (size_t i = soa->numWaypointsForAgent[index]; i > 0; --i)
		{
			size_t currentPos = index * soa->maxWaypointPerAgent + i;
			size_t prevPos = index * soa->maxWaypointPerAgent + i - 1;
			soa->waypointList[currentPos] = soa->waypointList[prevPos];
		}

		if (soa->currentWaypointIndex[index] == -1)
		{
			soa->currentWaypointIndex[index] = 0;
			soa->destX[index] = wp->getx();
			soa->destY[index] = wp->gety();
		}

		// Add the new waypoint at the front
		soa->waypointList[index * soa->maxWaypointPerAgent] = wp->getid();
		soa->numWaypointsForAgent[index]++;
	}

	void flipArray(float maskArray[4])
	{
		// For an array of 4, we only need to swap 2 pairs.
		for (int i = 0; i < 2; i++)
		{
			float temp = maskArray[i];
			maskArray[i] = maskArray[3 - i];
			maskArray[3 - i] = temp;
		}
	}

// Retrieve the next destination.
#include <immintrin.h>
#include <stdexcept>
#include <iostream>

	void Ped::TagentSoA::getNextDestination()
	{
		// Process agents in groups of 4.
		for (size_t i = 0; i < soa->numAgents; i += 4)
		{
			__m128i waypointsIndexInt = _mm_load_si128(reinterpret_cast<const __m128i *>(soa->currentWaypointIndex + i));
			__m128 waypointsIndex = _mm_cvtepi32_ps(waypointsIndexInt);


			__m128 no_waypoint_mask = _mm_cmpeq_ps(_mm_round_ps(waypointsIndex, _MM_FROUND_TO_NEAREST_INT), _mm_set1_ps(-1.0f));
			__m128 valid_waypoints = _mm_blendv_ps(waypointsIndex, _mm_set1_ps(0.0f), no_waypoint_mask);


			__m128 currX = _mm_load_ps(soa->x + i);
			__m128 currY = _mm_load_ps(soa->y + i);
			__m128 destX = _mm_load_ps(soa->destX + i);
			__m128 destY = _mm_load_ps(soa->destY + i);

			__m128 diffX = _mm_sub_ps(destX, currX);
			__m128 diffY = _mm_sub_ps(destY, currY);
			__m128 length = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(diffX, diffX), _mm_mul_ps(diffY, diffY)));

			__m128 radius = _mm_set_ps(
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i + 3]]],
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i + 2]]],
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i + 1]]],
				soa->waypoints->r[soa->waypointList[soa->currentWaypointIndex[i]]]);


			__m128 reached_destination_mask = _mm_cmplt_ps(length, radius);
			__m128 next_waypoint = _mm_add_ps(valid_waypoints, _mm_set1_ps(1.0f));

			// Blend: if destination reached, use next_waypoint; else, keep the current value.
			__m128 newWaypointIndex = _mm_blendv_ps(waypointsIndex, next_waypoint, reached_destination_mask);


			__m128i newWaypointIndexInt = _mm_cvtps_epi32(newWaypointIndex);
			_mm_store_si128(reinterpret_cast<__m128i *>(soa->currentWaypointIndex + i), newWaypointIndexInt);

			for (int k = 0; k < 4; k++)
			{
				size_t idx = i + k;
				if (soa->currentWaypointIndex[idx] >= soa->numWaypointsForAgent[idx])
				{
					soa->currentWaypointIndex[idx] = soa->currentWaypointIndex[idx] % soa->numWaypointsForAgent[idx];
				}
			}


			alignas(16) float maskArray[4];
			_mm_store_ps(maskArray, reached_destination_mask);

			alignas(16) float newWaypointIndexArray[4];
			_mm_store_ps(newWaypointIndexArray, newWaypointIndex);

			for (int j = 0; j < 4; j++)
			{
				// A nonzero mask value indicates this lane (agent) has reached its destination.
				if (maskArray[j] != 0.0f)
				{
					int agentIndex = static_cast<int>(i + j);
					int waypointSlot = static_cast<int>(newWaypointIndexArray[j]);

					if (waypointSlot == -1)
					{
						throw std::runtime_error("No valid waypoint slot found for this agent.");
					}
					else
					{
						int listIndex = static_cast<int>(agentIndex * soa->maxWaypointPerAgent + waypointSlot);
						int wp_id = soa->waypointList[listIndex];
						soa->destX[agentIndex] = soa->waypoints->x[wp_id];
						soa->destY[agentIndex] = soa->waypoints->y[wp_id];

						// Debug output for the first agent.
						if (i == 0 && j == 0)
						{
							std::cout << "Agent " << agentIndex << ", waypoint slot " << waypointSlot << std::endl;
							std::cout << "Modulo result: " << soa->currentWaypointIndex[0] % soa->numWaypointsForAgent[0] << std::endl;
						}
					}
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
