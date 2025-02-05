#pragma once

#include "ped_waypoint.h"
#include <deque>
#include <vector>
#include <xmmintrin.h>

namespace Ped
{

	struct AgentSoA
	{
		size_t numAgents;
		float *x;
		float *y;
		float *desiredX;
		float *desiredY;
		float *destX;
		float *destY;
		Twaypoint **destination;
		Twaypoint **lastDestination;
		std::vector<std::deque<Twaypoint *>> waypoints;

		AgentSoA(size_t numAgents) : numAgents(numAgents), waypoints(numAgents)
		{
			x = (float *)_mm_malloc(numAgents * sizeof(float), 16);
			y = (float *)_mm_malloc(numAgents * sizeof(float), 16);
			desiredX = (float *)_mm_malloc(numAgents * sizeof(float), 16);
			desiredY = (float *)_mm_malloc(numAgents * sizeof(float), 16);
			destX = (float *)_mm_malloc(numAgents * sizeof(float), 16);
			destY = (float *)_mm_malloc(numAgents * sizeof(float), 16);
			destination = (Twaypoint **)_mm_malloc(numAgents * sizeof(Twaypoint *), 16);
			lastDestination = (Twaypoint **)_mm_malloc(numAgents * sizeof(Twaypoint *), 16);

			for (size_t i = 0; i < numAgents; i++)
			{
				destination[i] = nullptr;
				lastDestination[i] = nullptr;
			}
		}

		~AgentSoA()
		{
			_mm_free(x);
			_mm_free(y);
			_mm_free(desiredX);
			_mm_free(desiredY);
			_mm_free(destX);
			_mm_free(destY);
			_mm_free(destination);
			_mm_free(lastDestination);
		}
	};
	class Tagent
	{
	public:
		size_t index;
		AgentSoA *soa;

		Tagent(size_t idx, AgentSoA *soaPtr);

		void computeNextDesiredPosition();
		void addWaypoint(Twaypoint *wp);
		Twaypoint *getNextDestination();

		// Inline accessors to mimic the old interface:
		inline int getX() const { return static_cast<int>(soa->x[index]); }
		inline int getY() const { return static_cast<int>(soa->y[index]); }
		inline void setX(int newX) { soa->x[index] = static_cast<float>(newX); }
		inline void setY(int newY) { soa->y[index] = static_cast<float>(newY); }
		inline int getDesiredX() const { return static_cast<int>(soa->desiredX[index]); }
		inline int getDesiredY() const { return static_cast<int>(soa->desiredY[index]); }
		inline void setDesiredX(int val) { soa->desiredX[index] = static_cast<float>(val); }
		inline void setDesiredY(int val) { soa->desiredY[index] = static_cast<float>(val); }
		inline Twaypoint *getDestination() const { return soa->destination[index]; }
		inline void setDestination(Twaypoint *dest) { soa->destination[index] = dest; }
	};

}
