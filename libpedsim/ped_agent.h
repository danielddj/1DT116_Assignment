#pragma once

#include "ped_waypoint.h"
#include <deque>
#include <vector>
#include <nmmintrin.h>

namespace Ped
{
	class Tagent
	{
	public:
		virtual void computeNextDesiredPosition() = 0;
		virtual void addWaypoint(Twaypoint *wp) = 0;

		virtual ~Tagent() = default;

		virtual int getX() const noexcept = 0;
		virtual int getY() const noexcept = 0;
		virtual void setX(int newX) noexcept = 0;
		virtual void setY(int newY) noexcept = 0;

		virtual int getDesiredX() const noexcept = 0;
		virtual int getDesiredY() const noexcept = 0;
		virtual void setDesiredX(int val) noexcept = 0;
		virtual void setDesiredY(int val) noexcept = 0;

		virtual Twaypoint *getDestination() const noexcept = 0;
		virtual void setDestination(Twaypoint *dest) noexcept = 0;
	};

	class TagentSoA : public Tagent
	{
	public:
		struct AgentSoA
		{
			size_t numAgents;
			size_t maxWaypointPerAgent;
			size_t *numWaypointsForAgent;
			float *x;
			float *y;
			float *desiredX;
			float *desiredY;
			float *destX;
			float *destY;
			int *currentWaypointIndex;
			int *waypointList;
			Twaypoint **destination;
			Twaypoint **lastDestination;
			waypointSoA *waypoints;

			AgentSoA(size_t numAgents, size_t maxWaypointPerAgent) : numAgents(numAgents), maxWaypointPerAgent(maxWaypointPerAgent)
			{
				x = (float *)_mm_malloc(numAgents * sizeof(float), 16);
				y = (float *)_mm_malloc(numAgents * sizeof(float), 16);
				desiredX = (float *)_mm_malloc(numAgents * sizeof(float), 16);
				desiredY = (float *)_mm_malloc(numAgents * sizeof(float), 16);
				destX = (float *)_mm_malloc(numAgents * sizeof(float), 16);
				destY = (float *)_mm_malloc(numAgents * sizeof(float), 16);
				destination = (Twaypoint **)_mm_malloc(numAgents * sizeof(Twaypoint *), 16);
				lastDestination = (Twaypoint **)_mm_malloc(numAgents * sizeof(Twaypoint *), 16);
				currentWaypointIndex = (int *)_mm_malloc(numAgents * sizeof(int), 16);
				waypointList = (int *)_mm_malloc(numAgents * maxWaypointPerAgent * sizeof(int *), 16);
				numWaypointsForAgent = (size_t *)_mm_malloc(numAgents * sizeof(size_t), 16);

				for (size_t i = 0; i < numAgents * maxWaypointPerAgent; i++)
				{
					waypointList[i] = -1;
				}

				for (size_t i = 0; i < numAgents; i++)
				{
					currentWaypointIndex[i] = -1;
				}

				for (size_t i = 0; i < numAgents; i++)
				{
					numWaypointsForAgent[i] = 0;
				}

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
				_mm_free(currentWaypointIndex);
				_mm_free(waypointList);
				_mm_free(numWaypointsForAgent);
			}
		};

	public:
		size_t index;
		AgentSoA *soa;

		TagentSoA(size_t idx, AgentSoA *soaPtr);

		void computeNextDesiredPosition();
		void addWaypoint(Twaypoint *wp);
		void getNextDestination();

		// Inline accessors to mimic the old interface:
		inline int getX() const noexcept { return static_cast<int>(soa->x[index]); }
		inline int getY() const noexcept { return static_cast<int>(soa->y[index]); }
		inline void setX(int newX) noexcept { soa->x[index] = static_cast<float>(newX); }
		inline void setY(int newY) noexcept { soa->y[index] = static_cast<float>(newY); }

		inline int getDesiredX() const noexcept { return static_cast<int>(soa->desiredX[index]); }
		inline int getDesiredY() const noexcept { return static_cast<int>(soa->desiredY[index]); }
		inline void setDesiredX(int val) noexcept { soa->desiredX[index] = static_cast<float>(val); }
		inline void setDesiredY(int val) noexcept { soa->desiredY[index] = static_cast<float>(val); }

		inline Twaypoint *getDestination() const noexcept { return soa->destination[index]; }
		inline void setDestination(Twaypoint *dest) noexcept { soa->destination[index] = dest; }
	};

	class Twaypoint;

	class TagentOld : public Tagent
	{
	public:
		TagentOld(int posX, int posY);
		TagentOld(double posX, double posY);

		// Returns the coordinates of the desired position
		int getDesiredX() const noexcept { return desiredPositionX; }
		int getDesiredY() const noexcept { return desiredPositionY; }
		void setDesiredX(int val) noexcept { desiredPositionX = val; }
		void setDesiredY(int val) noexcept { desiredPositionY = val; }

		// Sets the agent's position
		void setX(int newX) noexcept { x = newX; }
		void setY(int newY) noexcept { y = newY; }

		// Update the position according to get closer
		// to the current destination
		void computeNextDesiredPosition();

		// Position of agent defined by x and y
		int getX() const noexcept { return x; };
		int getY() const noexcept { return y; };

		// Adds a new waypoint to reach for this agent
		void addWaypoint(Twaypoint *wp);

	private:
		TagentOld() {};

		// The agent's current position
		int x;
		int y;

		// The agent's desired next position
		int desiredPositionX;
		int desiredPositionY;

		// The current destination (may require several steps to reach)
		Twaypoint *destination;

		// The last destination
		Twaypoint *lastDestination;

		// The queue of all destinations that this agent still has to visit
		deque<Twaypoint *> waypoints;

		// Internal init function
		void init(int posX, int posY);

		// Returns the next destination to visit
		Twaypoint *getNextDestination();
	};

}
