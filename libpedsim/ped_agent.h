//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// TAgent represents an agent in the scenario. Each
// agent has a position (x,y) and a number of destinations
// it wants to visit (waypoints). The desired next position
// represents the position it would like to visit next as it
// will bring it closer to its destination.
// Note: the agent will not move by itself, but the movement
// is handled in ped_model.cpp. 
//

#ifndef _ped_agent_h_
#define _ped_agent_h_ 1

#include <vector>
#include <deque>
//#include <immintrin.h>
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse3.h>
#include <set>

using namespace std;

namespace Ped {

    extern std::vector<float> X;
    extern std::vector<float> Y;
    extern std::vector<float> desiredX;
    extern std::vector<float> desiredY;
    extern std::vector<float> destinationX;
    extern std::vector<float> destinationY;
    extern std::vector<float> destinationR;

    class Twaypoint;

    class Tagent {
    public:

        Tagent(int posX, int posY);
        Tagent(double posX, double posY);

        // Returns the coordinates of the desired position
        int getDesiredX() const { return desiredPositionX; }
        int getDesiredY() const { return desiredPositionY; }

        void setDesiredX(int x) { desiredPositionX = x; }
        void setDesiredY(int y) { desiredPositionY = y; }

        // Sets the agent's position
        void setX(int newX) { x = newX; }
        void setY(int newY) { y = newY; }

        // Sets agent's ID
        void setID(int agentID) { id = agentID; }

        // Update the position according to get closer
        // to the current destination
        void computeNextDesiredPosition();

        // Call getNextDestination() 
        void callNextDestination();

        //Get agent id
        int getId() const { return id; };

        void setRegion(int region) { regionIndex = region; };
        int getRegion() const { return regionIndex; };
        const std::set<const Tagent*>& getCachedNeighbors() const {    
            return cachedNeighbors;
        }

        void setCachedNeighbors(const std::set<const Tagent*>& neighbors) { 
            cachedNeighbors = neighbors;
        }


        // Position of agent defined by x and y
        int getX() const { return x; };
        int getY() const { return y; };

        // Adds a new waypoint to reach for this agent
        void addWaypoint(Twaypoint* wp);

    private:
        Tagent() {};

        int id; // Agent index, for global arrays

        // The agent's current position
        int x;
        int y;

        int regionIndex;

        // The agent's desired next position
        int desiredPositionX;
        int desiredPositionY;

        // The current destination (may require several steps to reach)
        Twaypoint* destination;

        // cached neighbors
        std::set<const Tagent*> cachedNeighbors;


        // The last destination
        Twaypoint* lastDestination;

        // The queue of all destinations that this agent still has to visit
        deque<Twaypoint*> waypoints;

        // Internal init function 
        void init(int posX, int posY);

        // Returns the next destination to visit
        Twaypoint* getNextDestination();
    };
}

#endif