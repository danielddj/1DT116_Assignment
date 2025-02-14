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
#include <immintrin.h>

using namespace std;

namespace Ped {
    class Twaypoint;

    class Tagent {
    public:

        Tagent(int posX, int posY);
        Tagent(double posX, double posY);

        // Initializes with pointers to vectors that store agent data
        void initialize(int i, 
            std::vector<float>* X, std::vector<float>* Y, 
            std::vector<float>* desiredX, std::vector<float>* desiredY,
            std::vector<float>* destinationX, std::vector<float>* destinationY, std::vector<float>* destinationR);

        // Returns the coordinates of the desired position
        int getDesiredX() const { return desiredPositionX; }
        int getDesiredY() const { return desiredPositionY; }

        // Sets the agent's position
        void setX(int newX) { x = newX; }
        void setY(int newY) { y = newY; }

        // Update the position to get closer to the current destination
        void computeNextDesiredPosition();

        // Call getNextDestination() 
        void callNextDestination();

        //Get agent id
        int getId() const { return id; };

        // Position of agent defined by x and y
        int getX() const { return x; };
        int getY() const { return y; };

        // Adds a new waypoint to reach for this agent
        void addWaypoint(Twaypoint* wp);

    private:
        Tagent() {};

        // Agent index, for vectors
        int id; 

        // Pointers to vectors
        std::vector<float>* X, Y;
        std::vector<float>* desiredX, desiredY;
        std::vector<float>* destinationX, destinationY, destinationR;

        // The agent's current position
        int x;
        int y;

        // The agent's desired next position
        int desiredPositionX;
        int desiredPositionY;

        // The current destination (may require several steps to reach)
        Twaypoint* destination;

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