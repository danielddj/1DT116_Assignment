//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include <iostream>
#include "ped_agent.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <math.h>

#include <stdlib.h>

Ped::Tagent::Tagent(int posX, int posY) {
    Ped::Tagent::init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY) {
    Ped::Tagent::init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::init(int posX, int posY) {
    x = posX;
    y = posY;
    destination = NULL;
    lastDestination = NULL;
}


void Ped::Tagent::callNextDestination() {
    destination = getNextDestination();
}


void Ped::Tagent::computeNextDesiredPosition() {
    destination = getNextDestination();
    
    if (destination == NULL) {
        // no destination, no need to compute where to move to
        return;
    } 

    double diffX = destination->getx() - X[id];
    double diffY = destination->gety() - Y[id];
    double len = sqrt(diffX * diffX + diffY * diffY);

    //Store into vectors instead of Agent object
    desiredX[id] = (float)round(X[id] + diffX / len);
    desiredY[id] = (float)round(Y[id] + diffY / len);	
    
}

void Ped::Tagent::addWaypoint(Twaypoint* wp) {
    waypoints.push_back(wp);
}

Ped::Twaypoint* Ped::Tagent::getNextDestination() {
    Ped::Twaypoint* nextDestination = NULL;
    bool agentReachedDestination = false;

    if (destination != NULL) {
        // compute if agent reached its current destination
        double diffX = destinationX[id] - X[id];
        double diffY = destinationY[id] - Y[id];
        double length = sqrt(diffX * diffX + diffY * diffY);
        agentReachedDestination = length < destination->getr();
    }

    if ((agentReachedDestination || destination == NULL) && !waypoints.empty()) {
        // Case 1: agent has reached destination (or has no current destination);
        // get next destination if available
        waypoints.push_back(destination);
        nextDestination = waypoints.front();
        waypoints.pop_front();
    } else {
        // Case 2: agent has not yet reached destination, continue to move towards
        // current destination
        nextDestination = destination;
    }

    // Store next destination in global vectors
    if (nextDestination != NULL) {
        destinationX[id] = nextDestination->getx();
        destinationY[id] = nextDestination->gety();
    } else {
        // Arrived
        destinationX[id] = X[id];
        destinationY[id] = Y[id];
    }
    return nextDestination;
}
