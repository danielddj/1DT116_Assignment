//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_waypoint.h"

#include <cmath>

#include <stdlib.h>

// Constructor: Sets some intial values. The agent has to pass within the given radius.
Ped::Twaypoint::Twaypoint(double px, double py, double pr, waypointSoA *soa) : soa(soa)
{
    id = soa->staticid;
    soa->x[soa->staticid] = px;
    soa->y[soa->staticid] = py;
    soa->r[soa->staticid] = pr;
    soa->staticid++;
};

// Constructor - sets the most basic parameters.
Ped::Twaypoint::Twaypoint(waypointSoA *soa) : soa(soa)
{
    id = soa->staticid;
    soa->x[soa->staticid] = 0;
    soa->y[soa->staticid] = 0;
    soa->r[soa->staticid] = 0;
    soa->staticid++;
};

Ped::Twaypoint::~Twaypoint() {};
