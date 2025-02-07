//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_waypoint.h"
#include <math.h>

#include <stdlib.h>

Ped::Tagent::Tagent(int posX, int posY)
{
    Ped::Tagent::init(posX, posY);
}

Ped::Tagent::Tagent(double posX, double posY)
{
    Ped::Tagent::init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::init(int posX, int posY)
{
    x = posX;
    y = posY;
    destination = NULL;
    lastDestination = NULL;
}

void Ped::Tagent::computeNextDesiredPosition()
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

void Ped::Tagent::addWaypoint(Twaypoint *wp)
{
    waypoints.push_back(wp);
}

// get the waypoints
std::deque<Ped::Twaypoint *> Ped::Tagent::getWaypoints()
{
    return waypoints;
}

/*
What the funciton needs:
- x, y, destX, destY, destR, destID
- outDestX, outDestY, outDestR, outDestID
- waypoitns
*/

Ped::Twaypoint *Ped::Tagent::getNextDestination()
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

/*
void Ped::Tagent::getNextDestinationSIMD(const float *x, const float *y, const float *destX, const float *destY, const float *destR, const int *destID, float *outDestX, float *outDestY, float *outDestR, int *outDestID, int numAgents, std::vector<std::deque<float>> &waypointsX, std::vector<std::deque<float>> &waypointsY, std::vector<std::deque<float>> &waypointsR, std::vector<std::deque<int>> &waypointsID)
{
    int i = 0;
    // Process in batches of 4 floats (128 bits)
    for (; i <= numAgents - 4; i += 4)
    {
        // agent's current position
        float32x4_t xVec = vld1q_f32(&x[i]);
        float32x4_t yVec = vld1q_f32(&y[i]);
        // agent's current destination
        float32x4_t destXVec = vld1q_f32(&destX[i]);
        float32x4_t destYVec = vld1q_f32(&destY[i]);
        float32x4_t destRVec = vld1q_f32(&destR[i]);
        int32x4_t destIDVec = vld1q_s32(&destID[i]);
        // agent's next destination
        float32x4_t outDestXVec = vld1q_f32(&outDestX[i]);
        float32x4_t outDestYVec = vld1q_f32(&outDestY[i]);
        float32x4_t outDestRVec = vld1q_f32(&outDestR[i]);
        int32x4_t outDestIDVec = vld1q_s32(&outDestID[i]);

        // mask for valid destinations (where NAN in dest vector means no valid destination)
        uint32x4_t isNotNaN_X = vceqq_f32(destXVec, destXVec);
        uint32x4_t isNotNaN_Y = vceqq_f32(destYVec, destYVec);
        uint32x4_t mask_valid_dest = vandq_u32(isNotNaN_X, isNotNaN_Y);
        uint32x4_t isNaN_X = vmvnq_u32(vceqq_f32(destXVec, destXVec));
        uint32x4_t isNaN_Y = vmvnq_u32(vceqq_f32(destYVec, destYVec));
        uint32x4_t mask_invalid_dest = vorrq_u32(isNaN_X, isNaN_Y);

        // compute if agent reached its current destination
        // Zero out invalid destinations BEFORE computation
        float32x4_t maskedDestX = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(destXVec), mask_valid_dest));
        float32x4_t maskedDestY = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(destYVec), mask_valid_dest));

        // Compute only for valid destinations
        float32x4_t diffX = vsubq_f32(maskedDestX, xVec);
        float32x4_t diffY = vsubq_f32(maskedDestY, yVec);
        float32x4_t diffX_sq = vmulq_f32(diffX, diffX);
        float32x4_t diffY_sq = vmulq_f32(diffY, diffY);
        float32x4_t length = vsqrtq_f32(vaddq_f32(diffX_sq, diffY_sq));

        // Perform comparison only for valid values
        uint32x4_t reachedDest = vcltq_f32(length, destRVec);

        // Mask out invalid results at the end
        reachedDest = vandq_u32(reachedDest, mask_valid_dest);

        // int32x4_t waypointsEmptyMask = {waypointsX[i].empty(), waypointsX[i + 1].empty(), waypointsX[i + 2].empty(), waypointsX[i + 3].empty()};

        uint32_t maskArr[4] = {
            waypointsX[i].empty() ? 0xFFFFFFFF : 0,
            waypointsX[i + 1].empty() ? 0xFFFFFFFF : 0,
            waypointsX[i + 2].empty() ? 0xFFFFFFFF : 0,
            waypointsX[i + 3].empty() ? 0xFFFFFFFF : 0};
        uint32x4_t waypointsEmptyMask = vld1q_u32(maskArr);

        // turn this into mask:  if ((agentReachedDestination || destination == NULL) && !waypoints.empty())
        // Case 1: agent has reached destination (or has no current destination)

        // isNotNaN is your current name for "destination is valid"

        // => true if either destX or destY is NaN => "destination == NULL"

        uint32x4_t reachedOrNoDest = vorrq_u32(reachedDest, mask_invalid_dest);

        // Mask for reached destination OR no current destination
        //uint32x4_t reachedOrNoDest = vorrq_u32(reachedDest, mask_valid_dest);

        // Mask for waypoints NOT being empty (assuming you have a boolean array waypointsEmpty)
        uint32x4_t waypointsNotEmptyMask = vmvnq_u32(waypointsEmptyMask); // Invert empty mask

        // Final mask: If (agent reached destination OR no current destination) AND waypoints are NOT empty
        uint32x4_t finalMask = vandq_u32(reachedOrNoDest, waypointsNotEmptyMask);

        uint32_t finalMaskArr[4];
        vst1q_u32(finalMaskArr, finalMask);

        for (int j = i; j < i + 4; j++)
        {
            // If the agent has reached its destination or has no current destination
            // and the waypoints are not empty
            if (finalMaskArr[j-i] == 0xFFFFFFFF)
            {
                // get next destination if available
                waypointsX[j].push_back(destX[j]);
                waypointsY[j].push_back(destY[j]);
                waypointsR[j].push_back(destR[j]);
                waypointsID[j].push_back(destID[j]);

                outDestX[j] = waypointsX[j].front();
                outDestY[j] = waypointsY[j].front();
                outDestR[j] = waypointsR[j].front();
                outDestID[j] = waypointsID[j].front();

                waypointsX[j].pop_front();
                waypointsY[j].pop_front();
                waypointsR[j].pop_front();
                waypointsID[j].pop_front();
            }
            else
            {
                // Case 2: agent has not yet reached destination, continue to move towards
                // current destination
                outDestX[j] = destX[j];
                outDestY[j] = destY[j];
                outDestR[j] = destR[j];
                outDestID[j] = destID[j];
            }
        }


    }

        // handle the rest of the agents
        for (; i < numAgents; i++)
        {
        }

}
*/
