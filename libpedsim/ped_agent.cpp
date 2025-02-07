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

void Ped::Tagent::getNextDestinationSIMD(const float *x, const float *y,
                                         const float *destX, const float *destY,
                                         const float *destR, const int *destID,
                                         float *outDestX, float *outDestY,
                                         float *outDestR, int *outDestID,
                                         int numAgents,
                                         std::vector<std::deque<float>> &waypointsX,
                                         std::vector<std::deque<float>> &waypointsY,
                                         std::vector<std::deque<float>> &waypointsR,
                                         std::vector<std::deque<int>> &waypointsID)
{
    int i = 0;

    // Process in batches of 4
    for (; i <= numAgents - 4; i += 4)
    {
        // 1) Load positions/destinations for these 4 agents
        float32x4_t xVec = vld1q_f32(&x[i]);
        float32x4_t yVec = vld1q_f32(&y[i]);
        float32x4_t destXVec = vld1q_f32(&destX[i]);
        float32x4_t destYVec = vld1q_f32(&destY[i]);
        float32x4_t destRVec = vld1q_f32(&destR[i]);
        int32x4_t destIDVec = vld1q_s32(&destID[i]);

        // 2) Check for “destination == NULL” by testing if (destX/destY) is NaN
        //    mask_valid_dest = ~isNaN
        uint32x4_t isNaN_X = vmvnq_u32(vceqq_f32(destXVec, destXVec));
        uint32x4_t isNaN_Y = vmvnq_u32(vceqq_f32(destYVec, destYVec));
        uint32x4_t mask_invalid = vorrq_u32(isNaN_X, isNaN_Y); // 1 if dest is NULL
        uint32x4_t mask_valid_dest = vmvnq_u32(mask_invalid);  // 1 if dest is not NULL

        // 3) Compute distance for valid destinations
        float32x4_t maskedDestX = vreinterpretq_f32_u32(
            vandq_u32(vreinterpretq_u32_f32(destXVec), mask_valid_dest));
        float32x4_t maskedDestY = vreinterpretq_f32_u32(
            vandq_u32(vreinterpretq_u32_f32(destYVec), mask_valid_dest));

        float32x4_t dx = vsubq_f32(maskedDestX, xVec);
        float32x4_t dy = vsubq_f32(maskedDestY, yVec);
        float32x4_t dx2 = vmulq_f32(dx, dx);
        float32x4_t dy2 = vmulq_f32(dy, dy);
        float32x4_t dist = vsqrtq_f32(vaddq_f32(dx2, dy2));

        // 4) Check if reachedDest: distance < destR (only for valid destinations)
        uint32x4_t reachedMask = vcltq_f32(dist, destRVec); // 1 if distance < destR
        reachedMask = vandq_u32(reachedMask, mask_valid_dest);

        // 5) Combine: “(reachedDest || nullDest) && !waypoints.empty()”
        //    => (reachedMask OR mask_invalid) AND (waypoints not empty)
        uint32x4_t reachedOrNull = vorrq_u32(reachedMask, mask_invalid);

        // Build a mask for empty vs non‐empty waypoints
        // 0xFFFFFFFF = empty, 0 = not empty. We will invert it below.
        uint32_t empties[4] = {
            waypointsX[i + 0].empty() ? 0xFFFFFFFF : 0,
            waypointsX[i + 1].empty() ? 0xFFFFFFFF : 0,
            waypointsX[i + 2].empty() ? 0xFFFFFFFF : 0,
            waypointsX[i + 3].empty() ? 0xFFFFFFFF : 0};
        uint32x4_t emptiesVec = vld1q_u32(empties);
        uint32x4_t waypointsNotEmpty = vmvnq_u32(emptiesVec); // 1 if NOT empty

        // Final mask: must satisfy both conditions
        uint32x4_t finalMask = vandq_u32(reachedOrNull, waypointsNotEmpty);

        // 6) Extract the mask bits to an array so we can do per-lane logic in C++
        uint32_t finalMaskArr[4];
        vst1q_u32(finalMaskArr, finalMask);

        // 7) For each lane/agent in this 4-wide chunk:
        for (int lane = 0; lane < 4; lane++)
        {
            int idx = i + lane; // the agent index

            // If bitwise all 1s => we do “Case 1”
            //   => push_back(currentDest), newDest=front(), pop_front()
            // Else => keep current destination
            if (finalMaskArr[lane] == 0xFFFFFFFF)
            {
                // Push the old dest to the back:
                waypointsX[idx].push_back(destX[idx]);
                waypointsY[idx].push_back(destY[idx]);
                waypointsR[idx].push_back(destR[idx]);
                waypointsID[idx].push_back(destID[idx]);

                // Next destination is the front of the queue
                outDestX[idx] = waypointsX[idx].front();
                outDestY[idx] = waypointsY[idx].front();
                outDestR[idx] = waypointsR[idx].front();
                outDestID[idx] = waypointsID[idx].front();

                // Pop front
                waypointsX[idx].pop_front();
                waypointsY[idx].pop_front();
                waypointsR[idx].pop_front();
                waypointsID[idx].pop_front();
            }
            else
            {
                // “Case 2”: keep the current destination
                outDestX[idx] = destX[idx];
                outDestY[idx] = destY[idx];
                outDestR[idx] = destR[idx];
                outDestID[idx] = destID[idx];
            }
        }
    }

    // -----------------------------------------------------------------
    // SCALAR FALLBACK for leftover agents (numAgents not multiple of 4)
    // -----------------------------------------------------------------
    for (; i < numAgents; i++)
    {
        // 1) “destination == NULL” <=> (destX[i], destY[i]) = (NaN, NaN)
        bool hasNoDestination = (std::isnan(destX[i]) || std::isnan(destY[i]));

        // 2) Check “agentReachedDestination” if we have a valid destination
        bool reachedDestination = false;
        if (!hasNoDestination)
        {
            float dx = destX[i] - x[i];
            float dy = destY[i] - y[i];
            float dist = std::sqrt(dx * dx + dy * dy);
            reachedDestination = (dist < destR[i]);
        }

        // 3) If ((reached || noDest) && !waypoints.empty()):
        if ((reachedDestination || hasNoDestination) && !waypointsX[i].empty())
        {
            // push_back
            waypointsX[i].push_back(destX[i]);
            waypointsY[i].push_back(destY[i]);
            waypointsR[i].push_back(destR[i]);
            waypointsID[i].push_back(destID[i]);

            // nextDestination = front()
            outDestX[i] = waypointsX[i].front();
            outDestY[i] = waypointsY[i].front();
            outDestR[i] = waypointsR[i].front();
            outDestID[i] = waypointsID[i].front();

            // pop_front
            waypointsX[i].pop_front();
            waypointsY[i].pop_front();
            waypointsR[i].pop_front();
            waypointsID[i].pop_front();
        }
        else
        {
            // keep the old destination
            outDestX[i] = destX[i];
            outDestY[i] = destY[i];
            outDestR[i] = destR[i];
            outDestID[i] = destID[i];
        }
    }
}

// Vectorized version (processing 4 agents per loop)
void Ped::Tagent::computeDesiredPositionsSIMD(const float *x, const float *y,
                                              const float *destX, const float *destY,
                                              float *outDesiredX, float *outDesiredY,
                                              int numAgents)
{
    int i = 0;
    // Process in batches of 4 floats (128 bits)
    for (; i <= numAgents - 4; i += 4)
    {
        // We load the current positions and destination positions for 4 agents
        float32x4_t posX = vld1q_f32(&x[i]);
        float32x4_t posY = vld1q_f32(&y[i]);
        float32x4_t dX = vld1q_f32(&destX[i]);
        float32x4_t dY = vld1q_f32(&destY[i]);

        // is desired NAN
        uint32x4_t valid = vceqq_f32(dX, dX);

        // Compute differences: diff = destination - current position
        float32x4_t diffX = vsubq_f32(dX, posX);
        float32x4_t diffY = vsubq_f32(dY, posY);

        // Compute squared differences and sum them: sumSq = diffX^2 + diffY^2
        float32x4_t diffX2 = vmulq_f32(diffX, diffX);
        float32x4_t diffY2 = vmulq_f32(diffY, diffY);
        float32x4_t sumSq = vaddq_f32(diffX2, diffY2);

        // Compute the length (distance) via square root
        float32x4_t len = vsqrtq_f32(sumSq);

        float32x4_t recip = vrecpeq_f32(sumSq);

        /*
        // Compute the reciprocal of len using NEON reciprocal approx
        float32x4_t recip = vrecpeq_f32(len);
        // Refine the reciprocal estimate for better accuracy (2 iterations)
        recip = vmulq_f32(vrecpsq_f32(len, recip), recip);
        recip = vmulq_f32(vrecpsq_f32(len, recip), recip);
        */

        // Compute ratio = diff / len
        float32x4_t ratioX = vmulq_f32(diffX, recip);
        float32x4_t ratioY = vmulq_f32(diffY, recip);

        // Compute new desired positions: newPos = current position + ratio
        float32x4_t newX = vaddq_f32(posX, ratioX);
        float32x4_t newY = vaddq_f32(posY, ratioY);

        // Round the results. ARMv8 provides vrndnq_f32 (round to nearest, ties to even).
        float32x4_t newXRounded = vrndnq_f32(newX);
        float32x4_t newYRounded = vrndnq_f32(newY);

        // update positions only if the destination is valid

        float tmpX[4], tmpY[4];
        vst1q_f32(tmpX, newXRounded);
        vst1q_f32(tmpY, newYRounded);

        // check if destination is valid (not NAN)
        uint32_t validMaskArr[4];
        vst1q_u32(validMaskArr, valid);

        for (int lane = 0; lane < 4; lane++)
        {
            if (validMaskArr[lane])
            {
                outDesiredX[i + lane] = tmpX[lane];
                outDesiredY[i + lane] = tmpY[lane];
            }
            else
            {
                outDesiredX[i + lane] = x[i + lane];
                outDesiredY[i + lane] = y[i + lane];
            }
        }

        // vst1q_f32(&outDesiredX[i], newXRounded);
        // vst1q_f32(&outDesiredY[i], newYRounded);
    }

    /*
    // Store the results to temporary arrays so that we can check the valid flags
    float newXArr[4], newYArr[4];
    vst1q_f32(newXArr, newXRounded);
    vst1q_f32(newYArr, newYRounded);

    // For each of the 4 lanes, check if the result is valid.
    // If valid, write the computed value; otherwise, leave the original value.
    for (int j = 0; j < 4; ++j)
    {
        if (valid[i + j])
        {
            outDesiredX[i + j] = newXArr[j];
            outDesiredY[i + j] = newYArr[j];
        }
        else
        {
            outDesiredX[i + j] = x[i + j];
            outDesiredY[i + j] = y[i + j];
        }
    }
}
*/

    // Process any remaining agents with a scalar fallback.
    // The number of agents might not be a multiple of 4.
    for (; i < numAgents; i++)
    {
        float diffX = destX[i] - x[i];
        float diffY = destY[i] - y[i];
        float len = sqrtf(diffX * diffX + diffY * diffY);
        outDesiredX[i] = roundf(x[i] + diffX / len);
        outDesiredY[i] = roundf(y[i] + diffY / len);
    }
}
/*
Original scalar logic (per agent):
----------------------------------
if (destination == NULL) {
    // no movement
    desiredX = x;
    desiredY = y;
} else {
    double diffX = destX - x;
    double diffY = destY - y;
    double len   = sqrt(diffX*diffX + diffY*diffY);
    if (len > 0) {
        desiredX = round(x + diffX / len);
        desiredY = round(y + diffY / len);
    } else {
        // length==0 => no movement
        desiredX = x;
        desiredY = y;
    }
}
*/

void Ped::Tagent::computeDesiredPositionsSIMD2(const float *x, const float *y,
                                               const float *destX, const float *destY,
                                               float *outDesiredX, float *outDesiredY,
                                               int numAgents)
{
    int i = 0;

    // We process agents in batches of 4
    for (; i <= numAgents - 4; i += 4)
    {
        // Current position and destination
        float32x4_t posX = vld1q_f32(&x[i]);
        float32x4_t posY = vld1q_f32(&y[i]);
        float32x4_t dX = vld1q_f32(&destX[i]);
        float32x4_t dY = vld1q_f32(&destY[i]);

        // Identify invalid destinations (NaN) meaning "destination == NULL"
        // isNaN = 1 if NaN, else 0
        uint32x4_t isNaN_X = vmvnq_u32(vceqq_f32(dX, dX));
        uint32x4_t isNaN_Y = vmvnq_u32(vceqq_f32(dY, dY));
        // invalidDestMask = 1 if either destX or destY is NaN
        uint32x4_t invalidDestMask = vorrq_u32(isNaN_X, isNaN_Y);

        float32x4_t diffX = vsubq_f32(dX, posX);
        float32x4_t diffY = vsubq_f32(dY, posY);

        // squared differences
        float32x4_t diffX2 = vmulq_f32(diffX, diffX);
        float32x4_t diffY2 = vmulq_f32(diffY, diffY);
        float32x4_t dist2 = vaddq_f32(diffX2, diffY2);

        float32x4_t dist = vsqrtq_f32(dist2);

        // We want to do ratioX = diffX / dist, ratioY = diffY / dist
        // but only for lanes where destination is valid (not NaN) and dist > 0.
        // We'll zero them out for invalid or zero-length cases.

        // dist < epsilon => treat as zero movement (idk if this is necessary)
        float32x4_t eps = vdupq_n_f32(1e-6f);
        uint32x4_t nearZeroMask = vcltq_f32(dist2, eps); // 1 if dist^2 < 1e-6 => dist ~ 0

        // validDestMask: 1 if not NaN
        uint32x4_t validDestMask = vmvnq_u32(invalidDestMask);
        // combine masks: we only want to move if "validDestMask" and "not nearZero"
        uint32x4_t canMoveMask = vandq_u32(validDestMask, vmvnq_u32(nearZeroMask));

        // ratio = diff / dist
        float32x4_t ratioX = vdivq_f32(diffX, dist);
        float32x4_t ratioY = vdivq_f32(diffY, dist);

        // Zero out ratio for invalid or zero-dist lanes
        ratioX = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ratioX), canMoveMask));
        ratioY = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(ratioY), canMoveMask));

        // newX = posX + ratioX, newY = posY + ratioY
        float32x4_t newX = vaddq_f32(posX, ratioX);
        float32x4_t newY = vaddq_f32(posY, ratioY);

        // Rounding
        // armv8 neon has vrndnq_f32 (round to nearest) or vcvtnq_f32.
        // We'll use vrndnq_f32
        float32x4_t roundedX = vrndnq_f32(newX);
        float32x4_t roundedY = vrndnq_f32(newY);

        // We have two final possibilities for each lane:
        //    - If the destination was invalid or length ~ 0, we keep the old position (posX, posY).
        //    - Otherwise, we use the new (roundedX, roundedY).

        // We'll do per-lane merging in C++ by extracting the results to temporary arrays.
        float storeX[4], storeY[4];
        vst1q_f32(storeX, roundedX);
        vst1q_f32(storeY, roundedY);

        float originalX[4], originalY[4];
        vst1q_f32(originalX, posX);
        vst1q_f32(originalY, posY);

        // Extract mask bits
        uint32_t canMoveMaskArr[4];
        vst1q_u32(canMoveMaskArr, canMoveMask);

        // 10) Write back per-lane
        for (int lane = 0; lane < 4; lane++)
        {
            int idx = i + lane;
            if (canMoveMaskArr[lane] == 0xFFFFFFFF)
            {
                // This lane has a valid destination & dist > 0 => move
                outDesiredX[idx] = storeX[lane];
                outDesiredY[idx] = storeY[lane];
            }
            else
            {
                // Either dest == NULL (NaN) or dist ~ 0 => no move
                outDesiredX[idx] = originalX[lane];
                outDesiredY[idx] = originalY[lane];
            }
        }
    }

    // -----------------------------------------------------------
    // SCALAR FALLBACK FOR REMAINING AGENTS (numAgents % 4 != 0)
    // -----------------------------------------------------------
    for (; i < numAgents; i++)
    {
        bool hasNoDestination = (std::isnan(destX[i]) || std::isnan(destY[i]));
        if (hasNoDestination)
        {
            // No movement, keep old position
            outDesiredX[i] = x[i];
            outDesiredY[i] = y[i];
        }
        else
        {
            float diffX = destX[i] - x[i];
            float diffY = destY[i] - y[i];
            float dist = std::sqrt(diffX * diffX + diffY * diffY);

            if (dist > 1e-6f)
            {
                float nx = x[i] + diffX / dist;
                float ny = y[i] + diffY / dist;
                outDesiredX[i] = std::round(nx);
                outDesiredY[i] = std::round(ny);
            }
            else
            {
                // dist is zero or extremely small => no movement
                outDesiredX[i] = x[i];
                outDesiredY[i] = y[i];
            }
        }
    }
}
