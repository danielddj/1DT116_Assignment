//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_agent.h"
#include "ped_model.h"
#include "ped_waypoint.h"
#include <iostream>
#include <math.h>

#include <stdlib.h>

Ped::Tagent::Tagent(int posX, int posY) { Ped::Tagent::init(posX, posY); }

Ped::Tagent::Tagent(double posX, double posY) {
  Ped::Tagent::init((int)round(posX), (int)round(posY));
}

void Ped::Tagent::initialize(int i, std::vector<float> *X,
                             std::vector<float> *Y,
                             std::vector<float> *desiredX,
                             std::vector<float> *desiredY,
                             std::vector<float> *destinationX,
                             std::vector<float> *destinationY,
                             std::vector<float> *destinationR) {
  this->id = i;
  this->X = X;
  this->Y = Y;
  this->desiredX = desiredX;
  this->desiredY = desiredY;
  this->destinationX = destinationX;
  this->destinationY = destinationY;
  this->destinationR = destinationR;
}

void Ped::Tagent::init(int posX, int posY) {
  x = posX;
  y = posY;
  destination = NULL;
  lastDestination = NULL;
}

void Ped::Tagent::callNextDestination() { destination = getNextDestination(); }

void Ped::Tagent::computeNextDesiredPosition() {
  if (id == 382) {
    std::cout << "tes" << std::endl;
  }
  destination = getNextDestination();
  if (destination == NULL) {
    // No destination, so nothing to compute.
    return;
  }

  double diffX = destination->getx() - (*X)[id];
  double diffY = destination->gety() - (*Y)[id];
  double len = sqrt(diffX * diffX + diffY * diffY);

  // Check to avoid division by zero.
  if (len == 0) {
    // We're exactly at the destination, so desired position remains the same.
    (*desiredX)[id] = (*X)[id];
    (*desiredY)[id] = (*Y)[id];
    desiredPositionX = (*X)[id];
    desiredPositionY = (*Y)[id];
    return;
  }

  // Compute the desired position.
  float newDesiredX = (float)round((*X)[id] + diffX / len);
  float newDesiredY = (float)round((*Y)[id] + diffY / len);

  (*desiredX)[id] = newDesiredX;
  (*desiredY)[id] = newDesiredY;
  desiredPositionX = newDesiredX;
  desiredPositionY = newDesiredY;
}

void Ped::Tagent::addWaypoint(Twaypoint *wp) { waypoints.push_back(wp); }

Ped::Twaypoint *Ped::Tagent::getNextDestination() {
  Ped::Twaypoint *nextDestination = NULL;
  bool agentReachedDestination = false;

  if (destination != NULL) {
    // compute if agent reached its current destination
    double diffX = (*destinationX)[id] - (*X)[id];
    double diffY = (*destinationY)[id] - (*Y)[id];
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
    (*destinationX)[id] = nextDestination->getx();
    (*destinationY)[id] = nextDestination->gety();
    (*destinationR)[id] = nextDestination->getr();
  } else {
    // Arrived
    (*destinationX)[id] = (*X)[id];
    (*destinationY)[id] = (*Y)[id];
  }
  return nextDestination;
}
