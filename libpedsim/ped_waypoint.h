//
// Adapted for Low Level Parallel Programming 2017
//
// Twaypoint represents a destination in a scenario.
// It is described by its position (x,y) and a radius
// which is used to define how close an agent has to be
// to this destination until we consider that the agent
// has reached its destination.int Ped::Twaypoint::staticid = 0;//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
#ifndef _ped_waypoint_h_
#define _ped_waypoint_h_ 1

#ifdef WIN32
#define LIBEXPORT __declspec(dllexport)
#else
#define LIBEXPORT
#endif

#include <cstddef>
#include <nmmintrin.h>

using namespace std;

namespace Ped
{
	struct waypointSoA
	{
		int numberOfWaypoints;
		float *x;
		float *y;
		float *r;
		int staticid;

		waypointSoA(int numWaypoints) : staticid(0)
		{
			numberOfWaypoints = numWaypoints;
			x = (float *)_mm_malloc(numberOfWaypoints * sizeof(float), 16);
			y = (float *)_mm_malloc(numberOfWaypoints * sizeof(float), 16);
			r = (float *)_mm_malloc(numberOfWaypoints * sizeof(float), 16);
		}

		~waypointSoA()
		{
			_mm_free(x);
			_mm_free(y);
			_mm_free(r);
		}
	};

	class LIBEXPORT Twaypoint
	{
	public:
		Twaypoint(waypointSoA *soa);
		Twaypoint(double x, double y, double r, waypointSoA *soa);
		virtual ~Twaypoint();

		// Sets the coordinates and the radius of this waypoint
		void setx(double px) { soa->x[id] = px; };
		void sety(double py) { soa->y[id] = py; };
		void setr(double pr) { soa->r[id] = pr; };

		int getid() const { return id; };
		double getx() const { return soa->x[id]; };
		double gety() const { return soa->y[id]; };
		double getr() const { return soa->r[id]; };

	protected:
		// id incrementer, used for assigning unique ids
		waypointSoA *soa;

		// waypoint id
		int id;
	};
}

#endif
