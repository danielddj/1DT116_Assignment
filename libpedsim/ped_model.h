//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <deque>
#include <assert.h>
#include "AlignedAllocator.h"

#include <arm_neon.h>

#include "ped_agent.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ ,THREADS};

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// tick for sequential implementation
		void sequential_tick();

		// tick for openmp implementation
		void openmp_tick1();
		void openmp_tick2();

		// tick for c++ thread implementation
		void threads_tick();

		// tick for vector implementation (SIMD)
		void vector_tick();


		// Returns the agents of this scenario
		const std::vector<Tagent*>& getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		static int numberOfThreads;

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);


		// SoA (Structure of Arrays) layout for the SIMD implementation

		// Agent's positions
		/*
		std::vector<float> x;
		std::vector<float> y;

		// Agent's desired positions
		std::vector<float> desiredPositionX;
		std::vector<float> desiredPositionY;

		std::vector<float> destinationsX;
		std::vector<float> destinationsY;
		std::vector<float> destinationsR;
		std::vector<int> destinationsID;

		std::vector<float> lastDestX;
		std::vector<float> lastDestY;
		std::vector<float> lastDestR;
		std::vector<int> lastDestID;
		*/


		FloatVectorAligned32 x;
		FloatVectorAligned32 y;

		FloatVectorAligned32 desiredPositionX;
		FloatVectorAligned32 desiredPositionY;

		FloatVectorAligned32 destinationsX;
		FloatVectorAligned32 destinationsY;
		FloatVectorAligned32 destinationsR;
		IntVectorAligned32 destinationsID;

		FloatVectorAligned32 lastDestX;
		FloatVectorAligned32 lastDestY;
		FloatVectorAligned32 lastDestR;
		IntVectorAligned32 lastDestID;

		// The queue of all destinations that this agent still has to visit		
		// 		simd SoA layout of deque<Twaypoint *> waypoints;
		std::vector<std::deque<float>> waypointsX;
		std::vector<std::deque<float>> waypointsY;
		std::vector<std::deque<float>> waypointsR;
		std::vector<std::deque<int>> waypointsID;

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		void setupHeatmapSeq();
		void updateHeatmapSeq();
	};
}
#endif
