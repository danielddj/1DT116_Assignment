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
#include <mutex>
#include "ped_agent.h"

namespace Ped
{

    class Tagent;

    // The implementation modes for Assignment 1 + 2:
    // chooses which implementation to use for tick()
    enum IMPLEMENTATION
    {
        CUDA,
        VECTOR,
        OMP,
        PTHREAD,
        SEQ,
        THREADS,
        COLLISION_AVOIDANCE
    };

    class Model
    {
    public:
        // Sets everything up
        void setup(std::vector<Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation);

        // Coordinates a time step in the scenario: move all agents by one step (if applicable).
        void tick();

        // tick for sequential implementation
        void sequential_tick();

        void computeNextDesiredPosition_SIMD(int i);

        void vector_tick();

        // tick for openmp implementation
        void openmp_tick1();
        void openmp_tick2();

        // tick for c++ thread implementation
        void threads_tick();

        // tick for collision avoidanc
        void collision_avoidance_tick(); // parallelized via a global occupation lock grid 
        void collision_avoidance_tick2(); // parallelized via regions
        void collision_avoidance_tick_seq(); // sequential

            // Returns the agents of this scenario
            const std::vector<Tagent *> &getAgents() const
        {
            return agents;
        };

        // Adds an agent to the tree structure
        void placeAgent(const Ped::Tagent *a);

        // Cleans up the tree and restructures it. Worth calling every now and then.
        void cleanup();
        ~Model();

        // Returns the heatmap visualizing the density of agents
        int const *const *getHeatmap() const { return blurred_heatmap; };
        int getHeatmapSize() const;

        static int numberOfThreads;
        static int avoidanceAlgorithm;
        static bool parralelizeCollisionAvoidance;

    private:
        // Denotes which implementation (sequential, parallel implementations..)
        // should be used for calculating the desired positions of
        // agents (Assignment 1)
        IMPLEMENTATION implementation;

        // The agents in this scenario
        std::vector<Tagent *> agents;

        // The agents in this scenario, divided into regions
        std::vector<Tagent *> regionAgents[4];
        std::mutex regionMutex[4];

        // The waypoints in this scenario
        std::vector<Twaypoint *> destinations;

        // Moves an agent towards its next position
        void move(Ped::Tagent *agent);

        void move_parallelized(Ped::Tagent *agent);
        void move_parallelized2(Ped::Tagent *agent);

        int whichRegion(float x, float y);

        void removeFromRegion(Ped::Tagent *agent, int oldRegion);
        void addToRegion(Ped::Tagent *agent, int newRegion);

        ////////////
        /// Everything below here won't be relevant until Assignment 3
        ///////////////////////////////////////////////

        // Returns the set of neighboring agents for the specified position
        set<const Ped::Tagent *> getNeighbors(int x, int y, int dist) const;
        std::set<const Ped::Tagent *> getNeighborsParallel(int x, int y, int dist);

        ////////////
        /// Everything below here won't be relevant until Assignment 4
        ///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE *CELLSIZE

        // The heatmap representing the density of agents
        int **heatmap;

        // The scaled heatmap that fits to the view
        int **scaled_heatmap;

        // The final heatmap: blurred and scaled to fit the view
        int **blurred_heatmap;

        void setupHeatmapSeq();
        void updateHeatmapSeq();
    };
}
#endif
