#ifndef _ped_region_h
#define _ped_region_h

#include "ped_agent.h"
#include <vector>
#include <algorithm>
#include <atomic>
#include <iostream>

namespace Ped {
    class Region {
    private:
        // Region boundaries
        int x_start, x_end, y_start, y_end;  
        
        // Agents in the region
        std::vector<Ped::Tagent *> agents;

        // Occupancy grid (no locks)
        std::atomic<int> occupancy_grid[160][120];

    public:
        // Basic constructor, empty
        Region();

        // Constructor
        Region(int x1, int x2, int y1, int y2);

        // Getters
        int getXStart() const { return x_start; }
        int getXEnd() const { return x_end; }
        int getYStart() const { return y_start; }
        int getYEnd() const { return y_end; }

        // List of agents in region
        std::vector<Ped::Tagent *> &getAgents() { return agents; } 

        // Add one agent to region
        void addAgent(Ped::Tagent *agent);
        // Remove one agent from region
        void removeAgent(Ped::Tagent *agent);
        // Remove all agents from region
        void clearAgents();

        // Try to move inside occupancy grid
        bool attemptMove(int old_x, int old_y, int new_x, int new_y, int agent_id);
        // Return if spot is occupied by another agent
        bool isOccupied(int x, int y);
    };
}

#endif // _ped_region_h
