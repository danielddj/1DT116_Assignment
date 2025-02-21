#include "ped_region.h"

Ped::Region::Region() {

}

// Constructor
Ped::Region::Region(int x1, int x2, int y1, int y2) {
    // Set boundaries of the region
    x_start = x1;
    x_end = x2;
    y_start = y1;
    y_end = y2;

    // Initialize the occupancy grid
    for (int row = 0; row < 160; ++row) {
        for (int col = 0; col < 120; ++col) {
            // Set each position in the grid to 0 (unoccupied)
            occupancy_grid[row][col].store(0);
        }
    }

    // Debugging
    printf("Region created: x[%d, %d], y[%d, %d]\n", x_start, x_end, y_start, y_end);
}


// Add agent to region
void Ped::Region::addAgent(Ped::Tagent *agent) {
    agents.push_back(agent);
}

// Remove agent from region
void Ped::Region::removeAgent(Ped::Tagent *agent) {
    agents.erase(std::remove(agents.begin(), agents.end(), agent), agents.end());
}

// Remove all agents from region
void Ped::Region::clearAgents() {
    agents.clear();
}

// Return if spot is taken by another agent
bool Ped::Region::isOccupied(int x, int y) {
    return occupancy_grid[x][y].load() != 0;
}

// Try to move to desired position inside region
bool Ped::Region::attemptMove(int old_x, int old_y, int new_x, int new_y, int agent_id) {
    int empty = 0; // gives error if it's not a variable
    
    if (occupancy_grid[new_x][new_y].compare_exchange_strong(empty, agent_id)) {
        occupancy_grid[old_x][old_y] = 0; // Empty old position, now its free
        return true; // Attempt succesfful
    }

    return false; // Attempt unsuccessful
}


