#ifndef PED_REGION_H
#define PED_REGION_H

#include <vector>
#include <memory>
#include <atomic>
#include <iostream>
#include "ped_agent.h"
#include "ped_model.h"

namespace Ped {

// Forward declaration for Region_handler (assumed defined elsewhere)
class Region_handler;

// Node for the lock-free agent linked list.
// We store the next pointer as a plain std::shared_ptr, but perform atomic
// operations on it using the provided overloads.
struct AgentNode {
    Tagent* agent;
    std::shared_ptr<AgentNode> next; // not wrapped in std::atomic

    AgentNode(Tagent* a);
};

// The Region class supports lock-free insertion, removal, and transfer.
class Region {
public:
    // Region boundaries.
    int xMin, xMax, yMin, yMax;

    // Head pointer for the linked list.
    // We use a plain shared_ptr here and perform atomic operations via atomic_load etc.
    std::shared_ptr<AgentNode> agent_list_head;
    // Count of agents.
    std::atomic<int> agentCount;
    // Agents pending transfer into this region.
    std::vector<Tagent*> pending_transfers;

    // Constructor.
    Region(int x_min = 0, int x_max = 100, int y_min = 0, int y_max = 100);

    // Returns true if the agent's current position is in-region.
    bool contains(Tagent* agent) const;
    // Returns true if the agent's desired position is in-region.
    bool contains_desired(Tagent* agent) const;

    // Add an agent to this region.
    void addAgent(Tagent* agent);
    // Remove an agent from this region; returns true if removed.
    bool removeAgent(Tagent* agent);
    // Transfer an agent from this region to another region via the handler.
    void transfer_agents(Region_handler* handler, Tagent* agent);
    // Transfer an agent into this region.
    void transfer_to(Tagent* agent);
    // Process movement of agents in this region.
    void move_agents(Model* model, Region_handler* handler);
};

} // namespace Ped

#endif // PED_REGION_H
