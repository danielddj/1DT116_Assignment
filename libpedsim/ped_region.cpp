#include "ped_region.h"
#include "ped_agent.h"
#include "ped_model.h"
#include "ped_regionhandler.h" // Contains the definition of Region_handler

namespace Ped {

// AgentNode constructor.
AgentNode::AgentNode(Tagent *a) : agent(a), next(nullptr) {}

// Region constructor.
Region::Region(int x_min, int x_max, int y_min, int y_max)
    : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max),
      agent_list_head(nullptr), agentCount(0) {}

// Check if an agent's current position is within region boundaries.
bool Region::contains(Tagent *agent) const {
  int x = agent->X->at(agent->getId());
  int y = agent->Y->at(agent->getId());
  return (x >= xMin && x < xMax) && (y >= yMin && y < yMax);
}

// Check if an agent's desired position is within region boundaries.
bool Region::contains_desired(Tagent *agent) const {
  int x = agent->desiredX->at(agent->getId());
  int y = agent->desiredY->at(agent->getId());
  return (x >= xMin && x < xMax) && (y >= yMin && y < yMax);
}

// Add an agent to the head of the linked list using atomic operations on
// shared_ptr.
void Region::addAgent(Tagent *agent) {
  auto new_node = std::make_shared<AgentNode>(agent);
  // Atomically load the current head.
  std::shared_ptr<AgentNode> old_head =
      std::atomic_load_explicit(&agent_list_head, std::memory_order_relaxed);
  do {
    new_node->next = old_head; // Set new_node->next to the current head.
    // Attempt to atomically replace the head with new_node.
  } while (!std::atomic_compare_exchange_weak_explicit(
      &agent_list_head, &old_head, new_node, std::memory_order_release,
      std::memory_order_relaxed));
  agentCount.fetch_add(1, std::memory_order_relaxed);
}

// Remove an agent from the list using atomic operations on shared_ptr.
bool Region::removeAgent(Tagent *agent) {
  std::shared_ptr<AgentNode> prev = nullptr;
  auto curr =
      std::atomic_load_explicit(&agent_list_head, std::memory_order_acquire);
  while (curr) {
    if (curr->agent == agent) {
      // Atomically load the next pointer.
      std::shared_ptr<AgentNode> next =
          std::atomic_load_explicit(&curr->next, std::memory_order_acquire);
      if (prev) {
        // Try to update prev->next to bypass curr.
        if (!std::atomic_compare_exchange_strong_explicit(
                &prev->next, &curr, next, std::memory_order_release,
                std::memory_order_acquire)) {
          // The list changed concurrently; restart removal.
          return removeAgent(agent);
        }
      } else {
        // Remove from the head.
        if (!std::atomic_compare_exchange_strong_explicit(
                &agent_list_head, &curr, next, std::memory_order_release,
                std::memory_order_acquire)) {
          return removeAgent(agent);
        }
      }
      agentCount.fetch_sub(1, std::memory_order_relaxed);
      // The removed node will be deallocated automatically when no references
      // remain.
      return true;
    }
    prev = curr;
    curr = std::atomic_load_explicit(&curr->next, std::memory_order_acquire);
  }
  return false;
}

// Transfer an agent from this region to another region via the Region_handler.
void Region::transfer_agents(Region_handler *handler, Tagent *agent) {
  if (!removeAgent(agent)) {
    std::cerr << "[ERROR] Failed to remove agent from current region."
              << std::endl;
    return;
  }
  Region *target = handler->next_region_for_agent(agent);
  if (!target) {
    std::cerr << "[ERROR] No target region found for agent." << std::endl;
    return;
  }
  target->transfer_to(agent);
}

// Transfer an agent into this region.
void Region::transfer_to(Tagent *agent) {
  addAgent(agent);
  pending_transfers.push_back(agent);
}

// Process movement of agents in this region.
void Region::move_agents(Ped::Model *model, Region_handler *handler) {
  auto curr =
      std::atomic_load_explicit(&agent_list_head, std::memory_order_acquire);
  while (curr) {
    Tagent *agent = curr->agent;
    if (!agent) {
      std::cerr << "[ERROR] Agent pointer is null at node " << curr.get()
                << std::endl;
      break;
    }
    agent->computeNextDesiredPosition();
    if (contains_desired(agent)) {
      model->move(agent);
    } else {
      transfer_agents(handler, agent);
    }
    curr = std::atomic_load_explicit(&curr->next, std::memory_order_acquire);
  }
}

} // namespace Ped
