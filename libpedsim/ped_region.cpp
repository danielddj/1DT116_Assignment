#include "ped_region.h"
#include "ped_agent.h"
#include "ped_model.h"
#include <vector>

namespace Ped {

// Check if a point (x,y) is in this region.
bool Ped::Region::contains(Ped::Tagent *agent) const {
  int x = agent->getX();
  int y = agent->getY();

  return (x >= xMin && x < xMax) && (y >= yMin && y < yMax);
}

bool Ped::Region::contains_desired(Ped::Tagent *agent) const {
  int x = agent->getDesiredX();
  int y = agent->getDesiredY();

  return (x >= xMin && x < xMax) && (y >= yMin && y < yMax);
}

// Insert an agent (lock-free via CAS).
void Ped::Region::addAgent(Ped::Tagent *agent) {

  AgentNode *new_node = new AgentNode{agent, nullptr};

  AgentNode *old_head = agent_list_head.load(std::memory_order_relaxed);

  do {
    new_node->next.store(old_head, std::memory_order_relaxed);

  } while (!check_succesful_add(new_node, old_head));

  agentCount.fetch_add(1, std::memory_order_relaxed);
}

bool Ped::Region::check_succesful_add(AgentNode *new_head,
                                      AgentNode *old_head) {
  return agent_list_head.compare_exchange_weak(
      old_head, new_head, std::memory_order_release, std::memory_order_relaxed);
}

// Remove an agent (using CAS) – returns true if successful.
bool Ped::Region::removeAgent(Ped::Tagent *agent) {
  AgentNode *prev = nullptr;
  AgentNode *curr = agent_list_head.load(std::memory_order_acquire);

  while (curr) {
    if (curr->agent == agent) {
      AgentNode *next = curr->next.load(std::memory_order_acquire);

      if (prev) {
        if (!prev->next.compare_exchange_strong(curr, next,
                                                std::memory_order_release)) {
          // List changed, restart removal.
          return removeAgent(agent);
        }
      } else {
        if (!agent_list_head.compare_exchange_strong(
                curr, next, std::memory_order_release)) {
          return removeAgent(agent);
        }
      }
      agentCount.fetch_sub(1, std::memory_order_relaxed);

      delete curr;
      return true;
    }
    prev = curr;
    curr = curr->next.load(std::memory_order_acquire);
  }
  return false;
}

void Ped::Region::transfer_agents(TargetRegionFunc targetForAgent) {
  AgentNode *curr =
      agent_list_head.exchange(nullptr, std::memory_order_acquire);
  agentCount.store(0, std::memory_order_relaxed);

  while (curr) {
    AgentNode *next = curr->next.load(std::memory_order_acquire);

    Region *target = targetForAgent(curr->agent);
    if (target) {
      target->addAgent(curr->agent);
    }

    delete curr;
    curr = next;
  }
}

void Ped::Region::move_agents(Ped::Model *model) {
  AgentNode *curr = agent_list_head;
  while (curr) {
    curr->agent.load()->computeNextDesiredPosition();

    if (contains_desired(curr->agent.load())) {
      model->move(curr->agent);
      curr = curr->next;
    } else {
    }
  }
}

} // namespace Ped