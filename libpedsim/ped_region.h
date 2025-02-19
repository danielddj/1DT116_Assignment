#ifndef _ped_region_h_
#define _ped_region_h_

#include "ped_agent.h"
#include "ped_model.h"
#include "ped_regionhandler.h"
#include <atomic>
#include <functional>
#include <vector>

namespace Ped {
class Region_handler;
class Region {
public:
  Region(int x_min, int x_max, int y_min, int y_max)
      : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max),
        agent_list_head(nullptr), agentCount(0) {}

  void move_agents(Ped::Model *model, Ped::Region_handler *handler);
  bool contains_desired(Ped::Tagent *agent) const;
  void addAgent(Ped::Tagent *agent);

protected:
  Region(std::vector<Ped::Tagent> agentVector);
  ~Region();
  bool is_in_region(Ped::Tagent agent);

private:
  struct AgentNode {
    std::atomic<Ped::Tagent *> agent;
    std::atomic<AgentNode *> next;
  };

  std::vector<Ped::Tagent *> agents_in_region;
  std::vector<Ped::Tagent *> pending_transfers;

  int xMin, xMax, yMin, yMax;

  std::atomic<AgentNode *> agent_list_head;
  std::atomic<int> agentCount;
  bool removeAgent(Ped::Tagent *agent);
  bool contains(Ped::Tagent *agent) const;
  bool check_succesful_add(AgentNode *new_head, AgentNode *old_head);

  typedef std::function<Region *(Tagent *)> TargetRegionFunc;
  void transfer_agents(Ped::Region_handler *handler);
  void transfer_to(Ped::Tagent *agent);
  int startRegionX;
  int endRegionX;
  int startRegionY;
  int endRegionY;
};

} // Namespace Ped

#endif