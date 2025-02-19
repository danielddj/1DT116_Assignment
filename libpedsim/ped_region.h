#include "ped_agent.h"
#include "ped_model.h"
#include <atomic>
#include <functional>
#include <vector>

namespace Ped {
class Region {
public:
  Region(int x_min, int x_max, int y_min, int y_max)
      : xMin(x_min), xMax(x_max), yMin(y_min), yMax(y_max),
        agent_list_head(nullptr), agentCount(0) {}

  void move_agents(Ped::Model *model);

protected:
  Region(std::vector<Ped::Tagent> agentVector);
  ~Region();

  bool is_in_region(Ped::Tagent agent);

private:
  struct AgentNode {
    std::atomic<Ped::Tagent *> agent;
    std::atomic<AgentNode *> next;
  };

  std::vector<Ped::Tagent> agentsInRegion;

  int xMin, xMax, yMin, yMax;

  std::atomic<AgentNode *> agent_list_head;
  std::atomic<int> agentCount;
  bool removeAgent(Ped::Tagent *agent);
  void addAgent(Ped::Tagent *agent);
  bool contains(Ped::Tagent *agent) const;
  bool check_succesful_add(AgentNode *new_head, AgentNode *old_head);
  bool contains_desired(Ped::Tagent *agent) const;

  typedef std::function<Region *(Tagent *)> TargetRegionFunc;
  void transfer_agents(TargetRegionFunc targetForAgent);
  int startRegionX;
  int endRegionX;
  int startRegionY;
  int endRegionY;
};

} // Namespace Ped