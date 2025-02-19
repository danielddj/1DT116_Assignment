#ifndef _ped_regionhandler_h_
#define _ped_regionhandler_h_

#include "ped_agent.h"
#include "ped_region.h"
#include <vector>

namespace Ped {
class Region;
class Region_handler {

public:
  Region_handler(size_t start_regions, bool resize, size_t max_x, size_t max_y,
                 size_t max_agents, size_t min_agents);
  ~Region_handler();

  std::vector<Ped::Region *> regions;

  Ped::Region *next_region_for_agent(Ped::Tagent *agent);

protected:
private:
  size_t max_x, max_y, max_agents, min_agents;
  bool dynamic_resize;
  void resize_regions();
  void valid_region_count(size_t start_regions);
  void add_region(size_t x, size_t y, size_t width, size_t height);
  void tick_regions(Ped::Model *model);
};

} // namespace Ped

#endif