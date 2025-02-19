#include "ped_region.h"
#include <vector>

namespace Ped {
class Region_handler {
public:
  Region_handler(size_t start_regions, bool resize, size_t max_x, size_t max_y,
                 size_t max_agents, size_t min_agents);
  ~Region_handler();

  std::vector<Region *> regions;

protected:
  Ped::Region *get_region_for_agent();

private:
  size_t max_x, max_y, max_agents, min_agents;
  bool dynamic_resize;
  void resize_regions();
  void valid_region_count(size_t start_regions);
  void add_region(size_t x, size_t y, size_t width, size_t height);
  void tick_regions(Ped::Model *model);
};

} // namespace Ped