#include "ped_regionhandler.h"
#include "ped_region.h"
#include <iostream>
#include <math.h>

namespace Ped {
Region_handler::Region_handler(size_t start_regions, bool resize, size_t max_x,
                               size_t max_y, size_t max_agents,
                               size_t min_agents,
                               std::vector<Ped::Tagent *> agents)
    : dynamic_resize(resize), max_x(max_x), max_y(max_y),
      max_agents(max_agents), min_agents(min_agents) {

  valid_region_count(start_regions);

  size_t grid_dim = static_cast<size_t>(
      std::ceil(std::sqrt(static_cast<double>(start_regions))));

  // Calculate each region's approximate width and height.
  size_t region_width = max_x / grid_dim;
  size_t region_height = max_y / grid_dim;

  size_t added_agents = 0;

  // Create regions for each grid cell.
  for (size_t row = 0; row < grid_dim; ++row) {
    for (size_t col = 0; col < grid_dim; ++col) {
      // Calculate the region's top-left coordinate.
      size_t x = col * region_width;
      size_t y = row * region_height;

      // For the last column/row, adjust the width/height to cover the remaining
      // space.
      size_t width = (col == grid_dim - 1) ? (max_x - x) : region_width;
      size_t height = (row == grid_dim - 1) ? (max_y - y) : region_height;

      Region *added_region = add_region(x, y, width, height);

      for (auto agent : agents) {
        if (added_region->contains_desired(agent)) {
          added_region->addAgent(agent);
          added_agents++;
        }
      }
    }
  }
  if (added_agents != agents.size()) {
    std::runtime_error("ERROR: Some agents were not assigned region!");
  }
}

void Region_handler::valid_region_count(size_t start_regions) {
  if (start_regions < 4) {
    std::cout << "Less than four regions is not allowed, defaulting to 4!"
              << std::endl;

    start_regions = 4;
  }
}

void Region_handler::resize_regions() {}

Region *Region_handler::next_region_for_agent(Tagent *agent) {

  for (auto &region : regions) {
    if (region->contains_desired(agent)) {
      return region;
    }
  }

  return nullptr;
}

void Region_handler::tick_regions(Model *model) {

  std::vector<Ped::Tagent *> agents = model->getAgents();
#pragma omp parallel num_threads(4) shared(agents)
  {
// perform the tick operation for all agents
#pragma omp for schedule(static)
    for (auto &region : regions) {
      region->move_agents(model, this);
    }
  }
}

Ped::Region *Region_handler::add_region(size_t x, size_t y, size_t width,
                                        size_t height) {
  Ped::Region *new_region = new Region(x, y, width, height);
  regions.push_back(new_region);

  return new_region;
}
} // namespace Ped