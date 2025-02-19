#include "ped_region_handler.h"
#include <iostream>
#include <math.h>

namespace Ped {
Region_handler::Region_handler(size_t start_regions, bool resize, size_t max_x,
                               size_t max_y, size_t max_agents,
                               size_t min_agents)
    : dynamic_resize(resize), max_x(max_x), max_y(max_y),
      max_agents(max_agents), min_agents(min_agents) {

  valid_region_count(start_regions);

  size_t grid_dim = static_cast<size_t>(
      std::ceil(std::sqrt(static_cast<double>(start_regions))));

  // Calculate each region's approximate width and height.
  size_t region_width = max_x / grid_dim;
  size_t region_height = max_y / grid_dim;

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

      add_region(x, y, width, height);
    }
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

void Region_handler::tick_regions(Ped::Model *model) {
#pragma omp parallel num_threads(numberOfThreads) shared(agents)
  {
// perform the tick operation for all agents
#pragma omp for schedule(static)
    for (auto &region : regions) {
      region->move_agents(model);
    }
  }
}

void Region_handler::add_region(size_t x, size_t y, size_t width,
                                size_t height) {
  regions.push_back(new Region(x, y, width, height));
}

} // namespace Ped