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
        if (added_region->contains(agent)) {
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

static bool areAdjacent(Region *r1, Region *r2) {
  // Horizontal adjacency
  if (r1->yMin == r2->yMin && r1->yMax == r2->yMax) {
    if (r1->xMax == r2->xMin || r2->xMax == r1->xMin)
      return true;
  }
  // Vertical adjacency
  if (r1->xMin == r2->xMin && r1->xMax == r2->xMax) {
    if (r1->yMax == r2->yMin || r2->yMax == r1->yMin)
      return true;
  }
  return false;
}

static Region *mergeRegions(Region *r1, Region *r2) {
  int newXMin = std::min(r1->xMin, r2->xMin);
  int newXMax = std::max(r1->xMax, r2->xMax);
  int newYMin = std::min(r1->yMin, r2->yMin);
  int newYMax = std::max(r1->yMax, r2->yMax);
  Region *merged = new Region(newXMin, newXMax, newYMin, newYMax);

  // Redistribute agents from r1.
  auto curr = std::atomic_load_explicit(&r1->agent_list_head,
                                        std::memory_order_acquire);
  while (curr) {
    Tagent *agent = curr->agent;
    if (agent) {
      merged->addAgent(agent);
    }
    curr = std::atomic_load_explicit(&curr->next, std::memory_order_acquire);
  }
  // And from r2.
  curr = std::atomic_load_explicit(&r2->agent_list_head,
                                   std::memory_order_acquire);
  while (curr) {
    Tagent *agent = curr->agent;
    if (agent) {
      merged->addAgent(agent);
    }
    curr = std::atomic_load_explicit(&curr->next, std::memory_order_acquire);
  }

  return merged;
}

void Region_handler::resize_regions() {
  std::vector<Region *> new_regions;

  // First pass: for each region, if overcrowded, split it.
  for (auto region : regions) {
    if (region->agentCount.load(std::memory_order_relaxed) > max_agents) {

      // Compute midpoints.
      int midX = (region->xMin + region->xMax) / 2;
      int midY = (region->yMin + region->yMax) / 2;

      // Create four subregions.
      Region *r1 = new Region(region->xMin, midX, region->yMin, midY);
      Region *r2 = new Region(midX, region->xMax, region->yMin, midY);
      Region *r3 = new Region(region->xMin, midX, midY, region->yMax);
      Region *r4 = new Region(midX, region->xMax, midY, region->yMax);

      // Redistribute agents from the old region.
      auto curr = std::atomic_load_explicit(&region->agent_list_head,
                                            std::memory_order_acquire);
      while (curr) {
        Tagent *agent = curr->agent;
        if (agent) {
          if (r1->contains(agent))
            r1->addAgent(agent);
          else if (r2->contains(agent))
            r2->addAgent(agent);
          else if (r3->contains(agent))
            r3->addAgent(agent);
          else if (r4->contains(agent))
            r4->addAgent(agent);
          else
            r1->addAgent(agent);
        }
        curr =
            std::atomic_load_explicit(&curr->next, std::memory_order_acquire);
      }

      new_regions.push_back(r1);
      new_regions.push_back(r2);
      new_regions.push_back(r3);
      new_regions.push_back(r4);

      delete region;
    } else {
      // Region remains as is.
      new_regions.push_back(region);
    }
  }

  // Second pass: merge regions that are underpopulated.
  std::vector<Region *> merged_regions;
  std::vector<bool> merged(new_regions.size(), false);

  for (size_t i = 0; i < new_regions.size(); i++) {
    if (merged[i])
      continue;
    Region *current = new_regions[i];
    // If current region is underpopulated, try to merge it with adjacent
    // underpopulated regions.
    if (current->agentCount.load(std::memory_order_relaxed) < min_agents) {
      for (size_t j = i + 1; j < new_regions.size(); j++) {
        if (!merged[j] && new_regions[j]->agentCount.load(
                              std::memory_order_relaxed) < min_agents) {
          if (areAdjacent(current, new_regions[j])) {
            Region *mergedRegion = mergeRegions(current, new_regions[j]);
            // Mark region j as merged.
            merged[j] = true;
            // Update current to the newly merged region.
            current = mergedRegion;
          }
        }
      }
    }
    merged_regions.push_back(current);
  }

  regions = merged_regions;
}

Region *Region_handler::next_region_for_agent(Tagent *agent) {

  for (auto &region : regions) {
    if (region->contains_desired(agent)) {
      return region;
    }
  }

  return nullptr;
}

void Region_handler::tick_regions(Model *model) {
#pragma omp parallel num_threads(12)
  {
// perform the tick operation for all agents
#pragma omp for schedule(static)
    for (auto &region : regions) {
      region->move_agents(model, this);
    }
  }

  resize_regions();
}

Ped::Region *Region_handler::add_region(size_t x, size_t y, size_t width,
                                        size_t height) {
  Ped::Region *new_region = new Region(x, x + width, y, y + height);
  regions.push_back(new_region);

  return new_region;
}
} // namespace Ped