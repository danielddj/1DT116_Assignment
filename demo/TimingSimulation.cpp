#include "TimingSimulation.h"

using namespace std;

TimingSimulation::TimingSimulation(Ped::Model &model_, int maxSteps)
    : Simulation(model_, maxSteps) {}

void TimingSimulation::runSimulation() {
  if (model.getImplementation() == Ped::CUDA) {
    tickCounter = model.start_cuda(maxSimulationSteps, false, NULL, true);
  } else {

    for (int i = 0; i < maxSimulationSteps; i++) {
      tickCounter++;
      model.tick();
    }
  }
}
