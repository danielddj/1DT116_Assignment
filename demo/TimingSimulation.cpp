#include "TimingSimulation.h"

using namespace std;

TimingSimulation::TimingSimulation(Ped::Model &model_, int maxSteps) : Simulation(model_, maxSteps)
{
}

void TimingSimulation::runSimulation()
{
    if (model.getImplementation() == Ped::CUDA)
    {
        model.start_cuda();
        tickCounter = maxSimulationSteps;
    } else {
    
    for (int i = 0; i < maxSimulationSteps; i++)
    {
        tickCounter++;
        model.tick();
    }
    }
}
