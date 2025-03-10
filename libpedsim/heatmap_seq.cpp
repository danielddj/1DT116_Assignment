// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"

#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// Sets up the heatmap
void Ped::Model::setupHeatmapSeq()
{
	int *hm = (int*)calloc(SIZE*SIZE, sizeof(int));
	int *shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
	int *bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));

	heatmap = (int**)malloc(SIZE*sizeof(int*));

	scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
	blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

	for (int i = 0; i < SIZE; i++)
	{
		heatmap[i] = hm + SIZE*i;
	}
	for (int i = 0; i < SCALED_SIZE; i++)
	{
		scaled_heatmap[i] = shm + SCALED_SIZE*i;
		blurred_heatmap[i] = bhm + SCALED_SIZE*i;
	}
}
#include <chrono> // for high resolution timing

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq()
{
    using clock = std::chrono::steady_clock;

    //-------------------------------------
    // 1) Fade
    //-------------------------------------
    {
        auto start = clock::now();

        for (int x = 0; x < SIZE; x++)
        {
            for (int y = 0; y < SIZE; y++)
            {
                // heat fades
                heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
            }
        }

        auto end = clock::now();
        float elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalFadeTime += elapsedMs;
        std::cout << "Seq Fade time (ms): " << elapsedMs << std::endl;
    }

    //-------------------------------------
    // 2) Agent additions
    //-------------------------------------
    {
        auto start = clock::now();

        // Count how many agents want to go to each location
        for (int i = 0; i < agents.size(); i++)
        {
            Ped::Tagent* agent = agents[i];
            int x = agent->getDesiredX();
            int y = agent->getDesiredY();

            if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
            {
                continue;
            }
            // intensify heat
            heatmap[y][x] += 40;
        }

        auto end = clock::now();
        float elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalAddAgentsTime += elapsedMs;
        std::cout << "Seq AddAgents time (ms): " << elapsedMs << std::endl;
    }

    //-------------------------------------
    // 3) Clamp
    //-------------------------------------
    {
        auto start = clock::now();

        for (int x = 0; x < SIZE; x++)
        {
            for (int y = 0; y < SIZE; y++)
            {
                heatmap[y][x] = (heatmap[y][x] < 255) ? heatmap[y][x] : 255;
            }
        }

        auto end = clock::now();
        float elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalClampTime += elapsedMs;
        std::cout << "Seq Clamp time (ms): " << elapsedMs << std::endl;
    }

    //-------------------------------------
    // 4) Scale
    //-------------------------------------
    {
        auto start = clock::now();

        // Scale the data for visual representation
        for (int y = 0; y < SIZE; y++)
        {
            for (int x = 0; x < SIZE; x++)
            {
                int value = heatmap[y][x];
                for (int cellY = 0; cellY < CELLSIZE; cellY++)
                {
                    for (int cellX = 0; cellX < CELLSIZE; cellX++)
                    {
                        scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
                    }
                }
            }
        }

        auto end = clock::now();
        float elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalScaleTime += elapsedMs;
        std::cout << "Seq Scale time (ms): " << elapsedMs << std::endl;
    }

    //-------------------------------------
    // 5) Blur
    //-------------------------------------
    {
        auto start = clock::now();

        // Weights for blur filter
        const int w[5][5] = {
            { 1, 4, 7, 4, 1 },
            { 4, 16, 26, 16, 4 },
            { 7, 26, 41, 26, 7 },
            { 4, 16, 26, 16, 4 },
            { 1, 4, 7, 4, 1 }
        };

        #define WEIGHTSUM 273

        // Apply gaussian blur filter
        for (int i = 2; i < SCALED_SIZE - 2; i++)
        {
            for (int j = 2; j < SCALED_SIZE - 2; j++)
            {
                int sum = 0;
                for (int k = -2; k < 3; k++)
                {
                    for (int l = -2; l < 3; l++)
                    {
                        sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
                    }
                }
                int value = sum / WEIGHTSUM;
                blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
            }
        }

        auto end = clock::now();
        float elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        totalBlurTime += elapsedMs;
        std::cout << "Seq Blur time (ms): " << elapsedMs << std::endl;
    }

    // Count that we've done one sequential heatmap update
    heatmapTickCount++;
}
