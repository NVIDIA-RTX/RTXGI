#pragma once
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef GLOBAL_CB_H
#define GLOBAL_CB_H

struct GlobalConstants
{
    int enableJitter;
    int sharcDebug;
    int enableBackFaceCull;
    int bouncesMax;

    // Keep the scalar count above nrdHitDistanceParams a multiple of four: HLSL aligns float4 to 16 bytes, C++ does not.
    int frameIndex;
    float throughputThreshold;
    float padding0;
    float padding1;

    int enableEmissives;
    int enableLighting;
    int enableTransmission;
    int enableOcclusion;

    int enableAbsorption;
    int enableTransparentShadows;
    int enableSoftShadows;
    int enableRussianRoulette;

    int samplesPerPixel;
    int targetLight;
    uint debugOutputMode;
    float intensityScale;

    uint nrcEnableTerminationHeuristic;
    uint nrcSkipDeltaVertices;
    uint sharcMaterialDemodulation;
    float nrcTerminationHeuristicThreshold;

    float4 nrdHitDistanceParams;

    float roughnessMin;
    float roughnessMax;
    float metalnessMin;
    float metalnessMax;

    float transmissionIor;
    float3 padding2;
};

#define EXIT_MAX_BOUNCE       0
#define EXIT_HIT_SKY          1
#define EXIT_RUSSIAN_ROULETTE 2
#define EXIT_WITHIN_SURFACE   3
#define EXIT_SMALL_THROUGHPUT 4

// Set to 1 to over-ride the UI
#define USE_DEFAULT_SETTINGS 0

// Shadow rays traverse inline with RayQuery (1) or through the shadow hit group (0). Shared with the host so that
// initialization fails on a device without inline ray query support instead of loading shaders that require it.
#define SHADOW_RAY_USE_RAY_QUERY 1

#endif