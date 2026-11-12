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

#ifndef TONEMAPPING_CB_H
#define TONEMAPPING_CB_H

// Constants consumed only by the accumulation + tonemapping pass. Kept separate from
// GlobalConstants so the ray tracing permutations are not rebuilt or re-padded when post-processing changes.
struct TonemappingConstants
{
    float exposureScale; // EV compensation (as a multiplier) on top of the exposure derived from the adapted luminance
    uint toneMappingOperator;
    uint clamp;
    uint enableAccumulation;

    float recipAccumulatedFrames;
    float reinhardRecipWhitePointSquared; // 1 / Lwhite^2 of the extended Reinhard operator; 0 gives the simple L / (1 + L) curve
    uint agxLook;                         // AgxLook: 0 = Default, 1 = Golden, 2 = Punchy
    float padding0;
};

#endif
