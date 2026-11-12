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

#ifndef AUTO_EXPOSURE_CB_H
#define AUTO_EXPOSURE_CB_H

// Shared between AutoExposurePass.cpp, AutoExposure.hlsl and Tonemapping.hlsl.

#define AUTO_EXPOSURE_HISTOGRAM_BINS 256
#define AUTO_EXPOSURE_GROUP_SIZE     16 // Histogram thread group edge; the group must have at least one thread per bin.

// Scene luminance that is mapped to middle grey: exposure = AUTO_EXPOSURE_KEY / adaptedLuminance.
#define AUTO_EXPOSURE_KEY 0.18f

struct AutoExposureConstants
{
    uint2 sourceSize;
    float minLogLuminance;     // log2 of the darkest luminance the histogram resolves
    float logLuminanceRange;   // log2(maxLuminance) - minLogLuminance

    float recipLogLuminanceRange;
    float lowPercentile;       // Fraction of the pixels, counted from the dark end, excluded from the metering
    float highPercentile;      // Fraction of the pixels, counted from the dark end, up to which the metering extends
    float adaptationSpeedUp;   // Rate toward a brighter target: the remaining EV gap halves every 1/speed seconds

    float adaptationSpeedDown; // Rate toward a darker target
    float frameTime;
    uint resetAdaptation;      // Snap to the metered luminance instead of adapting toward it
    uint padding0;
};

#endif
